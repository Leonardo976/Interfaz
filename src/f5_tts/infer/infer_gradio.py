import re
import tempfile
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from num2words import num2words

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Cargar el vocoder y el modelo de F5-TTS
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

# Función para traducir números a texto
def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang="es")

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)

    return texto_traducido


# Función principal de inferencia
@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1, show_info=gr.Info
):
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=show_info)

    ema_model = F5TTS_ema_model

    if not gen_text.startswith(" "):
        gen_text = " " + gen_text
    if not gen_text.endswith(". "):
        gen_text += ". "

    gen_text = gen_text.lower()
    gen_text = traducir_numero_a_texto(gen_text)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Eliminar silencios, si es necesario
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Guardar espectrograma
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path


# Fase 1: Subida de audio inicial
def phase1():
    def accept_audio(audio_path):
        """Acepta el audio y avanza a la siguiente fase."""
        if audio_path:
            return "Audio aceptado, pasando a la Fase 2", gr.update(visible=False), gr.update(visible=True)
        else:
            return "Por favor, sube un audio válido.", gr.update(), gr.update()

    def cancel_audio():
        """Reinicia la fase 1."""
        return "Sube un audio válido para comenzar.", gr.update(visible=True), gr.update(visible=False)

    with gr.Blocks() as phase1_app:
        gr.Markdown("### Fase 1: Subida de Audio Inicial")
        uploaded_audio = gr.Audio(label="Sube un audio generado por TTS cualquiera", type="filepath")
        accept_button = gr.Button("Aceptar")
        cancel_button = gr.Button("Cancelar")

        # Mensaje de estado
        status_message = gr.Textbox(label="Estado", value="Sube un audio válido para comenzar.", interactive=False)

        # Contenedores de fases
        phase1_container = gr.Row(visible=True)
        phase2_container = gr.Row(visible=False)

        # Acciones de los botones
        accept_button.click(
            accept_audio,
            inputs=[uploaded_audio],
            outputs=[status_message, phase1_container, phase2_container],
        )

        cancel_button.click(
            cancel_audio,
            inputs=[],
            outputs=[status_message, phase1_container, phase2_container],
        )

    return phase1_app


# App principal con integración de fases
with gr.Blocks() as app:
    gr.Markdown("# Spanish-F5 TTS - Multi-Fase")
    phase1_app = phase1()  # Fase 1
    phase1_app.render()
def phase2():
    def accept_reference(audio_path, text):
        """Acepta el audio de referencia y avanza a la siguiente fase."""
        if audio_path and text:
            return (
                "Audio de referencia aceptado. Pasando a la Fase 3.",
                gr.update(visible=False),
                gr.update(visible=True),
            )
        else:
            return "Por favor, sube un audio y texto válidos.", gr.update(), gr.update()

    def cancel_reference():
        """Reinicia la fase 2."""
        return "Sube un audio y texto válidos para continuar.", gr.update(visible=True), gr.update(visible=False)

    with gr.Row(visible=False) as phase2_container:
        gr.Markdown("### Fase 2: Subida o Grabación de Audio de Referencia")

        with gr.Column():
            ref_audio = gr.Audio(label="Sube o graba un audio de referencia", type="filepath")
            ref_text = gr.Textbox(
                label="Texto de ejemplo para referencia (15 segundos de lectura)",
                placeholder="Ejemplo: Hola, este es un texto de referencia para clonar la voz.",
            )

            accept_ref_button = gr.Button("Aceptar")
            cancel_ref_button = gr.Button("Cancelar")

        # Mensaje de estado
        ref_status_message = gr.Textbox(label="Estado", value="", interactive=False)

        # Acciones de los botones
        accept_ref_button.click(
            accept_reference,
            inputs=[ref_audio, ref_text],
            outputs=[ref_status_message, phase2_container, gr.update(visible=True)],
        )
        cancel_ref_button.click(
            cancel_reference,
            inputs=[],
            outputs=[ref_status_message, phase2_container, gr.update(visible=True)],
        )

    return phase2_container

def phase3():
    def add_emotion(emotion_name, emotion_audio):
        """Agrega un nuevo tipo de habla/emoción."""
        if emotion_name and emotion_audio:
            return f"Tipo de habla '{emotion_name}' agregado.", gr.update(), gr.update()
        else:
            return "Por favor, proporciona un nombre y un archivo de audio.", gr.update()

    def delete_emotion(emotion_name):
        """Elimina un tipo de habla."""
        if emotion_name:
            return f"Tipo de habla '{emotion_name}' eliminado.", gr.update()
        else:
            return "Por favor, selecciona un tipo de habla para eliminar.", gr.update()

    with gr.Row(visible=False) as phase3_container:
        gr.Markdown("### Fase 3: Configuración de Tipos de Habla")

        with gr.Column():
            emotion_name = gr.Textbox(label="Nombre de la Emoción")
            emotion_audio = gr.Audio(label="Sube o graba un audio para esta emoción", type="filepath")
            add_emotion_button = gr.Button("Agregar Emoción")
            delete_emotion_button = gr.Button("Eliminar Emoción")

            # Mensaje de estado
            emotion_status_message = gr.Textbox(label="Estado", value="", interactive=False)

        add_emotion_button.click(
            add_emotion, inputs=[emotion_name, emotion_audio], outputs=[emotion_status_message]
        )
        delete_emotion_button.click(delete_emotion, inputs=[emotion_name], outputs=[emotion_status_message])

    return phase3_container
def phase4():
    def modify_text(transcription, emotion_name, text_mark):
        """Modifica el texto transcrito con emociones o marcas de texto."""
        if transcription and emotion_name:
            updated_text = transcription + f"{{{emotion_name}}} "
        elif transcription and text_mark:
            updated_text = transcription + f"{{{text_mark}}} "
        else:
            updated_text = transcription
        return updated_text

    with gr.Row(visible=False) as phase4_container:
        gr.Markdown("### Fase 4: Modificación del Texto Transcrito")

        with gr.Column():
            transcription = gr.Textbox(
                label="Texto Transcrito",
                placeholder="Ejemplo: Hola, este es un texto transcrito de referencia.",
                lines=5,
            )
            emotion_dropdown = gr.Dropdown(
                label="Selecciona una emoción",
                choices=["Feliz", "Triste", "Sorprendido", "Enojado", "Regular"],
                interactive=True,
            )
            text_mark_dropdown = gr.Dropdown(
                label="Selecciona una marca de texto",
                choices=["Velocidad +", "Velocidad -", "Grave", "Agudo", "Silencio"],
                interactive=True,
            )
            add_to_text_button = gr.Button("Agregar al Texto")

            # Salida del texto modificado
            updated_transcription = gr.Textbox(label="Texto Modificado", interactive=False, lines=5)

        add_to_text_button.click(
            modify_text,
            inputs=[transcription, emotion_dropdown, text_mark_dropdown],
            outputs=updated_transcription,
        )

    return phase4_container

def phase5():
    def run_inference(ref_audio, ref_text, gen_text, remove_silence):
        """Ejecuta el proceso de inferencia y clonación de voz."""
        if not ref_audio or not ref_text or not gen_text:
            return "Por favor, asegúrate de haber completado las fases previas."

        try:
            sample_rate, generated_audio = infer(
                ref_audio,
                ref_text,
                gen_text,
                F5TTS_ema_model,
                remove_silence,
            )
            return (
                f"Proceso completado. Audio generado exitosamente.",
                (sample_rate, generated_audio),
            )
        except Exception as e:
            return f"Error durante la inferencia: {e}"

    with gr.Row(visible=False) as phase5_container:
        gr.Markdown("### Fase 5: Inferencia y Clonación de Voz")

        with gr.Column():
            remove_silence_checkbox = gr.Checkbox(label="Eliminar silencios durante la inferencia", value=False)
            start_inference_button = gr.Button("Iniciar Inferencia")
            progress_message = gr.Textbox(label="Progreso", interactive=False)

            # Salida de audio final
            generated_audio = gr.Audio(label="Audio Generado", type="numpy", interactive=False)

        start_inference_button.click(
            run_inference,
            inputs=[ref_audio, ref_text, gen_text_input_multistyle, remove_silence_checkbox],
            outputs=[progress_message, generated_audio],
        )

    return phase5_container
with gr.Blocks() as app:
    gr.Markdown(
        """
        # Spanish-F5: Clonación de Voz Multi-Estilo
        Esta herramienta utiliza el modelo F5-TTS para realizar clonación de voz con múltiples estilos y emociones.
        """
    )

    # Fase 1
    phase1_container = phase1()

    # Fase 2
    phase2_container = phase2()

    # Fase 3
    phase3_container = phase3()

    # Fase 4
    phase4_container = phase4()

    # Fase 5
    phase5_container = phase5()

    # Control de transiciones
    def next_phase(current_phase):
        """Controla las transiciones entre fases."""
        if current_phase == 1:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "Fase 2: Subida o grabación de referencia activa.",
            )
        elif current_phase == 2:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "Fase 3: Configuración de tipos de habla activa.",
            )
        elif current_phase == 3:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "Fase 4: Modificación del texto transcrito activa.",
            )
        elif current_phase == 4:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                "Fase 5: Proceso de inferencia activo.",
            )
        else:
            return (
                gr.update(),
                gr.update(),
                "Error: No se pudo determinar la fase actual.",
            )

    # Botón de transición
    transition_button = gr.Button("Siguiente Fase")
    current_phase = gr.State(value=1)
    transition_message = gr.Textbox(label="Estado", interactive=False)

    transition_button.click(
        next_phase,
        inputs=[current_phase],
        outputs=[
            phase1_container,
            phase2_container,
            transition_message,
        ],
    )
@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default="0.0.0.0", help="Host para ejecutar la aplicación")
@click.option("--share", "-s", default=True, is_flag=True, help="Siempre habilitar el enlace live (Gradio public URL).")
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API")
def main(port, host, share, api):
    """
    Inicia la aplicación Spanish-F5 con Gradio.
    """
    print("Iniciando la aplicación Spanish-F5...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,  # Siempre habilitar el enlace live
        show_api=api,
    )


if __name__ == "__main__":
    main()
