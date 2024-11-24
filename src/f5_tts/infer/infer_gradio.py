import os
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

# Solución al problema de registro duplicado de CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silencia logs innecesarios
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"  # Limita uso de memoria XLA para evitar conflictos

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

# Importaciones del modelo F5-TTS y utilidades
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Cargar vocoder y modelo preentrenado
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

# Variables iniciales
chat_model_state = None
chat_tokenizer_state = None

# Función para traducir números a texto en español
def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)

    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang="es")

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
    return texto_traducido
# Fase 1: Subida de Audio Inicial
def phase1():
    def accept_audio(audio_path):
        """Acepta el audio y avanza a la siguiente fase."""
        if audio_path:
            return "Audio aceptado. Avanzando a la Fase 2.", gr.update(visible=False), gr.update(visible=True)
        else:
            return "Por favor, sube un audio válido.", gr.update(), gr.update()

    def cancel_audio():
        """Reinicia la fase 1."""
        return "Por favor, sube un audio para continuar.", gr.update(visible=True), gr.update(visible=False)

    with gr.Blocks() as phase1_app:
        gr.Markdown("### Fase 1: Subida de Audio Inicial")
        uploaded_audio = gr.Audio(label="Sube un audio generado por TTS cualquiera", type="filepath")
        accept_button = gr.Button("Aceptar")
        cancel_button = gr.Button("Cancelar")
        status_message = gr.Textbox(label="Estado", value="Sube un audio válido para continuar.", interactive=False)

        # Acciones de los botones
        accept_button.click(
            accept_audio,
            inputs=[uploaded_audio],
            outputs=[status_message, gr.update(visible=False), gr.update(visible=True)],
        )

        cancel_button.click(
            cancel_audio,
            inputs=[],
            outputs=[status_message, gr.update(visible=True), gr.update(visible=False)],
        )

    return phase1_app
# Fase 2: Subida o grabación de audio de referencia
def phase2():
    def accept_reference(audio_path, ref_text):
        """Acepta el audio de referencia y el texto, avanzando a la siguiente fase."""
        if audio_path and ref_text:
            return "Audio y texto aceptados. Avanzando a la Fase 3.", gr.update(visible=False), gr.update(visible=True)
        else:
            return "Por favor, sube un audio de referencia y texto válidos.", gr.update(), gr.update()

    def cancel_reference():
        """Reinicia la fase 2."""
        return "Sube un audio de referencia y texto válidos para continuar.", gr.update(visible=True), gr.update(visible=False)

    with gr.Row(visible=False) as phase2_container:
        gr.Markdown("### Fase 2: Subida de Referencia")
        ref_audio = gr.Audio(label="Sube o graba un audio de referencia", type="filepath")
        ref_text = gr.Textbox(
            label="Texto de referencia (15 segundos de lectura)",
            placeholder="Ejemplo: Hola, este es un texto para clonar la voz.",
        )
        accept_ref_button = gr.Button("Aceptar")
        cancel_ref_button = gr.Button("Cancelar")
        status_message = gr.Textbox(label="Estado", value="", interactive=False)

        # Acciones de los botones
        accept_ref_button.click(
            accept_reference,
            inputs=[ref_audio, ref_text],
            outputs=[status_message, gr.update(visible=False), gr.update(visible=True)],
        )
        cancel_ref_button.click(
            cancel_reference,
            inputs=[],
            outputs=[status_message, gr.update(visible=True), gr.update(visible=False)],
        )

    return phase2_container
# Fase 3: Configuración de Tipos de Habla
def phase3():
    def add_emotion(emotion_name, emotion_audio):
        """Agrega un nuevo tipo de habla."""
        if emotion_name and emotion_audio:
            return f"Tipo de habla '{emotion_name}' agregado.", gr.update()
        return "Error: Proporciona un nombre y audio válidos.", gr.update()

    def delete_emotion(emotion_name):
        """Elimina un tipo de habla existente."""
        if emotion_name:
            return f"Tipo de habla '{emotion_name}' eliminado.", gr.update()
        return "Error: Selecciona un tipo de habla para eliminar.", gr.update()

    with gr.Row(visible=False) as phase3_container:
        gr.Markdown("### Fase 3: Configuración de Tipos de Habla")
        emotion_name = gr.Textbox(label="Nombre de la Emoción")
        emotion_audio = gr.Audio(label="Sube o graba un audio para esta emoción", type="filepath")
        add_button = gr.Button("Agregar Emoción")
        delete_button = gr.Button("Eliminar Emoción")
        status_message = gr.Textbox(label="Estado", value="", interactive=False)

        add_button.click(add_emotion, inputs=[emotion_name, emotion_audio], outputs=[status_message])
        delete_button.click(delete_emotion, inputs=[emotion_name], outputs=[status_message])

    return phase3_container
# Fase 4: Modificación del Texto Transcrito
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
        transcription = gr.Textbox(
            label="Texto Transcrito",
            placeholder="Texto transcrito de referencia.",
            lines=5,
        )
        emotion_dropdown = gr.Dropdown(
            label="Selecciona una emoción",
            choices=["Feliz", "Triste", "Sorprendido", "Enojado", "Regular"],
        )
        text_mark_dropdown = gr.Dropdown(
            label="Selecciona una marca de texto",
            choices=["Velocidad +", "Velocidad -", "Grave", "Agudo", "Silencio"],
        )
        add_button = gr.Button("Agregar al Texto")
        updated_transcription = gr.Textbox(label="Texto Modificado", interactive=False, lines=5)

        add_button.click(
            modify_text,
            inputs=[transcription, emotion_dropdown, text_mark_dropdown],
            outputs=updated_transcription,
        )

    return phase4_container
# Fase 5: Inferencia y Clonación de Voz
def phase5():
    def run_inference(ref_audio, ref_text, gen_text, remove_silence):
        """Ejecuta el proceso de inferencia y genera el audio clonado."""
        if not ref_audio or not ref_text or not gen_text:
            return "Error: Completa todas las fases previas antes de continuar.", None

        try:
            # Llamada a la función de inferencia definida anteriormente
            (sample_rate, generated_audio), spectrogram_path = infer(
                ref_audio,
                ref_text,
                gen_text,
                F5TTS_ema_model,
                remove_silence,
            )
            return (
                "Proceso de inferencia completado. Revisa el audio generado.",
                (sample_rate, generated_audio),
                spectrogram_path,
            )
        except Exception as e:
            return f"Error durante la inferencia: {e}", None, None

    with gr.Row(visible=False) as phase5_container:
        gr.Markdown("### Fase 5: Inferencia y Clonación de Voz")
        remove_silence_checkbox = gr.Checkbox(label="Eliminar silencios durante la inferencia", value=False)
        start_inference_button = gr.Button("Iniciar Inferencia")
        progress_message = gr.Textbox(label="Progreso", interactive=False)

        # Salida de audio generado y espectrograma
        generated_audio = gr.Audio(label="Audio Generado", type="numpy", interactive=False)
        generated_spectrogram = gr.Image(label="Espectrograma Generado", type="filepath", interactive=False)

        start_inference_button.click(
            run_inference,
            inputs=[None, None, None, remove_silence_checkbox],
            outputs=[progress_message, generated_audio, generated_spectrogram],
        )

    return phase5_container
# Control de transiciones entre fases
def next_phase(current_phase):
    """Controla las transiciones entre las fases."""
    if current_phase == 1:
        return (
            gr.update(visible=False),  # Ocultar fase 1
            gr.update(visible=True),   # Mostrar fase 2
            "Fase 2: Subida o grabación de referencia activa.",
            2,                         # Actualizar estado de fase
        )
    elif current_phase == 2:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Fase 3: Configuración de tipos de habla activa.",
            3,
        )
    elif current_phase == 3:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Fase 4: Modificación del texto transcrito activa.",
            4,
        )
    elif current_phase == 4:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Fase 5: Proceso de inferencia activo.",
            5,
        )
    else:
        return (
            gr.update(visible=True),
            gr.update(),
            "Error: No se pudo determinar la fase actual.",
            current_phase,
        )
# Construcción de la aplicación con Gradio
with gr.Blocks() as app:
    gr.Markdown(
        """
        # Spanish-F5: Clonación de Voz Multi-Estilo
        Bienvenido a la herramienta de clonación de voz con F5-TTS. Sigue cada fase para configurar el proceso y generar el audio deseado.
        """
    )

    # Contenedores de cada fase
    phase1_container = phase1()
    phase2_container = phase2()
    phase3_container = phase3()
    phase4_container = phase4()
    phase5_container = phase5()

    # Estado actual de la fase
    current_phase = gr.State(value=1)

    # Botón de transición entre fases
    transition_button = gr.Button("Siguiente Fase")
    transition_message = gr.Textbox(label="Estado", value="Fase 1: Subida de audio inicial activa.", interactive=False)

    transition_button.click(
        next_phase,
        inputs=[current_phase],
        outputs=[phase1_container, phase2_container, phase3_container, phase4_container, transition_message, current_phase],
    )
# Configuración para ejecución en Gradio
@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default="0.0.0.0", help="Host para ejecutar la aplicación")
@click.option("--share", "-s", default=True, is_flag=True, help="Habilitar el enlace público (Gradio live URL).")
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API.")
def main(port, host, share, api):
    """
    Ejecuta la aplicación principal con Gradio y las configuraciones necesarias.
    """
    print("Iniciando la aplicación Spanish-F5...")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,  # Enlace público habilitado
        show_api=api,
    )


if __name__ == "__main__":
    main()
