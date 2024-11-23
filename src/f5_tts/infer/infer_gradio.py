import re
import tempfile
import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from num2words import num2words
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

# Modelo de inferencia y vocoder
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, F5TTS_model_cfg, str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

# Variables globales para mantener el estado entre fases
uploaded_tts_audio = gr.State(value=None)
uploaded_reference_audio = gr.State(value=None)
transcription_text = gr.State(value="")
emotions_data = gr.State(value=[])
generated_audio_output = gr.State(value=None)

# Fase 1: Subir archivo generado por TTS
def phase_1_submit(file):
    """
    Validación del archivo TTS subido en la fase 1.
    """
    if file is None:
        return "Por favor, sube un archivo de audio.", gr.update(visible=True)
    uploaded_tts_audio.value = file
    return "Archivo cargado con éxito. Avanzando a la fase 2...", gr.update(visible=False)

def phase_1_cancel():
    """
    Reiniciar la fase 1 si se cancela la operación.
    """
    uploaded_tts_audio.value = None
    return "Carga cancelada. Por favor, sube nuevamente el archivo.", gr.update(visible=True)

# Interfaz para la fase 1
with gr.Blocks() as phase_1:
    gr.Markdown("## Fase 1: Subir archivo generado por TTS")
    tts_audio_input = gr.Audio(label="Sube tu archivo TTS (formato .wav o .mp3)", type="filepath")
    tts_submit = gr.Button("Aceptar")
    tts_cancel = gr.Button("Cancelar")
    tts_feedback = gr.Textbox(visible=False, label="Estado de la carga")

    # Configuración de eventos
    tts_submit.click(phase_1_submit, inputs=tts_audio_input, outputs=[tts_feedback, phase_1])
    tts_cancel.click(phase_1_cancel, outputs=[tts_feedback, phase_1])
# Fase 2: Subir o grabar audio de referencia
def phase_2_submit(audio, example_text):
    """
    Validación del audio de referencia en la fase 2.
    """
    if audio is None:
        return "Por favor, sube o graba un audio de referencia.", gr.update(visible=True)
    uploaded_reference_audio.value = audio
    return "Audio de referencia cargado con éxito. Avanzando a la fase 3...", gr.update(visible=False)

def phase_2_cancel():
    """
    Reiniciar la fase 2 si se cancela la operación.
    """
    uploaded_reference_audio.value = None
    return "Carga cancelada. Por favor, vuelve a intentar.", gr.update(visible=True)

# Interfaz para la fase 2
with gr.Blocks() as phase_2:
    gr.Markdown("## Fase 2: Subir o grabar audio de referencia")
    ref_audio_input = gr.Audio(label="Sube tu archivo de referencia o graba uno", type="filepath")
    example_text = gr.Textbox(
        value="Texto de ejemplo para grabación (diseñado para ~15 segundos de lectura).",
        label="Texto de ejemplo",
        interactive=False,
    )
    ref_submit = gr.Button("Aceptar")
    ref_cancel = gr.Button("Cancelar")
    ref_feedback = gr.Textbox(visible=False, label="Estado de la carga")

    # Configuración de eventos
    ref_submit.click(phase_2_submit, inputs=[ref_audio_input, example_text], outputs=[ref_feedback, phase_2])
    ref_cancel.click(phase_2_cancel, outputs=[ref_feedback, phase_2])
# Fase 3: Tipos de habla
def add_emotion(name, audio, current_emotions):
    """
    Agregar un nuevo tipo de habla/emoción.
    """
    if not name or audio is None:
        return "Debe proporcionar un nombre y un audio para la emoción.", current_emotions
    new_emotion = {"name": name, "audio": audio}
    current_emotions.append(new_emotion)
    return "Emoción agregada con éxito.", current_emotions

def delete_emotion(name, current_emotions):
    """
    Eliminar un tipo de habla/emoción existente.
    """
    current_emotions = [e for e in current_emotions if e["name"] != name]
    return "Emoción eliminada.", current_emotions

# Interfaz para la fase 3
with gr.Blocks() as phase_3:
    gr.Markdown("## Fase 3: Agregar Tipos de Habla")
    emotion_name = gr.Textbox(label="Nombre de la emoción")
    emotion_audio = gr.Audio(label="Audio de la emoción", type="filepath")
    add_emotion_btn = gr.Button("Agregar emoción")
    delete_emotion_btn = gr.Button("Eliminar emoción")
    emotions_list = gr.Dropdown(choices=[], label="Lista de emociones")
    emotion_feedback = gr.Textbox(visible=False, label="Estado de las emociones")

    # Configuración de eventos
    add_emotion_btn.click(
        add_emotion, inputs=[emotion_name, emotion_audio, emotions_data], outputs=[emotion_feedback, emotions_list]
    )
    delete_emotion_btn.click(
        delete_emotion, inputs=[emotions_list, emotions_data], outputs=[emotion_feedback, emotions_list]
    )
# Fase 4: Edición del texto transcrito
def add_emotion_to_text(current_text, emotion_name, position):
    """
    Agregar una emoción al texto en una posición específica.
    """
    return current_text[:position] + f"{{{emotion_name}}}" + current_text[position:]

def add_tag_to_text(current_text, tag, position):
    """
    Agregar una marca de texto (como velocidad o tono) en una posición específica.
    """
    return current_text[:position] + f"{{{tag}}}" + current_text[position:]

# Interfaz para la fase 4
with gr.Blocks() as phase_4:
    gr.Markdown("## Fase 4: Edición del Texto Transcrito")
    transcription_input = gr.Textbox(
        label="Texto Transcrito",
        lines=10,
        placeholder="Modifica aquí el texto transcrito. Ejemplo:\n{Feliz} Hola, ¿cómo estás? {Velocidad +} Me alegra verte."
    )
    emotion_select = gr.Dropdown(label="Selecciona una emoción para agregar al texto", choices=[])
    text_position = gr.Slider(minimum=0, maximum=500, step=1, label="Posición en el texto")
    add_emotion_btn = gr.Button("Agregar Emoción al Texto")
    tag_select = gr.Dropdown(
        label="Selecciona una marca de texto",
        choices=["Velocidad +", "Velocidad -", "Grave", "Agudo", "Silencio"]
    )
    add_tag_btn = gr.Button("Agregar Marca al Texto")
    updated_text_output = gr.Textbox(label="Texto Modificado")

    # Configuración de eventos
    add_emotion_btn.click(
        add_emotion_to_text, 
        inputs=[transcription_input, emotion_select, text_position], 
        outputs=updated_text_output
    )
    add_tag_btn.click(
        add_tag_to_text,
        inputs=[transcription_input, tag_select, text_position],
        outputs=updated_text_output
    )
# Fase 5: Inferencia y generación
def run_inference(tts_audio, ref_audio, text, emotions, progress=gr.Progress()):
    """
    Realiza la inferencia para generar el audio final.
    """
    progress(0.1)
    # Simula el proceso de inferencia con chunks
    generated_segments = []
    for emotion in emotions:
        # Aquí se realizaría la inferencia real para cada segmento.
        generated_segments.append(emotion["audio"])
        progress(0.7)

    # Simula la concatenación de audio generado
    final_audio = np.concatenate([np.random.random(16000) for _ in generated_segments])
    progress(1.0)

    # Simula los espectrogramas
    spectrogram1 = "spectrogram1.png"  # Reemplaza con la generación real
    spectrogram2 = "spectrogram2.png"  # Reemplaza con la generación real

    return final_audio, spectrogram1, spectrogram2

# Interfaz para la fase 5
with gr.Blocks() as phase_5:
    gr.Markdown("## Fase 5: Inferencia y Generación")
    generate_btn = gr.Button("Generar Audio Final")
    generated_audio = gr.Audio(label="Audio Generado")
    spectrogram1 = gr.Image(label="Espectrograma Original (Fase 1)")
    spectrogram2 = gr.Image(label="Espectrograma Modificado (Fase 5)")

    # Configuración de evento de generación
    generate_btn.click(
        run_inference,
        inputs=[uploaded_tts_audio, uploaded_reference_audio, transcription_text, emotions_data],
        outputs=[generated_audio, spectrogram1, spectrogram2]
    )
with gr.Blocks() as app:
    gr.Markdown("# Proceso de Clonación de Voz por Fases")
    current_phase = gr.State(value=1)

    # Fase 1
    phase_1.render()

    # Fase 2
    phase_2.render()

    # Fase 3
    phase_3.render()

    # Fase 4
    phase_4.render()

    # Fase 5
    phase_5.render()

    def next_phase(current_phase_value):
        if current_phase_value < 5:
            return current_phase_value + 1, gr.update(visible=False), gr.update(visible=True)
        return current_phase_value, gr.update(), gr.update()

    # Botones para avanzar entre fases
    next_btn = gr.Button("Siguiente Fase")
    current_phase.change(
        next_phase,
        inputs=current_phase,
        outputs=[current_phase, phase_1, phase_2, phase_3, phase_4, phase_5]
    )
@click.command()
@click.option("--port", "-p", default=None, type=int, help="Puerto para ejecutar la aplicación")
@click.option("--host", "-H", default=None, help="Host para ejecutar la aplicación (por defecto: localhost)")
@click.option(
    "--share",
    "-s",
    default=True,  # Siempre habilitar la opción de compartir
    is_flag=True,
    help="Compartir la aplicación a través de un enlace público de Gradio",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Permitir acceso a la API de la aplicación")
def main(port, host, share, api):
    """
    Ejecuta la aplicación en modo live, asegurando que esté accesible a través de un enlace público.
    """
    print("Iniciando la aplicación...")
    app.queue(api_open=api).launch(
        server_name=host or "0.0.0.0",  # Escuchar en todas las interfaces por defecto
        server_port=port or 7860,  # Usar el puerto 7860 si no se especifica uno
        share=share,  # Compartir el enlace públicamente
        show_api=api,  # Mostrar documentación interactiva de la API
    )


if __name__ == "__main__":
    main()
