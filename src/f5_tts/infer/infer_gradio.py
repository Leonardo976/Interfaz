import re 
import tempfile
import os
import json
import time
import glob
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from num2words import num2words
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import soundfile as sf
import torchaudio
from pydub import AudioSegment
import librosa
import whisper_timestamped
import datetime
from prosody import modify_prosody
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from huggingface_hub import hf_hub_download
import uuid  # <--- Importación añadida
import pyrubberband as pyrb  # <--- Importación añadida

import torchvision
torchvision.disable_beta_transforms_warning()

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'temp_uploads'
GENERATED_AUDIO_FOLDER = 'generated_audios'
SPEECH_TYPES_FILE = 'speech_types.json'

# Crear las carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_AUDIO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_AUDIO_FOLDER'] = GENERATED_AUDIO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = None

try:
    vocoder = load_vocoder()
    F5TTS_model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4
    )
    model_path = hf_hub_download(repo_id="jpgallegoar/F5-Spanish", filename="model_1200000.safetensors")
    F5TTS_ema_model = load_model(
        DiT, 
        F5TTS_model_cfg, 
        model_path
    )
    logger.info("Modelos cargados exitosamente.")
except Exception as e:
    logger.exception(f"Error al cargar los modelos: {str(e)}")
    raise

speech_types_dict = {}

def save_speech_types():
    try:
        with open(SPEECH_TYPES_FILE, 'w', encoding='utf-8') as f:
            json.dump(speech_types_dict, f, ensure_ascii=False, indent=4)
        logger.info(f"Archivo JSON guardado exitosamente: {SPEECH_TYPES_FILE}")
    except Exception as e:
        logger.error(f"Error al guardar archivo JSON: {str(e)}")
        raise

def transcribe_audio_with_timestamps(audio_path, language='es'):
    try:
        audio = whisper_timestamped.load_audio(audio_path)
        # Usamos el modelo openai/whisper-large-v2
        model = whisper_timestamped.load_model("openai/whisper-large-v2", device="cpu")
        result = whisper_timestamped.transcribe(model, audio, language=language)

        formatted_transcript = ""
        for segment in result['segments']:
            for word in segment['words']:
                formatted_time = f"({word['start']:.2f})"
                formatted_transcript += f"{formatted_time} {word['text']} "
        
        return formatted_transcript.strip()
    except Exception as e:
        logger.exception(f"Error en la transcripción: {str(e)}")
        return None

def load_speech_types():
    global speech_types_dict
    try:
        if os.path.exists(SPEECH_TYPES_FILE):
            with open(SPEECH_TYPES_FILE, 'r', encoding='utf-8') as f:
                speech_types_dict = json.load(f)
            logger.info(f"Tipos de habla cargados: {speech_types_dict}")
        else:
            speech_types_dict = {}
            logger.info("No existe archivo de tipos de habla, se iniciará uno nuevo")
    except Exception as e:
        logger.error(f"Error al cargar tipos de habla: {str(e)}")
        speech_types_dict = {}

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

def generate_response(messages, model, tokenizer):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def traducir_numero_a_texto(texto):
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')
    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
    return texto_traducido

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'WAV', 'MP3', 'OGG', 'M4A'}
    return '.' in filename and filename.rsplit('.', 1)[1].upper() in ALLOWED_EXTENSIONS

@gpu_decorator
def infer(
    ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1
):
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)

        if not gen_text.endswith(". "):
            gen_text += ". "

        gen_text = gen_text.lower()
        gen_text = traducir_numero_a_texto(gen_text)

        final_wave, final_sample_rate, combined_spectrogram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            speed=speed
        )

        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(combined_spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        logger.exception(f"Error en infer: {str(e)}")
        raise

@app.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    try:
        data = request.get_json()
        audio_path = data.get('audio_path', '')
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'success': False, 'error': 'audio_path no válido'}), 400
        
        transcription = transcribe_audio_with_timestamps(audio_path, language='es')
        if transcription is None:
            return jsonify({'success': False, 'error': 'Error en transcripción'}), 500
        
        return jsonify({
            'success': True,
            'transcription': transcription
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def parse_speechtypes_text(gen_text):
    pattern = r"\{(.*?)\}"
    tokens = re.split(pattern, gen_text)
    segments = []
    current_style = "Regular"
    for i in range(len(tokens)):
        if i % 2 == 0:
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            current_style = tokens[i].strip()
    return segments

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    try:
        logger.info(f"Form data recibida: {request.form.to_dict()}")
        logger.info(f"Files recibidos: {request.files.keys()}")

        if 'audio' not in request.files:
            logger.error("No se encontró el archivo de audio en la solicitud")
            return jsonify({'error': 'No se proporcionó archivo de audio'}), 400
        
        file = request.files['audio']
        logger.info(f"Archivo recibido: {file.filename}")
        
        speech_type = request.form.get('speechType', 'Regular')
        ref_text = request.form.get('refText', '')
        
        logger.info(f"Tipo de habla: {speech_type}")
        logger.info(f"Texto de referencia: {ref_text}")
        
        if file.filename == '':
            logger.error("Nombre de archivo vacío")
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Tipo de archivo no permitido: {file.filename}")
            return jsonify({'error': 'Tipo de archivo no permitido'}), 400
        
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        timestamp = int(time.time())
        filename = secure_filename(f"{speech_type}_{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            logger.info(f"Archivo guardado en: {filepath}")
        except Exception as e:
            logger.error(f"Error al guardar el archivo: {str(e)}")
            return jsonify({'error': f'Error al guardar el archivo: {str(e)}'}), 500
        
        if not os.path.exists(filepath):
            logger.error("El archivo no se guardó correctamente")
            return jsonify({'error': 'Error al guardar el archivo'}), 500
        
        try:
            speech_types_dict[speech_type] = {
                'audio': filepath,
                'ref_text': ref_text
            }
            logger.info(f"Diccionario actualizado: {speech_types_dict}")
            
            with open(SPEECH_TYPES_FILE, 'w', encoding='utf-8') as f:
                json.dump(speech_types_dict, f, ensure_ascii=False, indent=4)
            logger.info("Tipos de habla guardados en JSON")
            
        except Exception as e:
            logger.error(f"Error al actualizar tipos de habla: {str(e)}")
            return jsonify({'error': f'Error al actualizar tipos de habla: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'filepath': filepath,
            'speechType': speech_type,
            'message': f'Tipo de habla {speech_type} guardado correctamente'
        })
        
    except Exception as e:
        logger.exception(f"Error general en upload_audio: {str(e)}")
        return jsonify({'error': f'Error al procesar la solicitud: {str(e)}'}), 500

@app.route('/api/generate_multistyle_speech', methods=['POST'])
def generate_multistyle_speech():
    try:
        data = request.json
        gen_text = data.get('gen_text', 'Este es un texto por defecto para generar audio.')
        remove_silence = data.get('remove_silence', False)
        cross_fade_duration = data.get('cross_fade_duration', 0.15)
        speed = data.get('speed_change', 1.0)
        ref_text_overrides = data.get('ref_text_overrides', {})
        just_audio = data.get('just_audio', False)

        if not gen_text:
            logger.error('gen_text es requerido')
            return jsonify({'error': 'gen_text es requerido'}), 400
    
        segments = parse_speechtypes_text(gen_text)
        logger.info(f"Segmentos obtenidos: {segments}")
        
        if "Regular" not in speech_types_dict:
            return jsonify({'error': 'No existe tipo de habla Regular configurado.'}), 400

        for segment in segments:
            style = segment["style"]
            if style not in speech_types_dict:
                logger.error(f'Tipo de habla no encontrado: {style}')
                return jsonify({'error': f'Tipo de habla no encontrado: {style}'}), 400
            
            ref_audio = speech_types_dict[style]['audio']
            if not os.path.exists(ref_audio):
                logger.error(f'Archivo de audio no encontrado para {style}: {ref_audio}')
                return jsonify({'error': f'Archivo de audio no encontrado para {style}: {ref_audio}'}), 404

        generated_audio_segments = []
        sample_rate = None

        for segment in segments:
            style = segment["style"]
            text = segment["text"]
            
            speech_type_data = speech_types_dict[style]
            ref_audio = speech_type_data['audio']
            ref_text_original = speech_type_data.get('ref_text', '')
            ref_text = ref_text_original

            audio, _ = infer(
                ref_audio,
                ref_text,
                text,
                F5TTS_ema_model,
                remove_silence,
                cross_fade_duration=cross_fade_duration,
                speed=speed
            )
            
            if sample_rate is None:
                sample_rate = audio[0]
            
            generated_audio_segments.append(audio[1])
            logger.info(f"Segmento generado para {style} guardado.")

        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            
            # Generar una ruta única para el audio generado
            generated_audio_filename = f"multi_style_{uuid.uuid4().hex}.wav"
            generated_audio_path = os.path.join(app.config['GENERATED_AUDIO_FOLDER'], generated_audio_filename)
            
            sf.write(generated_audio_path, final_audio_data, sample_rate)
            
            logger.info(f"Audio final multi-estilo guardado en: {generated_audio_path}")
            return jsonify({
                'success': True,
                'audio_path': generated_audio_path
            })
        else:
            logger.error('No se generó audio')
            return jsonify({'error': 'No se generó audio'}), 400

    except Exception as e:
        logger.exception(f'Error en generación multi-estilo: {str(e)}')
        return jsonify({'error': f'Error en generación multi-estilo: {str(e)}'}), 500

@app.route('/api/generate_timestamps_from_audio', methods=['POST'])
def generate_timestamps_from_audio():
    try:
        data = request.json
        audio_path = data.get('audio_path')

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'audio_path no válido'}), 400

        transcription = transcribe_audio_with_timestamps(audio_path)
        logger.info(f"Transcripción generada: {transcription}")

        final_text = (
            f"INFO:__main__:Audio final multi-estilo guardado en: {audio_path}\n"
            f"INFO:__main__:Transcripción generada: {transcription}"
        )

        return jsonify({
            'audio_path': audio_path,
            'transcription': final_text
        })
    except Exception as e:
        logger.exception(f'Error al generar marcas de tiempo desde audio: {str(e)}')
        return jsonify({'error': f'Error al generar marcas de tiempo: {str(e)}'}), 500

@app.route('/api/modify_prosody', methods=['POST'])
def modify_prosody_route():
    data = request.json
    audio_path = data.get('audio_path')
    modifications = data.get('modifications', [])

    # Renombrar pitch_shift a pitch_semitones internamente al pasar las modificaciones
    new_mods = []
    for m in modifications:
        m['pitch_semitones'] = m.pop('pitch_shift', 0)
        m['pitch_shift'] = m.pop('pitch_semitones', 0)
        new_mods.append(m)

    modifications = new_mods

    if not audio_path or not os.path.exists(audio_path):
        return jsonify({'error': 'audio_path no válido'}), 400

    try:
        # Generar una ruta única para el audio modificado
        modified_audio_filename = f"modified_{uuid.uuid4().hex}.wav"
        modified_audio_path = os.path.join(app.config['GENERATED_AUDIO_FOLDER'], modified_audio_filename)

        # Llamar a la función de modificación de prosodia
        modify_prosody(
            audio_path=audio_path,
            modifications=modifications,
            output_path=modified_audio_path
        )
        
        return jsonify({'success': True, 'output_audio_path': modified_audio_path}), 200
    except Exception as e:
        logger.exception(f'Error al modificar la prosodia: {e}')
        return jsonify({'success': False, 'message': f'Error al modificar la prosodia: {e}'}), 500

@app.route('/api/delete_audio', methods=['POST'])
def delete_audio():
    """
    Endpoint para eliminar un archivo de audio generado.
    Recibe JSON con 'audio_path'.
    """
    data = request.get_json()
    audio_path = data.get('audio_path', '')

    if not audio_path:
        return jsonify({'success': False, 'message': 'No se proporcionó la ruta del audio.'}), 400

    # Asegurar que la ruta es segura y pertenece a GENERATED_AUDIO_FOLDER
    secure_path = secure_filename(os.path.basename(audio_path))
    full_path = os.path.join(app.config['GENERATED_AUDIO_FOLDER'], secure_path)

    if not os.path.exists(full_path):
        return jsonify({'success': False, 'message': 'El archivo de audio no existe.'}), 404

    try:
        os.remove(full_path)
        logger.info(f"Audio eliminado: {full_path}")
        return jsonify({'success': True, 'message': 'Audio eliminado correctamente.'}), 200
    except Exception as e:
        logger.exception(f"Error al eliminar el audio: {e}")
        return jsonify({'success': False, 'message': f'Error al eliminar el audio: {str(e)}'}), 500

@app.route('/api/get_audio/<path:filename>')
def get_audio(filename):
    try:
        # Verificar que el archivo existe en GENERATED_AUDIO_FOLDER
        secure_path = secure_filename(os.path.basename(filename))
        full_path = os.path.join(app.config['GENERATED_AUDIO_FOLDER'], secure_path)
        if not os.path.exists(full_path):
            logger.error(f"Archivo de audio no encontrado: {full_path}")
            return jsonify({'error': 'Archivo de audio no encontrado.'}), 404
        return send_file(full_path, mimetype='audio/wav')
    except Exception as e:
        logger.exception(f"Error al servir archivo de audio {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 404

@app.route('/api/get_spectrogram/<path:filename>')
def get_spectrogram(filename):
    try:
        # Verificar que el archivo existe
        secure_path = secure_filename(os.path.basename(filename))
        full_path = os.path.join(tempfile.gettempdir(), secure_path)
        if not os.path.exists(full_path):
            logger.error(f"Espectrograma no encontrado: {full_path}")
            return jsonify({'error': 'Espectrograma no encontrado.'}), 404
        return send_file(full_path, mimetype='image/png')
    except Exception as e:
        logger.exception(f"Error al servir espectrograma {filename}: {str(e)}")
        return jsonify({'error': str(e)}), 404

@app.route('/api/get_speech_types', methods=['GET'])
def get_speech_types():
    try:
        speech_types = list(speech_types_dict.keys())
        logger.info(f"Tipos de habla solicitados: {speech_types}")
        return jsonify(speech_types)
    except Exception as e:
        logger.exception(f"Error al obtener tipos de habla: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_temp_files():
    try:
        # Limpiar archivos en UPLOAD_FOLDER que no se han modificado en la última hora
        temp_files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
        for f in temp_files:
            try:
                if os.path.isfile(f) and os.path.getmtime(f) < time.time() - 3600:
                    os.remove(f)
                    logger.info(f"Archivo temporal eliminado: {f}")
            except Exception as e:
                logger.error(f"Error al limpiar el archivo {f}: {e}")

        # Limpiar audios generados que no se han modificado en la última hora
        generated_audio_files = glob.glob(os.path.join(GENERATED_AUDIO_FOLDER, '*.wav'))
        for f in generated_audio_files:
            try:
                if os.path.isfile(f) and os.path.getmtime(f) < time.time() - 3600:
                    os.remove(f)
                    logger.info(f"Archivo generado eliminado: {f}")
            except Exception as e:
                logger.error(f"Error al limpiar el archivo {f}: {e}")
    except Exception as e:
        logger.error(f"Error en cleanup_temp_files: {e}")

if __name__ == '__main__':
    try:
        load_speech_types()
    except Exception as e:
        logger.exception(f"Error al cargar tipos de habla: {str(e)}")

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_temp_files, trigger="interval", hours=1)
    scheduler.start()
    logger.info("Scheduler de limpieza iniciado.")

    atexit.register(lambda: scheduler.shutdown())
    app.run(host='0.0.0.0', port=5000, debug=True)