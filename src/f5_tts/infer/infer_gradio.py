from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import re
import tempfile
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer
from num2words import num2words
import os
import json
from werkzeug.utils import secure_filename
from pydub import AudioSegment  # Importar pydub para la conversión

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# Importar módulos específicos de TTS
from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

app = Flask(__name__)
CORS(app)

# Agregar estas líneas para permitir archivos grandes
app.config['MAX_CONTENT_LENGTH'] = None  # Elimina el límite de tamaño
app.config['MAX_CONTENT_PATH'] = None    # Elimina el límite de ruta

# Configurar carpeta de subidas
UPLOAD_FOLDER = 'temp_uploads'
SPEECH_TYPES_FILE = 'speech_types.json'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Inicializar modelos
vocoder = load_vocoder()
F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
F5TTS_ema_model = load_model(
    DiT, 
    F5TTS_model_cfg, 
    str(cached_path("hf://jpgallegoar/F5-Spanish/model_1200000.safetensors"))
)

chat_model_state = None
chat_tokenizer_state = None

# Diccionario global para almacenar los tipos de habla
speech_types_dict = {}

def save_speech_types():
    """Guardar tipos de habla en archivo JSON"""
    try:
        with open(SPEECH_TYPES_FILE, 'w', encoding='utf-8') as f:
            json.dump(speech_types_dict, f, ensure_ascii=False, indent=4)
        print(f"Archivo JSON guardado exitosamente: {SPEECH_TYPES_FILE}")
    except Exception as e:
        print(f"Error al guardar archivo JSON: {str(e)}")
        raise

def load_speech_types():
    """Cargar tipos de habla desde archivo JSON"""
    global speech_types_dict
    try:
        if os.path.exists(SPEECH_TYPES_FILE):
            with open(SPEECH_TYPES_FILE, 'r', encoding='utf-8') as f:
                speech_types_dict = json.load(f)
            print(f"Tipos de habla cargados: {speech_types_dict}")
        else:
            speech_types_dict = {}
            print("No existe archivo de tipos de habla, se iniciará uno nuevo")
    except Exception as e:
        print(f"Error al cargar tipos de habla: {str(e)}")
        speech_types_dict = {}

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

def generate_response(messages, model, tokenizer):
    """Generar respuesta usando Qwen"""
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
    """Traducir números a texto en español"""
    texto_separado = re.sub(r'([A-Za-z])(\d)', r'\1 \2', texto)
    texto_separado = re.sub(r'(\d)([A-Za-z])', r'\1 \2', texto_separado)
    
    def reemplazar_numero(match):
        numero = match.group()
        return num2words(int(numero), lang='es')

    texto_traducido = re.sub(r'\b\d+\b', reemplazar_numero, texto_separado)
    return texto_traducido

@gpu_decorator
def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, cross_fade_duration=0.15, speed=1):
    """Función principal de inferencia para generación TTS"""
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, ref_text)

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
            F5TTS_ema_model,
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

        # Guardar espectrograma
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(combined_spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        print(f"Error en infer: {str(e)}")
        raise

def parse_speechtypes_text(gen_text):
    """Analizar texto con anotaciones de tipo de habla"""
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

def allowed_file(filename):
    """Verificar si la extensión del archivo está permitida"""
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'WAV', 'MP3', 'OGG', 'M4A'}
    return '.' in filename and filename.rsplit('.', 1)[1].upper() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path, output_path):
    """Convierte un archivo de audio a formato WAV."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        print(f"Archivo convertido a WAV: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error al convertir archivo a WAV: {str(e)}")
        raise

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    """Manejar subidas de archivos de audio y generar espectrograma"""
    try:
        # Imprimir todo el contenido del request para debug
        print("Form data recibida:", request.form.to_dict())
        print("Files recibidos:", request.files.keys())

        if 'audio' not in request.files:
            print("No se encontró el archivo de audio en la solicitud")
            return jsonify({'error': 'No se proporcionó archivo de audio'}), 400
        
        file = request.files['audio']
        print(f"Archivo recibido: {file.filename}")
        
        # Obtener datos del formulario con valor por defecto
        speech_type = request.form.get('speechType', 'Regular')  # Valor por defecto 'Regular'
        ref_text = request.form.get('refText', '')
        
        print(f"Tipo de habla: {speech_type}")
        print(f"Texto de referencia: {ref_text}")
        
        if file.filename == '':
            print("Nombre de archivo vacío")
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            print(f"Tipo de archivo no permitido: {file.filename}")
            return jsonify({'error': 'Tipo de archivo no permitido'}), 400
        
        # Crear directorio si no existe
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # Generar nombre de archivo único
        import time
        timestamp = int(time.time())
        filename = secure_filename(f"{speech_type}_{timestamp}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Guardar archivo original
        try:
            file.save(filepath)
            print(f"Archivo guardado en: {filepath}")
        except Exception as e:
            print(f"Error al guardar el archivo: {str(e)}")
            return jsonify({'error': f'Error al guardar el archivo: {str(e)}'}), 500
        
        # Verificar que el archivo se guardó correctamente
        if not os.path.exists(filepath):
            print("El archivo no se guardó correctamente")
            return jsonify({'error': 'Error al guardar el archivo'}), 500
        
        # Convertir a WAV si no lo es
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext != '.wav':
            wav_filename = f"{os.path.splitext(filename)[0]}.wav"
            wav_filepath = os.path.join(app.config['UPLOAD_FOLDER'], wav_filename)
            try:
                wav_filepath = convert_to_wav(filepath, wav_filepath)
                # Opcional: Eliminar el archivo original no WAV para ahorrar espacio
                try:
                    os.remove(filepath)
                    print(f"Archivo original eliminado: {filepath}")
                except Exception as e:
                    print(f"Error al eliminar archivo original: {str(e)}")
            except Exception as e:
                return jsonify({'error': f'Error al convertir archivo a WAV: {str(e)}'}), 500
        else:
            wav_filepath = filepath

        # Generar espectrograma para el audio en WAV
        try:
            waveform, sample_rate = torchaudio.load(wav_filepath)
            waveform = waveform.squeeze().numpy()

            # Generar el espectrograma
            spectrogram_filename = f"{os.path.splitext(filename)[0]}_spectrogram.png"
            spectrogram_path = os.path.join(app.config['UPLOAD_FOLDER'], spectrogram_filename)

            import matplotlib.pyplot as plt
            from matplotlib import mlab

            plt.figure(figsize=(10, 4))
            Pxx, freqs, bins, im = plt.specgram(waveform, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
            plt.axis('off')  # Ocultar ejes

            # Añadir epsilon para evitar log10(0)
            Pxx[Pxx == 0] = 1e-10

            plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Espectrograma guardado en: {spectrogram_path}")
        except Exception as e:
            print(f"Error al generar espectrograma: {str(e)}")
            return jsonify({'error': f'Error al generar espectrograma: {str(e)}'}), 500

        # Actualizar el diccionario de tipos de habla
        try:
            speech_types_dict[speech_type] = {
                'audio': wav_filepath,  # Usar la ruta del WAV
                'ref_text': ref_text,
                'spectrogram': spectrogram_path
            }
            print(f"Diccionario actualizado: {speech_types_dict}")
            
            # Guardar en el archivo JSON
            save_speech_types()
            print("Tipos de habla guardados en JSON")
            
        except Exception as e:
            print(f"Error al actualizar tipos de habla: {str(e)}")
            return jsonify({'error': f'Error al actualizar tipos de habla: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'filepath': wav_filepath,
            'speechType': speech_type,
            'spectrogramPath': spectrogram_path,
            'message': f'Tipo de habla {speech_type} guardado correctamente'
        })
        
    except Exception as e:
        print(f"Error general en upload_audio: {str(e)}")
        return jsonify({'error': f'Error al procesar la solicitud: {str(e)}'}), 500

@app.route('/api/generate_speech', methods=['POST'])
def generate_speech():
    """Generar habla de un solo estilo"""
    try:
        data = request.json
        speech_type = data.get('speechType', 'Regular')
        
        if speech_type not in speech_types_dict:
            return jsonify({'error': f'Tipo de habla no encontrado: {speech_type}'}), 400
        
        speech_type_data = speech_types_dict[speech_type]
        ref_audio = speech_type_data['audio']
        ref_text = speech_type_data.get('ref_text', '')
        gen_text = data.get('gen_text')
        remove_silence = data.get('remove_silence', False)

        # Verificar archivo de audio
        if not os.path.exists(ref_audio):
            return jsonify({'error': f'Archivo de audio no encontrado: {ref_audio}'}), 404

        # Generar habla
        audio, spectrogram_path = infer(
            ref_audio,
            ref_text,
            gen_text,
            "F5-TTS",
            remove_silence
        )

        # Guardar audio generado
        temp_audio_path = tempfile.mktemp(suffix='.wav')
        sf.write(temp_audio_path, audio[1], audio[0])

        # Generar espectrograma para el audio generado
        try:
            waveform, sample_rate_final = torchaudio.load(temp_audio_path)
            waveform = waveform.squeeze().numpy()

            spectrogram_generated_path = tempfile.mktemp(suffix='_spectrogram.png')
            
            import matplotlib.pyplot as plt
            from matplotlib import mlab

            plt.figure(figsize=(10, 4))
            Pxx, freqs, bins, im = plt.specgram(waveform, NFFT=1024, Fs=sample_rate_final, noverlap=512, cmap='viridis')
            plt.axis('off')  # Ocultar ejes

            # Añadir epsilon para evitar log10(0)
            Pxx[Pxx == 0] = 1e-10

            plt.savefig(spectrogram_generated_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Espectrograma del audio generado guardado en: {spectrogram_generated_path}")
        except Exception as e:
            print(f"Error al generar espectrograma del audio generado: {str(e)}")
            return jsonify({'error': f'Error al generar espectrograma del audio generado: {str(e)}'}), 500

        return jsonify({
            'audio_path': temp_audio_path,
            'spectrogram_path': spectrogram_generated_path
        })

    except Exception as e:
        return jsonify({'error': f'Error al generar habla: {str(e)}'}), 500

@app.route('/api/generate_multistyle_speech', methods=['POST'])
def generate_multistyle_speech():
    """Generar habla multi-estilo y generar espectrograma del audio final"""
    try:
        data = request.json
        gen_text = data.get('gen_text')
        remove_silence = data.get('remove_silence', False)

        # Obtener segmentos
        segments = parse_speechtypes_text(gen_text)
        
        # Verificar disponibilidad de tipos de habla
        for segment in segments:
            style = segment["style"]
            if style not in speech_types_dict:
                return jsonify({'error': f'Tipo de habla no encontrado: {style}'}), 400
            
            # Verificar archivos de audio
            ref_audio = speech_types_dict[style]['audio']
            if not os.path.exists(ref_audio):
                return jsonify({'error': f'Archivo de audio no encontrado para {style}: {ref_audio}'}), 404

        generated_audio_segments = []
        sample_rate = None

        # Generar audio para cada segmento
        for segment in segments:
            style = segment["style"]
            text = segment["text"]
            
            speech_type_data = speech_types_dict[style]
            ref_audio = speech_type_data['audio']
            ref_text = speech_type_data.get('ref_text', '')

            try:
                # Generar audio del segmento
                audio, _ = infer(
                    ref_audio,
                    ref_text,
                    text,
                    "F5-TTS",
                    remove_silence,
                    cross_fade_duration=0  # Ajuste si es necesario
                )
                
                if sample_rate is None:
                    sample_rate = audio[0]
                
                generated_audio_segments.append(audio[1])
            
            except Exception as e:
                return jsonify({'error': f'Error al generar segmento para {style}: {str(e)}'}), 500

        # Combinar segmentos
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            
            # Guardar audio final
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_audio_path, final_audio_data, sample_rate)
            
            # Generar espectrograma para el audio final
            try:
                waveform, sample_rate_final = torchaudio.load(temp_audio_path)
                waveform = waveform.squeeze().numpy()

                spectrogram_final_path = tempfile.mktemp(suffix='_spectrogram.png')
                
                import matplotlib.pyplot as plt
                from matplotlib import mlab

                plt.figure(figsize=(10, 4))
                Pxx, freqs, bins, im = plt.specgram(waveform, NFFT=1024, Fs=sample_rate_final, noverlap=512, cmap='viridis')
                plt.axis('off')  # Ocultar ejes

                # Añadir epsilon para evitar log10(0)
                Pxx[Pxx == 0] = 1e-10

                plt.savefig(spectrogram_final_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"Espectrograma del audio final guardado en: {spectrogram_final_path}")
            except Exception as e:
                print(f"Error al generar espectrograma del audio final: {str(e)}")
                return jsonify({'error': f'Error al generar espectrograma del audio final: {str(e)}'}), 500

            return jsonify({
                'audio_path': temp_audio_path,
                'spectrogram_path': spectrogram_final_path
            })
        else:
            return jsonify({'error': 'No se generó audio'}), 400

    except Exception as e:
        return jsonify({'error': f'Error en generación multi-estilo: {str(e)}'}), 500

@app.route('/api/get_audio/<path:filename>')
def get_audio(filename):
    """Servir archivos de audio generados"""
    try:
        return send_file(filename, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/get_spectrogram/<path:filename>')
def get_spectrogram(filename):
    """Servir espectrogramas generados"""
    try:
        return send_file(filename, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/get_speech_types', methods=['GET'])
def get_speech_types():
    """Obtener lista de tipos de habla disponibles"""
    return jsonify(list(speech_types_dict.keys()))

def cleanup_temp_files():
    """Limpiar archivos temporales periódicamente"""
    import glob
    import time
    
    # Limpiar archivos temporales
    temp_files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
    for f in temp_files:
        try:
            if os.path.isfile(f) and os.path.getmtime(f) < time.time() - 3600:  # Eliminar archivos más antiguos de 1 hora
                os.remove(f)
        except Exception as e:
            print(f"Error al limpiar el archivo {f}: {e}")

if __name__ == '__main__':
    import atexit
    from apscheduler.schedulers.background import BackgroundScheduler
    
    # Cargar tipos de habla existentes
    load_speech_types()
    
    # Programar tarea de limpieza
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_temp_files, trigger="interval", hours=1)
    scheduler.start()
    
    # Registrar limpieza al cerrar
    atexit.register(lambda: scheduler.shutdown())
    
    # Ejecutar aplicación Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
