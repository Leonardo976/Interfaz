import os
import logging
from flask import Blueprint, request, jsonify
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)
analyze_bp = Blueprint('analyze', __name__)

# Cargar el modelo Whisper de HuggingFace una sola vez
logger.info("Cargando el modelo Whisper large-v2 desde HuggingFace...")
model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
logger.info("Modelo Whisper large-v2 cargado con éxito.")

@analyze_bp.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    """
    Endpoint para analizar el audio generado y obtener transcripción con marcas de tiempo usando HuggingFace Whisper.
    Espera un JSON con:
    - audio_path: Ruta al archivo de audio a analizar.
    """
    try:
        data = request.json
        audio_path = data.get('audio_path')

        if not audio_path or not os.path.exists(audio_path):
            logger.error('Ruta de audio inválida o no existe el archivo')
            return jsonify({'error': 'Ruta de audio inválida o no existe el archivo'}), 400

        logger.info(f"Analizando el archivo de audio: {audio_path}")

        # Cargar el audio y convertirlo a 16 kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # Procesar el audio
        logger.info("Procesando el audio para la transcripción...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Generar la transcripción con los timestamps de las palabras
        logger.info("Generando la transcripción con Whisper...")
        with torch.no_grad():
            generated_output = model.generate(
                inputs.input_features, 
                return_dict_in_generate=True, 
                output_scores=True, 
                max_length=448,
                return_timestamps=True  # 🔥 Importante: return_timestamps=True
            )

        # Decodificar la transcripción
        transcription = processor.batch_decode(generated_output.sequences, skip_special_tokens=True)[0]

        logger.info(f"Transcripción completa: {transcription}")

        # Extraer los tokens y sus timestamps
        logger.info("Obteniendo marcas de tiempo para las palabras...")

        words_with_timestamps = []

        for idx, token_id in enumerate(generated_output.sequences[0]):
            token_text = processor.decode([token_id], skip_special_tokens=True).strip()
            
            # 🔥🔥 Verificar la forma de start_time y end_time 🔥🔥
            start_time = generated_output.scores[idx][0] if idx < len(generated_output.scores) else None
            end_time = generated_output.scores[idx + 1][0] if idx + 1 < len(generated_output.scores) else None

            # 🔥 Convertir los tensores a números reales
            if start_time is not None and isinstance(start_time, torch.Tensor):
                # Seleccionar el valor correcto
                if start_time.dim() > 0:  # Si tiene varias dimensiones
                    start_time = start_time[0].item()  # Toma el primer elemento
                else:
                    start_time = start_time.item()

            if end_time is not None and isinstance(end_time, torch.Tensor):
                # Seleccionar el valor correcto
                if end_time.dim() > 0:  # Si tiene varias dimensiones
                    end_time = end_time[0].item()  # Toma el primer elemento
                else:
                    end_time = end_time.item()

            if start_time is not None and end_time is not None and token_text:
                words_with_timestamps.append({
                    "word": token_text,
                    "start": round(start_time, 2),
                    "end": round(end_time, 2)
                })

        logger.info(f"Timestamps generados para las palabras: {words_with_timestamps}")

        segments_list = []
        words_list = []

        for word_data in words_with_timestamps:
            words_list.append({
                "word": word_data["word"],
                "start": word_data["start"],
                "end": word_data["end"]
            })

        # Agrupar las palabras en segmentos
        if len(words_list) > 0:
            segment = {
                "id": 1,
                "start": round(words_list[0]["start"], 2),
                "end": round(words_list[-1]["end"], 2),
                "text": transcription,
                "words": words_list
            }
            segments_list.append(segment)

        result = {
            "text": transcription,
            "segments": segments_list
        }

        logger.info(f"Transcripción y marcas de tiempo generadas con éxito para {audio_path}")
        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Error al analizar audio con Whisper: {e}")
        return jsonify({'error': str(e)}), 500
