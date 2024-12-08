# analyze_audio.py

import os
import logging
from flask import Blueprint, request, jsonify
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)
analyze_bp = Blueprint('analyze', __name__)

# Cargar el modelo Whisper una sola vez
# Ajusta el model_path a un modelo soportado, por ejemplo "base.en"
model_path = "base.en"
model = WhisperModel(model_path, device="auto", compute_type="float16")

@analyze_bp.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    """
    Endpoint para analizar el audio generado y obtener transcripción con marcas de tiempo usando faster_whisper.
    Espera un JSON con:
    - audio_path: Ruta al archivo de audio a analizar.
    """
    try:
        data = request.json
        audio_path = data.get('audio_path')

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({'error': 'Ruta de audio inválida o no existe el archivo'}), 400

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True
        )

        segments_list = []
        for seg in segments:
            words_list = []
            if seg.words:
                for w in seg.words:
                    words_list.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end
                    })
            segments_list.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words_list
            })

        full_text = " ".join([s["text"].strip() for s in segments_list]).strip()

        result = {
            "text": full_text,
            "segments": segments_list
        }

        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Error al analizar audio con faster_whisper: {e}")
        return jsonify({'error': str(e)}), 500
