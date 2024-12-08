import os
import logging
from flask import Blueprint, request, jsonify
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logger = logging.getLogger(__name__)
analyze_bp = Blueprint('analyze', __name__)

# Load Whisper model from HuggingFace once
logger.info("Loading Whisper large-v2 model from HuggingFace...")
model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
logger.info("Whisper large-v2 model loaded successfully.")

def generate_dynamic_timestamps(words, audio_duration):
    """
    Generate dynamic timestamps for words based on audio duration
    
    Args:
        words (list): List of words
        audio_duration (float): Total duration of the audio in seconds
    
    Returns:
        list: List of dictionaries with word, start, and end timestamps
    """
    if not words:
        return []

    # Calculate total word length and average word duration
    total_word_length = sum(len(word) for word in words)
    
    # Distribute timestamps proportionally
    words_with_timestamps = []
    current_time = 0
    
    for word in words:
        # Calculate word's proportional duration
        word_ratio = len(word) / total_word_length
        word_duration = word_ratio * audio_duration
        
        # Create timestamp entry
        words_with_timestamps.append({
            "word": word,
            "start": round(current_time, 2),
            "end": round(current_time + word_duration, 2)
        })
        
        current_time += word_duration
    
    return words_with_timestamps

def normalize_text(text):
    """
    Normalize text by removing extra whitespaces and converting to lowercase
    """
    return ' '.join(text.lower().split())

def format_timestamped_text(words_with_timestamps):
    """
    Format words with timestamps in the specified style
    
    Args:
        words_with_timestamps (list): List of dictionaries with word timestamps
    
    Returns:
        str: Formatted text with timestamps
    """
    return ' '.join([f"({w['start']}s) {w['word']}" for w in words_with_timestamps])

@analyze_bp.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    """
    Endpoint to analyze generated audio and get transcription with timestamps using HuggingFace Whisper
    """
    try:
        data = request.json
        audio_path = data.get('audio_path')
        ref_text = data.get('ref_text', '')

        if not audio_path or not os.path.exists(audio_path):
            logger.error('Invalid audio path or file does not exist')
            return jsonify({'error': 'Invalid audio path or file does not exist'}), 400

        logger.info(f"Analyzing audio file: {audio_path}")

        # Load audio and get duration
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = librosa.get_duration(y=audio, sr=sr)

        # Process the audio
        logger.info("Processing audio for transcription...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Generate transcription with word timestamps
        logger.info("Generating transcription with Whisper...")
        with torch.no_grad():
            generated_output = model.generate(
                inputs.input_features, 
                return_dict_in_generate=True, 
                output_scores=True, 
                max_length=448,
                return_timestamps=True
            )

        # Decode the transcription
        transcription = processor.batch_decode(generated_output.sequences, skip_special_tokens=True)[0]
        
        # Prepare processing of words
        normalized_ref_text = normalize_text(ref_text)
        normalized_transcription = normalize_text(transcription)
        
        # Fallback to reference text or transcription
        text_to_use = ref_text if ref_text else transcription
        words = text_to_use.split()

        # Generate timestamps
        words_with_timestamps = generate_dynamic_timestamps(words, audio_duration)

        # Format timestamped text
        formatted_timestamped_text = format_timestamped_text(words_with_timestamps)

        # Prepare result
        result = {
            "original_text": transcription,
            "ref_text": ref_text,
            "audio_duration": round(audio_duration, 2),
            "timestamped_text": formatted_timestamped_text,
            "words_with_timestamps": words_with_timestamps,
            "segments": [{
                "id": 1,
                "start": 0,
                "end": round(audio_duration, 2),
                "text": transcription,
                "words": words_with_timestamps
            }]
        }

        logger.info(f"Transcription and timestamps generated successfully for {audio_path}")
        return jsonify(result), 200

    except Exception as e:
        logger.exception(f"Error analyzing audio with Whisper: {e}")
        return jsonify({'error': str(e)}), 500

# Example usage in a route or function
def example_usage(audio_path, ref_text):
    with app.test_request_context(json={
        'audio_path': audio_path, 
        'ref_text': ref_text
    }):
        response = analyze_audio()
        print(response.json)  # Process the result as needed