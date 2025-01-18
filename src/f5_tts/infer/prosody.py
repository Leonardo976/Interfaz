import os
import numpy as np
import soundfile as sf
import tempfile
import logging
import subprocess
from pysoundtouch import SoundTouch  # Para velocidad y volumen
import librosa  # Importación añadida para pitch_shift


# Configuración básica del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Versión de librosa: {librosa.__version__}")


def insert_silence(duration, sr):
    """Crea un segmento de silencio de una duración específica."""
    silence = np.zeros(int(duration * sr))
    return silence

def apply_fade(y, sr, fade_duration=0.05):
    """Aplica un fade-in y fade-out al inicio y final del audio."""
    fade_samples = int(fade_duration * sr)
    if fade_samples > len(y):
        fade_samples = len(y)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    y[:fade_samples] *= fade_in
    y[-fade_samples:] *= fade_out
    return y

def crossfade_segments(seg1, seg2, sr, crossfade_duration=0.05):
    """Aplica un crossfade entre dos segmentos de audio."""
    crossfade_samples = int(crossfade_duration * sr)
    if crossfade_samples > len(seg1):
        crossfade_samples = len(seg1)
    if crossfade_samples > len(seg2):
        crossfade_samples = len(seg2)
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)
    seg1_end = seg1[-crossfade_samples:] * fade_out
    seg2_start = seg2[:crossfade_samples] * fade_in
    combined = np.concatenate([
        seg1[:-crossfade_samples],
        seg1_end + seg2_start,
        seg2[crossfade_samples:]
    ])
    return combined

def process_segment(y, sr, speed_change, pitch_shift, volume_change):
    """
    Aplica cambios de velocidad, tono y volumen a un segmento de audio.
    Usa pysoundtouch para velocidad y volumen, y librosa para tono.
    """
    # Ajustar volumen
    if volume_change != 0.0:
        factor = 10 ** (volume_change / 20)
        y = y * factor
        y = np.clip(y, -1.0, 1.0)
    
    # Configurar pysoundtouch para velocidad
    st = SoundTouch(sr, channels=1)
    st.set_rate(speed_change)
    st.put_samples(y)
    result_samples = []
    while True:
        samples = st.receive_samples()
        if len(samples) == 0:
            break
        result_samples.append(samples)
    y_processed = np.concatenate(result_samples) if result_samples else y
    
    # Aplicar cambio de tono usando librosa
    if pitch_shift != 0.0:
        try:
            # Convertimos el audio a float32 y creamos un diccionario de kwargs
            audio_float32 = y_processed.astype(np.float32)
            kwargs = {'sr': sr, 'n_steps': float(pitch_shift)}
            
            # Llamamos a pitch_shift con un solo argumento posicional y el resto como kwargs
            y_processed = librosa.effects.pitch_shift(audio_float32, **kwargs)
            
        except Exception as e:
            logger.error(f"Error aplicando cambio de tono con librosa: {e}")
            # Para debugging, vamos a imprimir más información sobre los argumentos
            logger.error(f"Argumentos usados - audio shape: {y_processed.shape}, dtype: {y_processed.dtype}, sr: {sr}, pitch_shift: {pitch_shift}")
    
    return y_processed

def modify_prosody(
    audio_path,
    modifications,
    remove_silence=False,
    min_silence_len=0.5,
    silence_thresh=-40,
    keep_silence=0.25,
    global_speed_change=1.0,
    global_pitch_change=0.0,        
    cross_fade_duration=0.05,
    output_path=None,
    fade_duration=0.05,
    silence_between_mods=0.02
):
    logger.info(f"Recibiendo modificaciones: {modifications}")
    if not os.path.exists(audio_path):
        logger.error(f"El archivo de audio no existe: {audio_path}")
        return {'success': False, 'message': f"El archivo de audio no existe: {audio_path}"}

    try:
        # Cargar audio usando soundfile en lugar de librosa
        y, sr = sf.read(audio_path)
        # Convertir a mono si es necesario
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        total_duration_sec = len(y) / sr
        logger.info(f"Cargado audio desde {audio_path} con tasa de muestreo {sr} Hz y {total_duration_sec:.2f}s de duración")
    except Exception as e:
        logger.error(f"Error al leer el archivo de audio: {e}")
        return {'success': False, 'message': f"Error al leer el archivo de audio: {e}"}

    if remove_silence:
        try:
            logger.info("Eliminando silencios del audio")
            silent_ranges = detect_silence(y, sr, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
            non_silent_audio = []
            prev_end = 0
            for start, end in silent_ranges:
                non_silent_audio.append(y[prev_end:start])
                non_silent_audio.append(insert_silence(keep_silence, sr))
                prev_end = end
            non_silent_audio.append(y[prev_end:])
            y = np.concatenate(non_silent_audio)
            total_duration_sec = len(y) / sr
            logger.info("Silencios eliminados")
        except Exception as e:
            logger.error(f"Error al eliminar silencios: {e}")
            return {'success': False, 'message': f"Error al eliminar silencios: {e}"}

    try:
        modifications_sorted = sorted(modifications, key=lambda x: x['start_time'])
        logger.info(f"Modificaciones ordenadas: {modifications_sorted}")
    except Exception as e:
        logger.error(f"Error al ordenar las modificaciones: {e}")
        return {'success': False, 'message': f"Error al ordenar las modificaciones: {e}"}

    segments = []
    last_end_sec = 0.0

    for idx, mod in enumerate(modifications_sorted):
        if mod.get('type') == 'silence':
            start_time = mod.get('start_time')
            duration = mod.get('duration', 1.0)
            if start_time is None:
                logger.error(f"Modificación {idx+1}: 'start_time' es necesario para silencios.")
                return {'success': False, 'message': f"Modificación {idx+1}: 'start_time' es necesario para silencios."}
            if start_time < 0 or start_time > total_duration_sec:
                logger.error(f"Modificación {idx+1}: 'start_time' inválido ({start_time} segundos).")
                return {'success': False, 'message': f"Modificación {idx+1}: 'start_time' inválido."}
            start_sample = int(start_time * sr)
            if start_time > last_end_sec:
                segment = y[int(last_end_sec * sr):start_sample]
                segments.append(segment)
                logger.info(f"Agregado segmento sin modificar: {last_end_sec}s a {start_time}s")
            silence_segment = insert_silence(duration, sr)
            segments.append(silence_segment)
            logger.info(f"Insertado silencio de {duration} segundos en la modificación {idx+1}")
            last_end_sec = start_time

        elif mod.get('type') == 'prosody':
            start_time = mod.get('start_time')
            end_time = mod.get('end_time')
            # Usar 'pitch_shift' directamente
            pitch_shift = mod.get('pitch_shift', 0.0)
            volume_change = mod.get('volume_change', 0.0)
            speed_change = mod.get('speed_change', 1.0)

            if start_time < 0 or end_time <= start_time:
                logger.error(f"Modificación {idx+1}: tiempos inválidos (start={start_time}, end={end_time}).")
                return {'success': False, 'message': f"Modificación {idx+1}: tiempos inválidos."}
            if end_time > total_duration_sec:
                logger.warning(f"Modificación {idx+1}: end_time {end_time} excede la duración total. Ajustando a {total_duration_sec}.")
                end_time = total_duration_sec
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            if start_time > last_end_sec:
                segment = y[int(last_end_sec * sr):start_sample]
                segments.append(segment)
                logger.info(f"Agregado segmento sin modificar: {last_end_sec}s a {start_time}s")
            segment = y[start_sample:end_sample]
            segment = process_segment(segment, sr, speed_change, pitch_shift, volume_change)
            try:
                segment = apply_fade(segment, sr, fade_duration=fade_duration)
                logger.info(f"Modificación {idx+1}: Aplicados fades de entrada y salida de {fade_duration}s")
            except Exception as e:
                logger.error(f"Error al aplicar fades en modificación {idx+1}: {e}")
                return {'success': False, 'message': f"Error al aplicar fades en modificación {idx+1}: {e}"}
            if idx < len(modifications_sorted) - 1:
                try:
                    silence = insert_silence(silence_between_mods, sr)
                    segments.append(silence)
                    logger.info(f"Modificación {idx+1}: Insertado silencio de {silence_between_mods}s entre modificaciones")
                except Exception as e:
                    logger.error(f"Error al insertar silencio entre modificaciones: {e}")
                    return {'success': False, 'message': f"Error al insertar silencio entre modificaciones: {e}"}
            segments.append(segment)
            last_end_sec = end_time

    if last_end_sec < total_duration_sec:
        try:
            segment = y[int(last_end_sec * sr):]
            segments.append(segment)
            logger.info(f"Agregado segmento final sin modificar: {last_end_sec}s a {total_duration_sec}s")
        except Exception as e:
            logger.error(f"Error al agregar segmento final sin modificar: {e}")
            return {'success': False, 'message': f"Error al agregar segmento final sin modificar: {e}"}

    if len(segments) == 0:
        logger.error("No se generaron segmentos de audio.")
        return {'success': False, 'message': "No se generaron segmentos de audio."}

    try:
        final_audio = np.concatenate(segments)
        logger.info("Segmentos concatenados sin crossfade")
    except Exception as e:
        logger.error(f"Error al concatenar segmentos: {e}")
        return {'success': False, 'message': f"Error al concatenar segmentos: {e}"}

    if global_speed_change != 1.0 or global_pitch_change != 0.0:
        try:
            final_audio = process_segment(final_audio, sr, global_speed_change, global_pitch_change, 0.0)
            logger.info(f"Aplicando cambio global: velocidad {global_speed_change}x, pitch {global_pitch_change} semitonos")
        except Exception as e:
            logger.error(f"Error al aplicar cambio global: {e}")
            return {'success': False, 'message': f"Error al aplicar cambio global: {e}"}

    try:
        max_val = np.max(np.abs(final_audio))
        if max_val > 1.0:
            final_audio = final_audio / max_val
            logger.info("Normalización aplicada para evitar clipping")
    except Exception as e:
        logger.error(f"Error al normalizar audio: {e}")
        return {'success': False, 'message': f"Error al normalizar audio: {e}"}

    try:
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
    except Exception as e:
        logger.error(f"Error al convertir audio a int16: {e}")
        return {'success': False, 'message': f"Error al convertir audio a int16: {e}"}

    if output_path is None:
        try:
            fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        except Exception as e:
            logger.error(f"Error al crear archivo temporal: {e}")
            return {'success': False, 'message': f"Error al crear archivo temporal: {e}"}

    try:
        sf.write(output_path, final_audio_int16, sr)
        logger.info(f"Audio modificado guardado en: {output_path}")
    except Exception as e:
        logger.error(f"Error al guardar el audio modificado: {e}")
        return {'success': False, 'message': f"Error al guardar el audio modificado: {e}"}

    return {'success': True, 'output_audio_path': output_path}
