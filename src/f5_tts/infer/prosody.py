# prosody.py

import os
import tempfile
import logging
from pydub import AudioSegment, effects
import librosa
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_prosody(
    audio_path,
    modifications,
    remove_silence=False,
    min_silence_len=500,    # ms
    silence_thresh=-40,     # dBFS
    keep_silence=250,       # ms
    global_speed_change=1.0,
    cross_fade_duration=0.05, # crossfade corto (50ms)
    output_path=None
):
    """
    Aplica las modificaciones de prosodia a un audio.

    Parámetros:
    - audio_path (str): Ruta al archivo de audio de entrada.
    - modifications (list): Lista de dicts {start_time, end_time, pitch_shift, volume_change, speed_change}.
    - remove_silence (bool): Eliminar silencios del audio.
    - min_silence_len (int): Duración mínima del silencio a eliminar (ms).
    - silence_thresh (int): Umbral de silencio en dBFS.
    - keep_silence (int): Duración de silencio a mantener después de eliminar.
    - global_speed_change (float): Factor global de velocidad (1.0 = sin cambio).
    - cross_fade_duration (float): Duración del crossfade entre segmentos (s).
    - output_path (str): Ruta para guardar el audio modificado. Si es None, se usa un archivo temporal.

    Retorna:
    - output_path (str): Ruta al archivo de audio modificado.
    """
    if not os.path.exists(audio_path):
        logger.error(f"El archivo de audio no existe: {audio_path}")
        raise FileNotFoundError(f"El archivo de audio no existe: {audio_path}")

    try:
        audio = AudioSegment.from_file(audio_path)
        logger.info(f"Cargado audio desde {audio_path} con duración {len(audio) / 1000:.2f}s")
    except Exception as e:
        logger.error(f"Error al cargar el audio: {e}")
        raise RuntimeError(f"Error al cargar el audio: {e}")

    if remove_silence:
        logger.info("Eliminando silencios del audio")
        audio = effects.strip_silence(
            audio,
            silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            padding=keep_silence
        )
        logger.info("Silencios eliminados")

    # Ordenar las modificaciones por start_time
    modifications = sorted(modifications, key=lambda x: x['start_time'])
    logger.info(f"Modificaciones ordenadas: {modifications}")

    # Obtener duración total del audio en segundos
    total_duration_sec = len(audio) / 1000.0

    # Si la primera modificación empieza en 0.0, extenderla al final del audio
    if modifications and modifications[0]['start_time'] == 0.0:
        logger.info("Modificación inicial encontrada en 0.0, extendiendo hasta el final del audio")
        modifications[0]['end_time'] = total_duration_sec
        # Eliminar cualquier modificación posterior para evitar superposición
        modifications = [modifications[0]]

    segments = []
    last_end_ms = 0

    cross_fade_ms = int(cross_fade_duration * 1000)

    for idx, mod in enumerate(modifications):
        start_time = mod['start_time']
        end_time = mod['end_time']
        pitch_shift = mod.get('pitch_shift', 0.0)
        volume_change = mod.get('volume_change', 0.0)
        speed_change = mod.get('speed_change', 1.0)

        if start_time < 0 or end_time <= start_time:
            logger.error(f"Modificación {idx+1}: tiempos inválidos (start={start_time}, end={end_time}).")
            raise ValueError(f"Modificación {idx+1}: tiempos inválidos.")

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        # Clamp end_time to the total duration
        if end_time > total_duration_sec:
            end_time = total_duration_sec
            end_ms = int(end_time * 1000)

        start_ms = max(0, min(start_ms, len(audio)))
        end_ms = max(0, min(end_ms, len(audio)))

        segment = audio[start_ms:end_ms]

        # Convertir el segmento a numpy array para procesar con librosa
        y = np.array(segment.get_array_of_samples()).astype(np.float32)
        y /= np.iinfo(segment.array_type).max
        sr = segment.frame_rate
        if segment.channels > 1:
            y = y.reshape((-1, segment.channels))
            y = y.mean(axis=1)

        # Aplicar pitch shift
        if pitch_shift != 0.0 and len(y) > 0:
            try:
                y = librosa.effects.pitch_shift(y, sr, n_steps=pitch_shift)
            except Exception as e:
                logger.error(f"Pitch shift error mod {idx+1}: {e}")

        # Aplicar speed change
        if speed_change != 1.0 and len(y) > 0:
            try:
                y = librosa.effects.time_stretch(y, rate=speed_change)
            except Exception as e:
                logger.error(f"Time stretch error mod {idx+1}: {e}")

        if len(y) == 0:
            modified_segment = AudioSegment.silent(duration=100)
        else:
            y = (y * np.iinfo('int16').max).astype('int16')
            modified_segment = AudioSegment(
                y.tobytes(),
                frame_rate=int(sr * speed_change),
                sample_width=segment.sample_width,
                channels=1
            )
            if volume_change != 0.0:
                modified_segment = modified_segment + volume_change

        # Agregar segmento sin modificar antes de la modificación actual
        if start_ms > last_end_ms:
            unmodified_segment = audio[last_end_ms:start_ms]
            segments.append(unmodified_segment)

        segments.append(modified_segment)
        last_end_ms = end_ms

    # Agregar el resto del audio sin modificar
    if last_end_ms < len(audio):
        unmodified_segment = audio[last_end_ms:]
        segments.append(unmodified_segment)

    # Unir segmentos con crossfade si hay más de uno
    if len(segments) == 1:
        final_audio = segments[0]
    else:
        final_audio = segments[0]
        for seg in segments[1:]:
            if len(seg) < cross_fade_ms:
                final_audio = final_audio + seg
            else:
                final_audio = final_audio.append(seg, crossfade=cross_fade_ms)

    # Aplicar cambio de velocidad global
    if global_speed_change != 1.0:
        logger.info(f"Aplicando cambio de velocidad global: {global_speed_change}x")
        y_final = np.array(final_audio.get_array_of_samples()).astype(np.float32)
        y_final /= np.iinfo(final_audio.array_type).max
        if final_audio.channels > 1:
            y_final = y_final.reshape((-1, final_audio.channels))
            y_final = y_final.mean(axis=1)
        y_final = librosa.effects.time_stretch(y_final, rate=global_speed_change)
        y_final = (y_final * np.iinfo('int16').max).astype('int16')
        final_audio = AudioSegment(
            y_final.tobytes(),
            frame_rate=int(final_audio.frame_rate * global_speed_change),
            sample_width=final_audio.sample_width,
            channels=1
        )

    # Guardar el audio modificado
    if output_path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            output_path = tmp_file.name

    final_audio.export(output_path, format='wav')
    logger.info(f"Audio modificado guardado en: {output_path}")

    return output_path
