# prosody.py

import os
import tempfile
import logging
import numpy as np
from scipy.io import wavfile
import librosa

# Configuración básica del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_silence(y, sr, min_silence_len=0.5, silence_thresh=-40):
    """
    Detecta silencios en la señal de audio.

    Parámetros:
    - y (np.ndarray): Señal de audio.
    - sr (int): Tasa de muestreo.
    - min_silence_len (float): Duración mínima del silencio en segundos.
    - silence_thresh (float): Umbral de silencio en dBFS.

    Retorna:
    - List of tuples con (start_sample, end_sample) de silencios detectados.
    """
    # Convertir umbral de dBFS a amplitud
    threshold = 10 ** (silence_thresh / 20)
    silence = np.abs(y) < threshold
    silent_regions = []
    start = None
    min_silence_samples = int(min_silence_len * sr)

    for i, is_silent in enumerate(silence):
        if is_silent:
            if start is None:
                start = i
        else:
            if start is not None:
                end = i
                if end - start >= min_silence_samples:
                    silent_regions.append((start, end))
                start = None
    if start is not None:
        silent_regions.append((start, len(y)))

    return silent_regions

def change_speed_pitch(y, sr, speed=1.0, pitch=0.0):
    """
    Cambia la velocidad y el pitch de la señal de audio utilizando librosa.

    Parámetros:
    - y (np.ndarray): Señal de audio mono.
    - sr (int): Tasa de muestreo.
    - speed (float): Factor de velocidad (1.0 = sin cambio).
    - pitch (float): Cambio de pitch en semitonos (positivo = subir, negativo = bajar).

    Retorna:
    - np.ndarray: Señal de audio con velocidad y pitch modificados.
    """
    try:
        # Validar speed_change
        if speed <= 0:
            logger.warning(f"speed_change={speed} es inválido. Ajustando a speed_change=0.1")
            speed = 0.1  # Evitar speed_change=0 o negativo

        # Cambiar la velocidad manteniendo el pitch
        if speed != 1.0:
            y = librosa.effects.time_stretch(y, rate=speed)
            logger.info(f"Cambio de velocidad aplicado: {speed}x")

        # Cambiar el pitch manteniendo la velocidad
        if pitch != 0.0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
            logger.info(f"Cambio de pitch aplicado: {pitch} semitonos")

        return y
    except Exception as e:
        logger.error(f"Error en change_speed_pitch con librosa: {e}")
        return y

def apply_fade(segment, sr, fade_duration=0.05):
    """
    Aplica un fade-in y fade-out al segmento de audio.

    Parámetros:
    - segment (np.ndarray): Segmento de audio.
    - sr (int): Tasa de muestreo.
    - fade_duration (float): Duración del fade en segundos.

    Retorna:
    - np.ndarray: Segmento de audio con fades aplicados.
    """
    fade_samples = int(fade_duration * sr)
    if fade_samples > len(segment):
        fade_samples = len(segment) // 2  # Evita que el fade sea más largo que el segmento

    # Fade-In
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    segment[:fade_samples] *= fade_in

    # Fade-Out
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    segment[-fade_samples:] *= fade_out

    return segment

def insert_silence(duration, sr):
    """
    Crea un segmento de silencio.

    Parámetros:
    - duration (float): Duración del silencio en segundos.
    - sr (int): Tasa de muestreo.

    Retorna:
    - np.ndarray: Segmento de silencio.
    """
    silence_samples = int(duration * sr)
    return np.zeros(silence_samples, dtype=np.float32)

def crossfade_segments(seg1, seg2, sr, crossfade_duration=0.05):
    """
    Aplica un crossfade entre dos segmentos de audio.

    Parámetros:
    - seg1 (np.ndarray): Primer segmento de audio.
    - seg2 (np.ndarray): Segundo segmento de audio.
    - sr (int): Tasa de muestreo.
    - crossfade_duration (float): Duración del crossfade en segundos.

    Retorna:
    - np.ndarray: Segmento de audio combinado con crossfade.
    """
    crossfade_samples = int(crossfade_duration * sr)
    if crossfade_samples > len(seg1):
        crossfade_samples = len(seg1)
    if crossfade_samples > len(seg2):
        crossfade_samples = len(seg2)

    # Crear rampas para crossfade
    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)

    # Aplicar rampas
    seg1_end = seg1[-crossfade_samples:] * fade_out
    seg2_start = seg2[:crossfade_samples] * fade_in

    # Concatenar segmentos
    combined = np.concatenate([
        seg1[:-crossfade_samples],
        seg1_end + seg2_start,
        seg2[crossfade_samples:]
    ])

    return combined

def modify_prosody(
    audio_path,
    modifications,
    remove_silence=False,
    min_silence_len=0.5,    # segundos
    silence_thresh=-40,     # dBFS
    keep_silence=0.25,      # segundos
    global_speed_change=1.0,
    cross_fade_duration=0.05, # segundos
    output_path=None,
    fade_duration=0.05,      # segundos para fade-in y fade-out
    silence_between_mods=0.02 # segundos de silencio entre modificaciones
):
    """
    Aplica las modificaciones de prosodia a un audio.

    Parámetros:
    - audio_path (str): Ruta al archivo de audio de entrada.
    - modifications (list): Lista de dicts {start_time, end_time, pitch_shift, volume_change, speed_change}.
    - remove_silence (bool): Eliminar silencios del audio.
    - min_silence_len (float): Duración mínima del silencio a eliminar (s).
    - silence_thresh (float): Umbral de silencio en dBFS.
    - keep_silence (float): Duración de silencio a mantener después de eliminar (s).
    - global_speed_change (float): Factor global de velocidad (1.0 = sin cambio).
    - cross_fade_duration (float): Duración del crossfade entre segmentos (s).
    - output_path (str): Ruta para guardar el audio modificado. Si es None, se usa un archivo temporal.
    - fade_duration (float): Duración del fade en segundos para cada segmento modificado.
    - silence_between_mods (float): Duración del silencio entre modificaciones en segundos.

    Retorna:
    - output_path (str): Ruta al archivo de audio modificado.
    """
    logger.info(f"Recibiendo modificaciones: {modifications}")

    if not os.path.exists(audio_path):
        logger.error(f"El archivo de audio no existe: {audio_path}")
        raise FileNotFoundError(f"El archivo de audio no existe: {audio_path}")

    # Leer el archivo de audio
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        logger.info(f"Cargado audio desde {audio_path} con tasa de muestreo {sr} Hz y {len(y)/sr:.2f}s de duración")
    except Exception as e:
        logger.error(f"Error al leer el archivo de audio: {e}")
        raise

    # Eliminar silencios si es necesario
    if remove_silence:
        try:
            logger.info("Eliminando silencios del audio")
            silent_ranges = detect_silence(y, sr, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
            non_silent_audio = []
            prev_end = 0
            for start, end in silent_ranges:
                # Mantener un poco de silencio después de eliminar
                non_silent_audio.append(y[prev_end:start])
                non_silent_audio.append(insert_silence(keep_silence, sr))
                prev_end = end
            non_silent_audio.append(y[prev_end:])
            y = np.concatenate(non_silent_audio)
            logger.info("Silencios eliminados")
        except Exception as e:
            logger.error(f"Error al eliminar silencios: {e}")
            raise RuntimeError(f"Error al eliminar silencios: {e}")

    # Ordenar las modificaciones por start_time
    try:
        modifications = sorted(modifications, key=lambda x: x['start_time'])
        logger.info(f"Modificaciones ordenadas: {modifications}")
    except Exception as e:
        logger.error(f"Error al ordenar las modificaciones: {e}")
        raise

    segments = []
    last_end_sec = 0.0

    # Aplicar modificaciones
    for idx, mod in enumerate(modifications):
        logger.info(f"Procesando modificación {idx+1}: {mod}")
        start_sec = mod.get('start_time', 0.0)
        end_sec = mod.get('end_time', 0.0)
        pitch_shift_steps = mod.get('pitch_shift', 0.0)
        volume_change_db = mod.get('volume_change', 0.0)
        speed_change = mod.get('speed_change', 1.0)

        # Validar tiempos
        if start_sec < 0 or end_sec <= start_sec:
            logger.error(f"Modificación {idx+1}: tiempos inválidos (start={start_sec}, end={end_sec}).")
            raise ValueError(f"Modificación {idx+1}: tiempos inválidos.")

        # Ajustar end_sec si excede la duración total
        total_duration_sec = len(y) / sr
        if end_sec > total_duration_sec:
            logger.warning(f"Modificación {idx+1}: end_time {end_sec} excede la duración total. Ajustando a {total_duration_sec}.")
            end_sec = total_duration_sec

        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)

        # Agregar segmento sin modificar si hay un gap
        if start_sec > last_end_sec:
            try:
                # Asegurarse de que los índices sean enteros
                unmodified_segment = y[int(last_end_sec * sr) : start_sample]
                segments.append(unmodified_segment)
                logger.info(f"Agregado segmento sin modificar: {last_end_sec}s a {start_sec}s")
            except Exception as e:
                logger.error(f"Error al agregar segmento sin modificar: {e}")
                raise RuntimeError(f"Error al agregar segmento sin modificar: {e}")

        # Extraer el segmento a modificar
        segment = y[start_sample:end_sample]

        # Interpretar speed_change=0 como una solicitud para silenciar el segmento
        if speed_change == 0.0:
            logger.info(f"Modificación {idx+1}: speed_change=0.0 interpretado como solicitud de silencio.")
            # Silenciar el segmento
            volume_change_db = -100.0  # Asegurar que el volumen sea prácticamente nulo
            speed_change = 1.0  # Resetear velocidad a normal

        # Aplicar cambios de velocidad y pitch usando librosa
        if speed_change != 1.0 or pitch_shift_steps != 0.0:
            try:
                segment = change_speed_pitch(segment, sr, speed=speed_change, pitch=pitch_shift_steps)
                logger.info(f"Modificación {idx+1}: Cambiado velocidad a {speed_change}x y pitch a {pitch_shift_steps} semitonos")
            except Exception as e:
                logger.error(f"Error al aplicar cambio de velocidad y pitch en modificación {idx+1}: {e}")
                raise RuntimeError(f"Error al aplicar cambio de velocidad y pitch en modificación {idx+1}: {e}")

        # Aplicar cambios de volumen
        if volume_change_db != 0.0:
            try:
                volume_change_factor = 10 ** (volume_change_db / 20)
                segment = segment * volume_change_factor
                # Asegurarse de que los valores no excedan [-1.0, 1.0]
                segment = np.clip(segment, -1.0, 1.0)
                logger.info(f"Modificación {idx+1}: Cambiado volumen a {volume_change_db} dB")
            except Exception as e:
                logger.error(f"Error al aplicar cambio de volumen en modificación {idx+1}: {e}")
                raise RuntimeError(f"Error al aplicar cambio de volumen en modificación {idx+1}: {e}")

        # Aplicar fades al segmento modificado
        try:
            segment = apply_fade(segment, sr, fade_duration=fade_duration)
            logger.info(f"Modificación {idx+1}: Aplicados fades de entrada y salida de {fade_duration}s")
        except Exception as e:
            logger.error(f"Error al aplicar fades en modificación {idx+1}: {e}")
            raise RuntimeError(f"Error al aplicar fades en modificación {idx+1}: {e}")

        # Insertar silencio entre modificaciones
        if idx < len(modifications) - 1:
            try:
                silence = insert_silence(silence_between_mods, sr)
                segments.append(silence)
                logger.info(f"Modificación {idx+1}: Insertado silencio de {silence_between_mods}s")
            except Exception as e:
                logger.error(f"Error al insertar silencio entre modificaciones: {e}")
                raise RuntimeError(f"Error al insertar silencio entre modificaciones: {e}")

        # Agregar segmento modificado
        segments.append(segment)
        last_end_sec = end_sec

    # Agregar el resto del audio sin modificar
    if last_end_sec < total_duration_sec:
        try:
            unmodified_segment = y[int(last_end_sec * sr) : ]
            segments.append(unmodified_segment)
            logger.info(f"Agregado segmento final sin modificar: {last_end_sec}s a {total_duration_sec}s")
        except Exception as e:
            logger.error(f"Error al agregar segmento final sin modificar: {e}")
            raise RuntimeError(f"Error al agregar segmento final sin modificar: {e}")

    # Concatenar segmentos con crossfade utilizando la función crossfade_segments
    if len(segments) == 0:
        logger.error("No se generaron segmentos de audio.")
        raise ValueError("No se generaron segmentos de audio.")

    try:
        final_audio = segments[0]
        for seg in segments[1:]:
            final_audio = crossfade_segments(final_audio, seg, sr, crossfade_duration=cross_fade_duration)
        logger.info("Segmentos concatenados con crossfade")
    except Exception as e:
        logger.error(f"Error al concatenar segmentos con crossfade: {e}")
        raise RuntimeError(f"Error al concatenar segmentos con crossfade: {e}")

    # Aplicar cambio de velocidad global usando librosa
    if global_speed_change != 1.0:
        try:
            final_audio = change_speed_pitch(final_audio, sr, speed=global_speed_change, pitch=0.0)
            logger.info(f"Aplicando cambio de velocidad global: {global_speed_change}x")
        except Exception as e:
            logger.error(f"Error al aplicar cambio de velocidad global: {e}")
            raise RuntimeError(f"Error al aplicar cambio de velocidad global: {e}")

    # Normalizar audio para evitar clipping
    try:
        max_val = np.max(np.abs(final_audio))
        if max_val > 1.0:
            final_audio = final_audio / max_val
            logger.info("Normalización aplicada para evitar clipping")
    except Exception as e:
        logger.error(f"Error al normalizar audio: {e}")
        raise RuntimeError(f"Error al normalizar audio: {e}")

    # Convertir a int16 para guardar en WAV
    try:
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
    except Exception as e:
        logger.error(f"Error al convertir audio a int16: {e}")
        raise RuntimeError(f"Error al convertir audio a int16: {e}")

    # Guardar el audio modificado
    if output_path is None:
        try:
            fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        except Exception as e:
            logger.error(f"Error al crear archivo temporal: {e}")
            raise RuntimeError(f"Error al crear archivo temporal: {e}")

    try:
        wavfile.write(output_path, sr, final_audio_int16)
        logger.info(f"Audio modificado guardado en: {output_path}")
    except Exception as e:
        logger.error(f"Error al guardar el audio modificado: {e}")
        raise RuntimeError(f"Error al guardar el audio modificado: {e}")

    return output_path

if __name__ == "__main__":
    # Definir las modificaciones correctamente
    modifications = [
        {
            'start_time': 0.40,
            'end_time': 1.78,
            'pitch_shift': 0.0,
            'volume_change': 0.0,
            'speed_change': 1.5
        },
        {
            'start_time': 16.00,
            'end_time': 18.78,
            'pitch_shift': 1.0,
            'volume_change': 2.0,
            'speed_change': 1.5
        },
        {
            'start_time': 19.00,
            'end_time': 19.78,          # Ajustado a la duración total
            'pitch_shift': -0.5,
            'volume_change': -1.0,
            'speed_change': 0.8
        }
    ]

    try:
        output_audio_path = modify_prosody(
            audio_path='ruta/al/audio_original.wav',  # Reemplaza con la ruta real
            modifications=modifications,
            remove_silence=True,         # Cambia a False para pruebas iniciales
            min_silence_len=0.5,
            silence_thresh=-40,
            keep_silence=0.25,
            global_speed_change=1.0,      # sin cambio de velocidad global
            cross_fade_duration=0.05,
            fade_duration=0.05,
            silence_between_mods=0.02
        )
        print(f"Audio modificado guardado en: {output_audio_path}")
    except Exception as e:
        logger.error(f"Error al modificar la prosodia: {e}")
        print(f"Error al modificar la prosodia: {e}")
