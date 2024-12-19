# prosody.py

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
import tempfile
import logging

# Configuración básica del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def change_speed_pitch(y, sr, speed=1.0, pitch=0.0):
    """
    Cambia la velocidad y el pitch de un segmento de audio.

    Parámetros:
    - y (np.ndarray): Audio en formato numpy array.
    - sr (int): Tasa de muestreo.
    - speed (float): Factor de cambio de velocidad (1.0 = sin cambio).
    - pitch (float): Cambio de pitch en semitonos (positivo = subir, negativo = bajar).

    Retorna:
    - np.ndarray: Audio modificado.
    """
    if speed != 1.0:
        y = librosa.effects.time_stretch(y, speed)
    if pitch != 0.0:
        y = librosa.effects.pitch_shift(y, sr, n_steps=pitch)
    return y

def insert_silence(duration, sr):
    """
    Crea un segmento de silencio de una duración específica.

    Parámetros:
    - duration (float): Duración del silencio en segundos.
    - sr (int): Tasa de muestreo.

    Retorna:
    - np.ndarray: Audio de silencio.
    """
    silence = np.zeros(int(duration * sr))
    return silence

def apply_fade(y, sr, fade_duration=0.05):
    """
    Aplica un fade-in y fade-out al inicio y final del audio.

    Parámetros:
    - y (np.ndarray): Audio en formato numpy array.
    - sr (int): Tasa de muestreo.
    - fade_duration (float): Duración del fade en segundos.

    Retorna:
    - np.ndarray: Audio con fades aplicados.
    """
    fade_samples = int(fade_duration * sr)
    if fade_samples > len(y):
        fade_samples = len(y)
    
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    
    y[:fade_samples] *= fade_in
    y[-fade_samples:] *= fade_out
    
    return y

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
    - modifications (list): Lista de dicts que pueden ser de tipo 'prosody' o 'silence'.
        Para 'prosody':
            {
                'type': 'prosody',
                'start_time': float,    # segundos
                'end_time': float,      # segundos
                'pitch_shift': float,   # semitonos
                'volume_change': float, # dB
                'speed_change': float   # factor
            }
        Para 'silence':
            {
                'type': 'silence',
                'start_time': float,    # segundos
                'duration': float       # segundos
            }
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
    - dict: {'success': bool, 'output_audio_path': str, 'message': str}
    """
    logger.info(f"Recibiendo modificaciones: {modifications}")

    if not os.path.exists(audio_path):
        logger.error(f"El archivo de audio no existe: {audio_path}")
        return {'success': False, 'message': f"El archivo de audio no existe: {audio_path}"}

    # Leer el archivo de audio
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        total_duration_sec = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Cargado audio desde {audio_path} con tasa de muestreo {sr} Hz y {total_duration_sec:.2f}s de duración")
    except Exception as e:
        logger.error(f"Error al leer el archivo de audio: {e}")
        return {'success': False, 'message': f"Error al leer el archivo de audio: {e}"}

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
            total_duration_sec = librosa.get_duration(y=y, sr=sr)  # Actualizar duración total
            logger.info("Silencios eliminados")
        except Exception as e:
            logger.error(f"Error al eliminar silencios: {e}")
            return {'success': False, 'message': f"Error al eliminar silencios: {e}"}

    # Ordenar las modificaciones por start_time
    try:
        modifications_sorted = sorted(modifications, key=lambda x: x['start_time'])
        logger.info(f"Modificaciones ordenadas: {modifications_sorted}")
    except Exception as e:
        logger.error(f"Error al ordenar las modificaciones: {e}")
        return {'success': False, 'message': f"Error al ordenar las modificaciones: {e}"}

    # Inicializar variables para manejar las modificaciones
    segments = []
    last_end_sec = 0.0

    # Procesar cada modificación
    for idx, mod in enumerate(modifications_sorted):
        if mod.get('type') == 'silence':
            # Procesar silencios
            start_time = mod.get('start_time')
            duration = mod.get('duration', 1.0)  # Duración por defecto de 1 segundo

            if start_time is None:
                logger.error(f"Modificación {idx+1}: 'start_time' es necesario para silencios.")
                return {'success': False, 'message': f"Modificación {idx+1}: 'start_time' es necesario para silencios."}

            # Validar tiempos
            if start_time < 0 or start_time > total_duration_sec:
                logger.error(f"Modificación {idx+1}: 'start_time' inválido ({start_time} segundos).")
                return {'success': False, 'message': f"Modificación {idx+1}: 'start_time' inválido."}

            # Convertir tiempos a muestras
            start_sample = int(start_time * sr)

            # Agregar segmento antes del silencio
            if start_time > last_end_sec:
                segment = y[int(last_end_sec * sr):start_sample]
                segments.append(segment)
                logger.info(f"Agregado segmento sin modificar: {last_end_sec}s a {start_time}s")

            # Insertar silencio
            silence_segment = insert_silence(duration, sr)
            segments.append(silence_segment)
            logger.info(f"Insertado silencio de {duration} segundos en la modificación {idx+1}")

            last_end_sec = start_time  # Actualizar el último punto procesado

        elif mod.get('type') == 'prosody':
            # Procesar modificaciones de prosodia
            start_time = mod.get('start_time')
            end_time = mod.get('end_time')
            pitch_shift = mod.get('pitch_shift', 0.0)
            volume_change = mod.get('volume_change', 0.0)
            speed_change = mod.get('speed_change', 1.0)

            # Validar tiempos
            if start_time < 0 or end_time <= start_time:
                logger.error(f"Modificación {idx+1}: tiempos inválidos (start={start_time}, end={end_time}).")
                return {'success': False, 'message': f"Modificación {idx+1}: tiempos inválidos."}

            # Ajustar end_time si excede la duración total
            if end_time > total_duration_sec:
                logger.warning(f"Modificación {idx+1}: end_time {end_time} excede la duración total. Ajustando a {total_duration_sec}.")
                end_time = total_duration_sec

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Agregar segmento sin modificar si hay un gap
            if start_time > last_end_sec:
                segment = y[int(last_end_sec * sr):start_sample]
                segments.append(segment)
                logger.info(f"Agregado segmento sin modificar: {last_end_sec}s a {start_time}s")

            # Extraer el segmento a modificar
            segment = y[start_sample:end_sample]

            # Aplicar cambios de velocidad y pitch
            if speed_change != 1.0 or pitch_shift != 0.0:
                try:
                    segment = change_speed_pitch(segment, sr, speed=speed_change, pitch=pitch_shift)
                    logger.info(f"Modificación {idx+1}: Cambiado velocidad a {speed_change}x y pitch a {pitch_shift} semitonos")
                except Exception as e:
                    logger.error(f"Error al aplicar cambio de velocidad y pitch en modificación {idx+1}: {e}")
                    return {'success': False, 'message': f"Error al aplicar cambio de velocidad y pitch en modificación {idx+1}: {e}"}

            # Aplicar cambios de volumen
            if volume_change != 0.0:
                try:
                    volume_change_factor = 10 ** (volume_change / 20)
                    segment = segment * volume_change_factor
                    # Asegurarse de que los valores no excedan [-1.0, 1.0]
                    segment = np.clip(segment, -1.0, 1.0)
                    logger.info(f"Modificación {idx+1}: Cambiado volumen a {volume_change} dB")
                except Exception as e:
                    logger.error(f"Error al aplicar cambio de volumen en modificación {idx+1}: {e}")
                    return {'success': False, 'message': f"Error al aplicar cambio de volumen en modificación {idx+1}: {e}"}

            # Aplicar fades al segmento modificado
            try:
                segment = apply_fade(segment, sr, fade_duration=fade_duration)
                logger.info(f"Modificación {idx+1}: Aplicados fades de entrada y salida de {fade_duration}s")
            except Exception as e:
                logger.error(f"Error al aplicar fades en modificación {idx+1}: {e}")
                return {'success': False, 'message': f"Error al aplicar fades en modificación {idx+1}: {e}"}

            # Insertar silencio entre modificaciones si es necesario
            if idx < len(modifications_sorted) - 1:
                try:
                    silence = insert_silence(silence_between_mods, sr)
                    segments.append(silence)
                    logger.info(f"Modificación {idx+1}: Insertado silencio de {silence_between_mods}s entre modificaciones")
                except Exception as e:
                    logger.error(f"Error al insertar silencio entre modificaciones: {e}")
                    return {'success': False, 'message': f"Error al insertar silencio entre modificaciones: {e}"}

            # Agregar segmento modificado
            segments.append(segment)
            last_end_sec = end_time

    # Agregar el resto del audio sin modificar
    if last_end_sec < total_duration_sec:
        try:
            segment = y[int(last_end_sec * sr):]
            segments.append(segment)
            logger.info(f"Agregado segmento final sin modificar: {last_end_sec}s a {total_duration_sec}s")
        except Exception as e:
            logger.error(f"Error al agregar segmento final sin modificar: {e}")
            return {'success': False, 'message': f"Error al agregar segmento final sin modificar: {e}"}

    # Concatenar segmentos con crossfade utilizando la función crossfade_segments
    if len(segments) == 0:
        logger.error("No se generaron segmentos de audio.")
        return {'success': False, 'message': "No se generaron segmentos de audio."}

    try:
        final_audio = segments[0]
        for seg in segments[1:]:
            final_audio = crossfade_segments(final_audio, seg, sr, crossfade_duration=cross_fade_duration)
        logger.info("Segmentos concatenados con crossfade")
    except Exception as e:
        logger.error(f"Error al concatenar segmentos con crossfade: {e}")
        return {'success': False, 'message': f"Error al concatenar segmentos con crossfade: {e}"}

    # Aplicar cambio de velocidad global usando librosa
    if global_speed_change != 1.0:
        try:
            final_audio = change_speed_pitch(final_audio, sr, speed=global_speed_change, pitch=0.0)
            logger.info(f"Aplicando cambio de velocidad global: {global_speed_change}x")
        except Exception as e:
            logger.error(f"Error al aplicar cambio de velocidad global: {e}")
            return {'success': False, 'message': f"Error al aplicar cambio de velocidad global: {e}"}

    # Normalizar audio para evitar clipping
    try:
        max_val = np.max(np.abs(final_audio))
        if max_val > 1.0:
            final_audio = final_audio / max_val
            logger.info("Normalización aplicada para evitar clipping")
    except Exception as e:
        logger.error(f"Error al normalizar audio: {e}")
        return {'success': False, 'message': f"Error al normalizar audio: {e}"}

    # Convertir a int16 para guardar en WAV
    try:
        final_audio_int16 = (final_audio * 32767).astype(np.int16)
    except Exception as e:
        logger.error(f"Error al convertir audio a int16: {e}")
        return {'success': False, 'message': f"Error al convertir audio a int16: {e}"}

    # Guardar el audio modificado
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

def detect_silence(y, sr, min_silence_len=0.5, silence_thresh=-40):
    """
    Detecta silencios en el audio.

    Parámetros:
    - y (np.ndarray): Audio en formato numpy array.
    - sr (int): Tasa de muestreo.
    - min_silence_len (float): Duración mínima del silencio a detectar en segundos.
    - silence_thresh (float): Umbral de silencio en dBFS.

    Retorna:
    - list: Lista de tuplas con (inicio, fin) de silencios.
    """
    logger.info("Detectando silencios en el audio")
    import librosa.effects

    # Convertir a dB
    y_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

    # Identificar dónde el audio está por debajo del umbral
    silent_frames = y_db < silence_thresh

    # Convertir frames a tiempos
    silent_segments = []
    start = None
    for i, is_silent in enumerate(silent_frames):
        if is_silent and start is None:
            start = i
        elif not is_silent and start is not None:
            end = i
            duration = (end - start) / sr
            if duration >= min_silence_len:
                silent_segments.append((start / sr, end / sr))
            start = None
    # Manejar caso donde el audio termina en silencio
    if start is not None:
        end = len(silent_frames)
        duration = (end - start) / sr
        if duration >= min_silence_len:
            silent_segments.append((start / sr, end / sr))
    logger.info(f"Silencios detectados: {silent_segments}")
    return silent_segments

if __name__ == "__main__":
    # Definir las modificaciones correctamente
    modifications = [
        {
            'type': 'prosody',
            'start_time': 0.40,
            'end_time': 1.78,
            'pitch_shift': 0.0,
            'volume_change': 0.0,
            'speed_change': 1.5
        },
        {
            'type': 'silence',
            'start_time': 2.00,  # Ejemplo de inserción de silencio en 2 segundos
            'duration': 2.0
        },
        {
            'type': 'prosody',
            'start_time': 16.00,
            'end_time': 18.78,
            'pitch_shift': 1.0,
            'volume_change': 2.0,
            'speed_change': 1.5
        },
        {
            'type': 'silence',
            'start_time': 19.00,  # Ejemplo de inserción de silencio en 19 segundos
            'duration': 1.5
        },
        {
            'type': 'prosody',
            'start_time': 19.50,
            'end_time': 20.28,
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
        if output_audio_path['success']:
            print(f"Audio modificado guardado en: {output_audio_path['output_audio_path']}")
        else:
            print(f"Error: {output_audio_path['message']}")
    except Exception as e:
        logger.error(f"Error al modificar la prosodia: {e}")
        print(f"Error al modificar la prosodia: {e}")
