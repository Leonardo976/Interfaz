# prosody.py

import os
import tempfile
import logging
from pydub import AudioSegment, effects
import librosa
import numpy as np
import soundfile as sf

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_prosody(
    audio_path,
    modifications,
    remove_silence=False,
    min_silence_len=500,    # en milisegundos
    silence_thresh=-40,      # en dBFS
    keep_silence=250,        # en milisegundos
    cross_fade_duration=0.15,  # en segundos
    global_speed_change=1.0,    # Factor global para cambio de velocidad
    output_path=None
):
    """
    Modifica la prosodia de un archivo de audio.

    Parámetros:
    - audio_path: Ruta al archivo de audio de entrada.
    - modifications: Lista de diccionarios, cada uno conteniendo:
        - start_time: Tiempo de inicio en segundos.
        - end_time: Tiempo de fin en segundos.
        - pitch_shift: Semitonos para cambiar el tono (0 para sin cambio).
        - volume_change: dB para aumentar/disminuir el volumen (0 para sin cambio).
        - speed_change: Factor para cambiar la velocidad (1.0 para sin cambio).

    - remove_silence: Si se debe eliminar silencios.
    - min_silence_len: Longitud mínima de silencio para eliminar (ms).
    - silence_thresh: Umbral de silencio en dBFS.
    - keep_silence: Duración del silencio a mantener en cada extremo (ms).
    - cross_fade_duration: Duración del cross-fade entre clips de audio (s).
    - global_speed_change: Factor global para cambiar la velocidad del audio completo.
    - output_path: Ruta para guardar el audio modificado. Si es None, se usará un archivo temporal.

    Retorna:
    - output_path: Ruta al archivo de audio modificado.
    """
    # Verificar que el archivo existe
    if not os.path.exists(audio_path):
        logger.error(f"El archivo de audio no existe: {audio_path}")
        raise FileNotFoundError(f"El archivo de audio no existe: {audio_path}")

    # Cargar el archivo de audio con pydub
    try:
        audio = AudioSegment.from_file(audio_path)
        logger.info(f"Cargado audio desde {audio_path} con duración {len(audio) / 1000:.2f}s")
    except Exception as e:
        logger.error(f"Error al cargar el audio: {e}")
        raise RuntimeError(f"Error al cargar el audio: {e}")

    # Aplicar eliminación de silencios si es necesario
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

    # Inicializar variables para segmentación
    segments = []
    last_end_ms = 0

    for idx, mod in enumerate(modifications):
        start_time = mod.get('start_time', 0)
        end_time = mod.get('end_time', len(audio) / 1000)  # en segundos
        pitch_shift = mod.get('pitch_shift', 0)
        volume_change = mod.get('volume_change', 0)
        speed_change = mod.get('speed_change', 1.0)

        # Validaciones de tiempo y velocidad
        if start_time < 0 or end_time < 0:
            logger.error(f"Modificación {idx + 1}: Los tiempos de inicio y fin no pueden ser negativos.")
            raise ValueError(f"Modificación {idx + 1}: Los tiempos de inicio y fin no pueden ser negativos.")
        if start_time >= end_time:
            logger.error(f"Modificación {idx + 1}: El tiempo de inicio debe ser menor que el tiempo de fin.")
            raise ValueError(f"Modificación {idx + 1}: El tiempo de inicio debe ser menor que el tiempo de fin.")
        if speed_change <= 0:
            logger.error(f"Modificación {idx + 1}: El factor de cambio de velocidad debe ser mayor que 0.")
            raise ValueError(f"Modificación {idx + 1}: El factor de cambio de velocidad debe ser mayor que 0.")

        # Convertir tiempos a milisegundos
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        # Asegurar que los índices estén dentro de los límites
        start_ms = max(0, min(start_ms, len(audio)))
        end_ms = max(0, min(end_ms, len(audio)))

        logger.info(f"Procesando modificación {idx + 1}: inicio={start_time}s ({start_ms}ms), fin={end_time}s ({end_ms}ms), "
                    f"pitch_shift={pitch_shift}, volume_change={volume_change}, speed_change={speed_change}")

        # Agregar segmento sin modificar antes de la modificación actual
        if start_ms > last_end_ms:
            unmodified_segment = audio[last_end_ms:start_ms]
            segments.append(unmodified_segment)
            logger.debug(f"Agregado segmento sin modificar: {last_end_ms}ms - {start_ms}ms")

        # Extraer el segmento a modificar
        segment = audio[start_ms:end_ms]
        logger.debug(f"Segmento a modificar: {len(segment) / 1000:.2f}s")

        # Convertir a numpy array para procesamiento con librosa
        y = np.array(segment.get_array_of_samples()).astype(np.float32)
        y /= np.iinfo(segment.array_type).max  # Normalizar

        # Si el audio es estéreo, convertir a mono para procesamiento
        if segment.channels > 1:
            y = y.reshape((-1, segment.channels))
            y = y.mean(axis=1)

        # Aplicar pitch shift
        if pitch_shift != 0:
            try:
                y = librosa.effects.pitch_shift(y, segment.frame_rate, n_steps=pitch_shift)
                logger.debug(f"Cambio de pitch aplicado: {pitch_shift} semitonos")
            except Exception as e:
                logger.error(f"Error al aplicar cambio de pitch en modificación {idx + 1}: {e}")
                # Mantener el segmento original en caso de fallo
                pass

        # Aplicar cambio de velocidad
        if speed_change != 1.0:
            try:
                y = librosa.effects.time_stretch(y, rate=speed_change)
                logger.debug(f"Cambio de velocidad aplicado: factor {speed_change}")
            except Exception as e:
                logger.error(f"Error al aplicar cambio de velocidad en modificación {idx + 1}: {e}")
                # Mantener el segmento original en caso de fallo
                pass

        # Convertir de vuelta a AudioSegment
        y = (y * np.iinfo('int16').max).astype('int16')
        modified_segment = AudioSegment(
            y.tobytes(),
            frame_rate=int(segment.frame_rate * speed_change),
            sample_width=segment.sample_width,
            channels=1  # Mono después del procesamiento
        )

        # Aplicar cambio de volumen
        if volume_change != 0:
            modified_segment = modified_segment + volume_change  # pydub permite sumar o restar dB
            logger.debug(f"Cambio de volumen aplicado: {volume_change} dB")

        # Evitar clipping en el segmento modificado
        max_amp = modified_segment.max
        if max_amp > 32767:
            modified_segment = modified_segment.apply_gain(-20)  # Reducir ganancia para evitar clipping
            logger.debug("Segmento modificado normalizado para evitar clipping")

        # Agregar el segmento modificado
        segments.append(modified_segment)
        logger.debug(f"Segmento modificado agregado: {start_ms}ms - {end_ms}ms")
        last_end_ms = end_ms

    # Agregar cualquier segmento restante sin modificar después de la última modificación
    if last_end_ms < len(audio):
        unmodified_segment = audio[last_end_ms:]
        segments.append(unmodified_segment)
        logger.debug(f"Agregado segmento sin modificar al final: {last_end_ms}ms - {len(audio)}ms")

    # Concatenar todos los segmentos con cross-fade si es necesario
    if cross_fade_duration > 0 and len(segments) > 1:
        cross_fade_ms = int(cross_fade_duration * 1000)
        final_audio = segments[0]
        for seg in segments[1:]:
            final_audio = final_audio.append(seg, crossfade=cross_fade_ms)
            logger.debug(f"Aplicado cross-fade de {cross_fade_ms}ms entre segmentos")
    else:
        final_audio = sum(segments)

    # Aplicar cambio de velocidad global si es necesario
    if global_speed_change != 1.0:
        logger.info(f"Aplicando cambio de velocidad global: {global_speed_change}x")
        # Exportar a numpy para procesar con librosa
        y_final = np.array(final_audio.get_array_of_samples()).astype(np.float32)
        y_final /= np.iinfo(final_audio.array_type).max  # Normalizar

        # Si el audio es estéreo, convertir a mono
        if final_audio.channels > 1:
            y_final = y_final.reshape((-1, final_audio.channels))
            y_final = y_final.mean(axis=1)

        # Aplicar cambio de velocidad global
        y_final = librosa.effects.time_stretch(y_final, rate=global_speed_change)
        logger.debug(f"Cambio de velocidad global aplicado: {global_speed_change}x")

        # Convertir de vuelta a AudioSegment
        y_final = (y_final * np.iinfo('int16').max).astype('int16')
        final_audio = AudioSegment(
            y_final.tobytes(),
            frame_rate=int(final_audio.frame_rate * global_speed_change),
            sample_width=final_audio.sample_width,
            channels=1  # Mono después del procesamiento
        )

        # Evitar clipping
        if final_audio.max > 32767:
            final_audio = final_audio.apply_gain(-20)
            logger.debug("Audio final normalizado para evitar clipping global")

    # Guardar el audio modificado
    if output_path is None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                output_path = tmp_file.name
                logger.debug(f"Ruta de salida no proporcionada, usando archivo temporal: {output_path}")
        except Exception as e:
            logger.error(f"Error al crear archivo temporal: {e}")
            raise RuntimeError(f"Error al crear archivo temporal: {e}")

    try:
        final_audio.export(output_path, format='wav')
        logger.info(f"Audio modificado guardado exitosamente en: {output_path}")
    except Exception as e:
        logger.error(f"Error al guardar el audio modificado: {e}")
        raise RuntimeError(f"Error al guardar el audio modificado: {e}")

    return output_path
