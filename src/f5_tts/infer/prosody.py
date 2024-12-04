# prosody.py

import numpy as np
import librosa
import soundfile as sf
import tempfile
import logging
import os
import inspect  # Para inspeccionar la función pitch_shift

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def modify_prosody(audio_path, modifications, output_path=None):
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

    - output_path: Ruta para guardar el audio modificado. Si es None, se usará un archivo temporal.

    Retorna:
    - output_path: Ruta al archivo de audio modificado.
    """
    # Verificar que el archivo existe
    if not os.path.exists(audio_path):
        logger.error(f"El archivo de audio no existe: {audio_path}")
        raise FileNotFoundError(f"El archivo de audio no existe: {audio_path}")

    # Cargar el archivo de audio
    try:
        y, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Cargado audio desde {audio_path} con sr={sr}, duración={len(y)/sr:.2f}s")
    except Exception as e:
        logger.error(f"Error al cargar el audio: {e}")
        raise RuntimeError(f"Error al cargar el audio: {e}")

    # Inspección de la función pitch_shift
    logger.debug(f"librosa.effects.pitch_shift: {librosa.effects.pitch_shift}")
    try:
        logger.debug(f"Firma de pitch_shift: {inspect.signature(librosa.effects.pitch_shift)}")
    except ValueError:
        logger.debug("No se pudo obtener la firma de la función pitch_shift.")

    total_samples = len(y)
    total_duration = total_samples / sr

    # Validar modificaciones
    for idx, mod in enumerate(modifications):
        start_time = mod.get('start_time', 0)
        end_time = mod.get('end_time', total_duration)
        pitch_shift_value = mod.get('pitch_shift', 0)
        volume_change = mod.get('volume_change', 0)
        speed_change = mod.get('speed_change', 1.0)

        # Validaciones de tiempo y velocidad
        if start_time < 0 or end_time < 0:
            logger.error(f"Modificación {idx}: Los tiempos de inicio y fin no pueden ser negativos.")
            raise ValueError(f"Modificación {idx}: Los tiempos de inicio y fin no pueden ser negativos.")
        if start_time >= end_time:
            logger.error(f"Modificación {idx}: El tiempo de inicio debe ser menor que el tiempo de fin.")
            raise ValueError(f"Modificación {idx}: El tiempo de inicio debe ser menor que el tiempo de fin.")
        if speed_change <= 0:
            logger.error(f"Modificación {idx}: El factor de cambio de velocidad debe ser mayor que 0.")
            raise ValueError(f"Modificación {idx}: El factor de cambio de velocidad debe ser mayor que 0.")

    # Ordenar las modificaciones por start_time
    modifications = sorted(modifications, key=lambda x: x['start_time'])
    logger.info(f"Modificaciones ordenadas: {modifications}")

    # Inicializar variables
    segments = []
    last_end_sample = 0

    for idx, mod in enumerate(modifications):
        start_time = mod['start_time']
        end_time = mod['end_time']
        pitch_shift_value = mod['pitch_shift']
        volume_change = mod['volume_change']
        speed_change = mod['speed_change']

        # Convertir tiempos a índices de muestras
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Asegurar que los índices estén dentro de los límites
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(0, min(end_sample, total_samples))

        logger.info(f"Procesando modificación {idx + 1}: inicio={start_time}s ({start_sample}), fin={end_time}s ({end_sample}), "
                    f"pitch_shift={pitch_shift_value}, volume_change={volume_change}, speed_change={speed_change}")

        # Agregar segmento sin modificar antes de la modificación actual
        if start_sample > last_end_sample:
            unmodified_segment = y[last_end_sample:start_sample]
            segments.append(unmodified_segment)
            logger.debug(f"Agregado segmento sin modificar: muestras {last_end_sample} - {start_sample}")

        # Extraer el segmento a modificar
        segment = y[start_sample:end_sample]
        logger.debug(f"Segmento a modificar: {len(segment)} muestras")

        # Aplicar cambio de tono
        if pitch_shift_value != 0:
            try:
                # Llamada correcta a pitch_shift
                segment = librosa.effects.pitch_shift(segment, sr, n_steps=pitch_shift_value)
                logger.debug(f"Cambio de tono aplicado: {pitch_shift_value} semitonos")
            except TypeError as e:
                logger.error(f"Error al aplicar cambio de tono en modificación {idx + 1}: {e}")
                # Mantener el segmento original en caso de fallo
                pass
            except Exception as e:
                logger.error(f"Error inesperado al aplicar cambio de tono en modificación {idx + 1}: {e}")
                pass

        # Aplicar cambio de velocidad
        if speed_change != 1.0:
            try:
                segment = librosa.effects.time_stretch(segment, rate=speed_change)
                logger.debug(f"Cambio de velocidad aplicado: factor {speed_change}")
            except librosa.util.exceptions.ParameterError as e:
                logger.error(f"Error al aplicar cambio de velocidad en modificación {idx + 1}: {e}")
                # Mantener el segmento original en caso de fallo
                pass
            except Exception as e:
                logger.error(f"Otro error al aplicar cambio de velocidad en modificación {idx + 1}: {e}")
                # Mantener el segmento original en caso de fallo
                pass

        # Aplicar cambio de volumen
        if volume_change != 0:
            try:
                volume_factor = 10 ** (volume_change / 20)
                segment *= volume_factor
                logger.debug(f"Cambio de volumen aplicado: {volume_change} dB")
            except Exception as e:
                logger.error(f"Error al aplicar cambio de volumen en modificación {idx + 1}: {e}")
                # Mantener el segmento original en caso de fallo
                pass

        # Evitar clipping en el segmento
        max_amp = np.max(np.abs(segment))
        if max_amp > 1.0:
            segment = segment / max_amp
            logger.debug("Segmento normalizado para evitar clipping")

        # Agregar segmento modificado
        segments.append(segment)
        last_end_sample = end_sample
        logger.debug(f"Segmento modificado agregado: muestras {start_sample} - {end_sample}")

    # Agregar cualquier segmento restante sin modificar después de la última modificación
    if last_end_sample < total_samples:
        unmodified_segment = y[last_end_sample:]
        segments.append(unmodified_segment)
        logger.debug(f"Agregado segmento sin modificar al final: muestras {last_end_sample} - {total_samples}")

    # Concatenar todos los segmentos
    try:
        y_modified = np.concatenate(segments)
        logger.info(f"Concatenación de segmentos completada, longitud total: {len(y_modified)} muestras")
    except Exception as e:
        logger.error(f"Error al concatenar segmentos: {e}")
        raise RuntimeError(f"Error al concatenar segmentos: {e}")

    # Normalizar para evitar clipping general
    max_amp = np.max(np.abs(y_modified))
    if max_amp > 1.0:
        y_modified = y_modified / max_amp
        logger.debug("Audio final normalizado para evitar clipping")

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
        sf.write(output_path, y_modified, sr)
        logger.info(f"Audio modificado guardado exitosamente en: {output_path}")
    except Exception as e:
        logger.error(f"Error al guardar el audio modificado: {e}")
        raise RuntimeError(f"Error al guardar el audio modificado: {e}")

    return output_path
