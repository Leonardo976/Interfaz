// src/components/MultiSpeechGenerator.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import SpeechTypeInput from './SpeechTypeInput';
import AudioPlayer from './AudioPlayer';

const MAX_SPEECH_TYPES = 100;

function MultiSpeechGenerator() {
  const [speechTypes, setSpeechTypes] = useState([
    { id: 'regular', name: 'Regular', isVisible: true }
  ]);
  const [generationText, setGenerationText] = useState('');
  const [removeSilence, setRemoveSilence] = useState(false);
  const [speedChange, setSpeedChange] = useState(1.0);
  const [crossFadeDuration, setCrossFadeDuration] = useState(0.15); 
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState(null);
  const [transcriptionData, setTranscriptionData] = useState(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeDone, setAnalyzeDone] = useState(false);

  const [editableTranscription, setEditableTranscription] = useState('');
  const [isProsodyGenerating, setIsProsodyGenerating] = useState(false);
  const [modifiedAudio, setModifiedAudio] = useState(null);

  const [audioData, setAudioData] = useState({
    regular: { audio: null, refText: '' }
  });

  // Estado para manejar la lista de audios generados
  const [generatedAudios, setGeneratedAudios] = useState([]);

  const handleAddSpeechType = () => {
    if (speechTypes.length < MAX_SPEECH_TYPES) {
      const newId = `speech-type-${speechTypes.length}`;
      setSpeechTypes([...speechTypes, { id: newId, name: '', isVisible: true }]);
    } else {
      toast.error('Se ha alcanzado el límite máximo de tipos de habla');
    }
  };

  const handleDeleteSpeechType = (idToDelete) => {
    setSpeechTypes(speechTypes.filter(type => type.id !== idToDelete));
    const newAudioData = { ...audioData };
    delete newAudioData[idToDelete];
    setAudioData(newAudioData);
  };

  const handleNameUpdate = (id, newName) => {
    setSpeechTypes(speechTypes.map(type => 
      type.id === id ? { ...type, name: newName } : type
    ));
  };

  const handleAudioUpload = async (id, file, refText, speechType) => {
    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('speechType', speechType);
      formData.append('refText', refText);

      const response = await axios.post('http://localhost:5000/api/upload_audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.success) {
        setAudioData({
          ...audioData,
          [id]: { 
            audio: response.data.filepath,
            refText: refText,
            speechType: speechType
          }
        });

        toast.success('Audio cargado correctamente');
      } else {
        toast.error(response.data.message || 'Error al cargar el audio');
      }
    } catch (error) {
      toast.error('Error al cargar el audio');
      console.error('Error:', error);
    }
  };

  const handleInsertSpeechType = (name) => {
    setGenerationText(prev => `${prev}{${name}} `);
  };

  // Nueva función para insertar silencios
  const handleInsertSilence = (duration) => {
    setGenerationText(prev => `${prev}<silencio ${duration}> `);
  };

  const handleGenerate = async () => {
    try {
      setIsGenerating(true);

      const mentionedTypes = [...generationText.matchAll(/\{([^}]+)\}/g)].map(match => match[1]);
      const availableTypes = speechTypes
        .filter(type => type.isVisible && audioData[type.id]?.audio)
        .map(type => type.name);

      const missingTypes = mentionedTypes.filter(type => !availableTypes.includes(type));
      
      if (missingTypes.length > 0) {
        toast.error(`Faltan audios de referencia para: ${missingTypes.join(', ')}`);
        return;
      }

      const speechTypesData = {};
      speechTypes.forEach(type => {
        if (type.isVisible && audioData[type.id]) {
          speechTypesData[type.name] = {
            audio: audioData[type.id].audio,
            ref_text: audioData[type.id].refText
          };
        }
      });

      const response = await axios.post('http://localhost:5000/api/generate_multistyle_speech', {
        speech_types: speechTypesData,
        gen_text: generationText,
        remove_silence: removeSilence,
        cross_fade_duration: crossFadeDuration,
        speed_change: speedChange
      });

      if (response.data.success && response.data.audio_path) {
        setGeneratedAudio(response.data.audio_path);
        setGeneratedAudios(prev => [...prev, response.data.audio_path]); // Añadir a la lista de audios generados
        setTranscriptionData(null);
        setAnalyzeDone(false);
        toast.success('Audio generado correctamente');
      } else {
        toast.error(response.data.message || 'Error al generar el audio');
      }
    } catch (error) {
      toast.error('Error al generar el audio');
      console.error('Error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleAnalyzeAudio = async () => {
    if (!generatedAudio) {
      toast.error('No hay audio generado para analizar.');
      return;
    }

    try {
      setIsAnalyzing(true);
      const response = await axios.post('http://localhost:5000/api/analyze_audio', {
        audio_path: generatedAudio
      });
      setTranscriptionData(response.data);
      if(response.data.success) {
        toast.success('Transcripción con timestamps obtenida');
        setAnalyzeDone(true);
      } else {
        toast.error('Error al obtener transcripción');
      }
    } catch (error) {
      toast.error('Error al obtener transcripción');
      console.error('Error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    if (transcriptionData && transcriptionData.success && transcriptionData.transcription) {
      setEditableTranscription(transcriptionData.transcription);
      setModifiedAudio(null);
    }
  }, [transcriptionData]);

  /**
   * Función para parsear las modificaciones desde el texto editable.
   * Ahora maneja tanto marcas de prosodia como de silencio.
   * Se ha modificado para incluir 'start_time' en las modificaciones de tipo 'silence'.
   */
  const parseModificationsFromText = (text) => {
    // Regex para encontrar marcas de tiempo
    const timePattern = /\((\d+\.\d+)\)/g;
    let matches = [];
    let match;
    while ((match = timePattern.exec(text)) !== null) {
      matches.push({time: parseFloat(match[1]), index: match.index});
    }

    const hasProsodyMarks = (segment_text) => {
      // Patrón insensible a mayúsculas/minúsculas para detectar marcas de prosodia
      const tagPattern = /<(pitch|volume|velocity)\s+([\d\.]+)>/i;
      return tagPattern.test(segment_text);
    };

    const hasSilenceMarks = (segment_text) => {
      // Patrón insensible a mayúsculas/minúsculas para detectar marcas de silencio
      const silencePattern = /<silencio\s+([\d\.]+)>/i;
      return silencePattern.test(segment_text);
    };

    const extractProsodyMarks = (segment_text) => {
      const result = {pitch_shift:0, volume_change:0, speed_change:1.0};
      const tagPatternAll = /<(pitch|volume|velocity)\s+([\d\.]+)>/gi;
      let tagM;
      while((tagM = tagPatternAll.exec(segment_text)) !== null) {
        const type = tagM[1].toLowerCase();
        const val = parseFloat(tagM[2]);
        if(type==='pitch') result.pitch_shift=val;
        if(type==='volume') result.volume_change=val;
        if(type==='velocity') result.speed_change=val;
      }
      return result;
    };

    const extractSilenceMarks = (segment_text) => {
      const silencePattern = /<silencio\s+([\d\.]+)>/i;
      const match = silencePattern.exec(segment_text);
      if (match) {
        return parseFloat(match[1]);
      }
      return null;
    };

    let modifications = [];
    let accumulatedStart = 0.0;

    // Detectar si hay marcas antes de la primera marca de tiempo
    let firstTime = (matches.length > 0) ? matches[0].time : null;
    const textBeforeFirstTime = (matches.length > 0) ? text.substring(0, matches[0].index) : text;

    // Extraer marcas antes de la primera marca de tiempo
    let initialProsodyMarks = {pitch_shift:0, volume_change:0, speed_change:1.0};
    let initialSilenceDuration = null;
    if (hasProsodyMarks(textBeforeFirstTime)) {
      initialProsodyMarks = extractProsodyMarks(textBeforeFirstTime);
      // Aplicar la modificación desde 0.0 hasta el primer tiempo encontrado o el final
      const firstTimeFound = (matches.length > 0) ? matches[0].time : 9999.0;
      modifications.push({
        type: 'prosody',
        start_time: 0.0,
        end_time: firstTimeFound,
        pitch_shift: initialProsodyMarks.pitch_shift,
        volume_change: initialProsodyMarks.volume_change,
        speed_change: initialProsodyMarks.speed_change
      });
      accumulatedStart = firstTimeFound;
    }

    if (hasSilenceMarks(textBeforeFirstTime)) {
      initialSilenceDuration = extractSilenceMarks(textBeforeFirstTime);
      if (initialSilenceDuration) {
        modifications.push({
          type: 'silence',
          start_time: accumulatedStart,
          duration: initialSilenceDuration
        });
        accumulatedStart += initialSilenceDuration;
      }
    }

    // Iterar sobre las marcas de tiempo
    for (let i = 0; i < matches.length; i++) {
      const start_time = matches[i].time;
      const start_index = matches[i].index;
      const end_index = (i < matches.length - 1) ? matches[i+1].index : text.length;
      const segment_text = text.substring(start_index, end_index);

      if (hasProsodyMarks(segment_text)) {
        // Extraer las marcas de esta palabra
        const m = extractProsodyMarks(segment_text);
        // Agregar un segmento sin modificar antes de esta palabra si hay espacio
        if (start_time > accumulatedStart && accumulatedStart < 9999.0) {
          modifications.push({
            type: 'prosody',
            start_time: accumulatedStart,
            end_time: start_time,
            pitch_shift:0,
            volume_change:0,
            speed_change:1.0
          });
        }

        // Agregar el segmento con modificación de prosodia
        const end_time = (i < matches.length - 1) ? matches[i+1].time : 9999.0; // Hasta el final si es el último
        modifications.push({
          type: 'prosody',
          start_time,
          end_time,
          pitch_shift: m.pitch_shift,
          volume_change: m.volume_change,
          speed_change: m.speed_change
        });

        // Actualizar el inicio acumulado
        accumulatedStart = end_time;
      }

      if (hasSilenceMarks(segment_text)) {
        const silenceDuration = extractSilenceMarks(segment_text);
        if (silenceDuration) {
          modifications.push({
            type: 'silence',
            start_time: accumulatedStart,
            duration: silenceDuration
          });
          accumulatedStart += silenceDuration;
        }
      }
      // Si no hay marcas, no hacemos nada (se acumulará en el siguiente segmento)
    }

    // Si no hay marcas de tiempo pero hay marcas de prosodia o silencio, asegurar modificación completa
    if (matches.length === 0) {
      if (hasProsodyMarks(text)) {
        const m = extractProsodyMarks(text);
        modifications.push({
          type: 'prosody',
          start_time: 0.0,
          end_time: 9999.0,
          pitch_shift: m.pitch_shift,
          volume_change: m.volume_change,
          speed_change: m.speed_change
        });
      }

      if (hasSilenceMarks(text)) {
        const silenceDuration = extractSilenceMarks(text);
        if (silenceDuration) {
          modifications.push({
            type: 'silence',
            start_time: 0.0,
            duration: silenceDuration
          });
        }
      }
    }

    return modifications;
  };

  const handleGenerarProsodia = async () => {
    if (!generatedAudio) {
      toast.error('No hay audio para generar prosodia');
      return;
    }

    if (!editableTranscription) {
      toast.error('No hay transcripción editable');
      return;
    }

    setIsProsodyGenerating(true);

    const modifications = parseModificationsFromText(editableTranscription);

    if (!modifications.length) {
      toast.error('No se detectaron suficientes marcas o tiempos.');
      setIsProsodyGenerating(false);
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/api/modify_prosody', {
        audio_path: generatedAudio,
        modifications
      });
      if (response.data.success) {
        setModifiedAudio(response.data.output_audio_path);
        setGeneratedAudios(prev => [...prev, response.data.output_audio_path]); // Añadir a la lista de audios generados
        toast.success(response.data.message || 'Prosodia generada con éxito');
      } else {
        toast.error(response.data.message || 'Error al generar prosodia');
      }
    } catch (error) {
      toast.error('Error al generar prosodia');
      console.error(error);
    } finally {
      setIsProsodyGenerating(false);
    }
  };

  // Nueva función para eliminar audios generados
  const handleDeleteAudio = async (audioPath) => {
    const confirmDelete = window.confirm('¿Estás seguro de que deseas eliminar este audio?');

    if (!confirmDelete) return;

    try {
      const response = await axios.post('http://localhost:5000/api/delete_audio', {
        audio_path: audioPath
      });

      if (response.data.success) {
        // Actualizar la lista de audios generados eliminando el audio eliminado
        setGeneratedAudios(prev => prev.filter(path => path !== audioPath));

        // Si el audio eliminado es el actualmente seleccionado, limpiar el estado
        if (generatedAudio === audioPath) {
          setGeneratedAudio(null);
        }

        if (modifiedAudio === audioPath) {
          setModifiedAudio(null);
        }

        toast.success('Audio eliminado correctamente');
      } else {
        toast.error(response.data.message || 'Error al eliminar el audio');
      }
    } catch (error) {
      toast.error('Error al eliminar el audio');
      console.error('Error:', error);
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          Generación de Múltiples Tipos de Habla
        </h2>
        
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Consejos:</h3>
          <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-4 rounded">
{`- Usa <velocity 1.5>, <pitch 2>, <volume 1.5>, <silencio 2> o <silencio 1.5> en las palabras que quieras modificar.
- Si colocas una marca antes de la primera (tiempo), se aplicará al audio completo.
- Las palabras sin marcas se acumulan en un solo segmento sin cortes.
- Las palabras con marca forman su propio segmento con crossfade corto.
- Escribe las marcas sin importar mayúsculas: <Velocity 1.5>, <velocity 1.5>, <Silencio 2> o <silencio 1.5> funcionarán igual.`}
          </pre>
          <div className="mt-4 flex space-x-2">
            <button
              onClick={() => handleInsertSilence(2.0)}
              className="px-3 py-1 bg-gray-300 text-gray-800 rounded hover:bg-gray-400 transition"
            >
              Insertar Silencio 2s
            </button>
            <button
              onClick={() => handleInsertSilence(1.5)}
              className="px-3 py-1 bg-gray-300 text-gray-800 rounded hover:bg-gray-400 transition"
            >
              Insertar Silencio 1.5s
            </button>
            {/* Puedes añadir más botones para diferentes duraciones si lo deseas */}
          </div>
        </div>

        <div className="space-y-6">
          {speechTypes.map((type) => (
            type.isVisible && (
              <SpeechTypeInput
                key={type.id}
                id={type.id}
                name={type.name}
                isRegular={type.id === 'regular'}
                onNameChange={(name) => handleNameUpdate(type.id, name)}
                onDelete={() => handleDeleteSpeechType(type.id)}
                onAudioUpload={(file, refText) => handleAudioUpload(type.id, file, refText, type.name)}
                onInsert={handleInsertSpeechType}
                uploadedAudio={audioData[type.id]?.audio}
                uploadedRefText={audioData[type.id]?.refText}
              />
            )
          ))}
        </div>

        <button
          onClick={handleAddSpeechType}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Agregar Tipo de Habla
        </button>

        <div className="mt-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Texto para Generar
          </label>
          <textarea
            value={generationText}
            onChange={(e) => setGenerationText(e.target.value)}
            className="w-full h-40 p-3 border rounded-md"
            placeholder="Ingresa el guion con los tipos de habla entre llaves y silencios entre <>..."
          />
        </div>

        <div className="mt-6 bg-gray-100 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">Configuraciones Avanzadas</h3>
          
          <div className="mb-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={removeSilence}
                onChange={(e) => setRemoveSilence(e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-700">Eliminar Silencios</span>
            </label>
            <p className="text-xs text-gray-500 mt-1">
              El modelo tiende a producir silencios... Esta opción los elimina.
            </p>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Velocidad ({speedChange.toFixed(1)}x)
            </label>
            <input
              type="range"
              step="0.1"
              min="0.3"
              max="2.0"
              value={speedChange}
              onChange={(e) => setSpeedChange(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Ajusta la velocidad del audio base. (1.0 = normal)
            </p>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Duración del Cross-Fade (s) ({crossFadeDuration.toFixed(2)}s)
            </label>
            <input
              type="range"
              step="0.05"
              min="0"
              max="1"
              value={crossFadeDuration}
              onChange={(e) => setCrossFadeDuration(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Un pequeño crossfade (ej: 0.05s) suaviza las transiciones entre segmentos con marca.
            </p>
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={`mt-6 px-6 py-3 bg-green-600 text-white rounded-lg font-medium
            ${isGenerating ? 'opacity-50 cursor-not-allowed' : 'hover:bg-green-700'} transition`}
        >
          {isGenerating ? 'Generando...' : 'Generar Habla Multi-Estilo'}
        </button>

        {/* Lista de Audios Generados con Opciones para Reproducir y Eliminar */}
        {generatedAudios.length > 0 && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Audios Generados:</h3>
            <ul className="space-y-4">
              {generatedAudios.map((audioPath, index) => (
                <li key={index} className="flex items-center justify-between bg-gray-100 p-4 rounded">
                  <div className="flex items-center space-x-4">
                    <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(audioPath)}`} />
                    <span className="text-sm text-gray-700">{audioPath.split('/').pop()}</span>
                  </div>
                  <button
                    onClick={() => handleDeleteAudio(audioPath)}
                    className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition"
                  >
                    Eliminar
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {generatedAudio && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Audio Generado Principal:</h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(generatedAudio)}`} />
            {!analyzeDone && (
              <button
                onClick={handleAnalyzeAudio}
                disabled={isAnalyzing}
                className={`mt-4 px-4 py-2 rounded text-white transition ${
                  isAnalyzing
                    ? 'bg-purple-400 cursor-not-allowed'
                    : 'bg-purple-600 hover:bg-purple-700'
                }`}
              >
                {isAnalyzing ? 'Generando Transcripción...' : 'Personalizar Audio'}
              </button>
            )}
          </div>
        )}

        {transcriptionData && transcriptionData.success && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Transcripción Generada (Editable):</h3>
            <textarea
              className="w-full h-64 p-3 border rounded-md whitespace-pre-wrap font-mono text-sm"
              value={editableTranscription}
              onChange={(e) => setEditableTranscription(e.target.value)}
            />
            <p className="text-xs text-gray-500 mt-1">
              - Coloca marcas &lt;pitch X&gt;, &lt;volume X&gt;, &lt;velocity X&gt;, &lt;silencio X&gt; solo en las palabras que quieras modificar.<br/>
              - Si colocas una marca antes de la primera (tiempo), se aplicará al audio completo.<br/>
              - Las palabras sin marcas se acumulan en un solo segmento sin cortes.<br/>
              - Las palabras con marca forman su propio segmento con crossfade corto.
            </p>
            <button
              onClick={handleGenerarProsodia}
              disabled={isProsodyGenerating}
              className={`mt-4 px-4 py-2 rounded text-white transition ${
                isProsodyGenerating ? 'bg-green-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isProsodyGenerating ? 'Generando Prosodia...' : 'Generar Prosodia'}
            </button>
          </div>
        )}

        {modifiedAudio && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Audio con Prosodia Aplicada:</h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(modifiedAudio)}`} />
            <button
              onClick={() => handleDeleteAudio(modifiedAudio)}
              className="mt-4 px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition"
            >
              Eliminar
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default MultiSpeechGenerator;
