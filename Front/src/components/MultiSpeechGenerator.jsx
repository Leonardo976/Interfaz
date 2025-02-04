// src/components/MultiSpeechGenerator.jsx

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import SpeechTypeInput from './SpeechTypeInput';
import AudioPlayer from './AudioPlayer';
import ProsodyModifier from './ProsodyModifier';

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
   * Si hay una marca antes de la primera marca de tiempo, se aplica al audio completo.
   */
  const parseModificationsFromText = (text) => {
    // Regex para encontrar marcas de tiempo
    const timePattern = /\((\d+\.\d+)\)/g;
    let matches = [];
    let match;
    while ((match = timePattern.exec(text)) !== null) {
      matches.push({time: parseFloat(match[1]), index: match.index});
    }

    const hasMarks = (segment_text) => {
      // Patrón insensible a mayúsculas/minúsculas para detectar marcas de prosodia
      const tagPattern = /<(pitch|volume|velocity)\s+([\d\.]+)>/i;
      return tagPattern.test(segment_text);
    };

    const extractMarks = (segment_text) => {
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

    let modifications = [];
    let accumulatedStart = 0.0;

    // Detectar si hay marcas antes de la primera marca de tiempo
    let firstTime = (matches.length > 0) ? matches[0].time : null;
    const textBeforeFirstTime = (matches.length > 0) ? text.substring(0, matches[0].index) : text;

    // Extraer marcas antes de la primera marca de tiempo
    let initialMarks = {pitch_shift:0, volume_change:0, speed_change:1.0};
    if (hasMarks(textBeforeFirstTime)) {
      initialMarks = extractMarks(textBeforeFirstTime);
      // Aplicar la modificación desde 0.0 hasta el final del audio
      modifications.push({
        start_time: 0.0,
        end_time: 9999.0, // Valor grande para indicar hasta el final; el backend lo ajustará
        pitch_shift: initialMarks.pitch_shift,
        volume_change: initialMarks.volume_change,
        speed_change: initialMarks.speed_change
      });
      accumulatedStart = 9999.0; // No habrá acumulación después
    }

    // Iterar sobre las marcas de tiempo
    for (let i = 0; i < matches.length; i++) {
      const start_time = matches[i].time;
      const start_index = matches[i].index;
      const end_index = (i < matches.length - 1) ? matches[i+1].index : text.length;
      const segment_text = text.substring(start_index, end_index);

      if (hasMarks(segment_text)) {
        // Extraer las marcas de esta palabra
        const m = extractMarks(segment_text);
        // Agregar un segmento sin modificar antes de esta palabra si hay espacio
        if (start_time > accumulatedStart && accumulatedStart < 9999.0) {
          modifications.push({
            start_time: accumulatedStart,
            end_time: start_time,
            pitch_shift:0,
            volume_change:0,
            speed_change:1.0
          });
        }

        // Agregar el segmento con modificación
        const end_time = (i < matches.length - 1) ? matches[i+1].time : 9999.0; // Hasta el final si es el último
        modifications.push({
          start_time,
          end_time,
          pitch_shift: m.pitch_shift,
          volume_change: m.volume_change,
          speed_change: m.speed_change
        });

        // Actualizar el inicio acumulado
        accumulatedStart = end_time;
      }
      // Si no hay marcas, no hacemos nada (se acumulará en el siguiente segmento)
    }

    // Si no hay marcas de tiempo pero hay marcas de prosodia, asegurar modificación completa
    if (matches.length === 0 && hasMarks(text)) {
      const m = extractMarks(text);
      modifications = [{
        start_time: 0.0,
        end_time: 9999.0, // Valor grande para indicar hasta el final; el backend lo ajustará
        pitch_shift: m.pitch_shift,
        volume_change: m.volume_change,
        speed_change: m.speed_change
      }];
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
      if (response.data.output_audio_path) {
        setModifiedAudio(response.data.output_audio_path);
        setGeneratedAudios(prev => [...prev, response.data.output_audio_path]); // Añadir a la lista de audios generados
        toast.success('Prosodia generada con éxito');
      } else {
        toast.error('Error al generar prosodia');
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
    <div className="space-y-8 max-w-6xl mx-auto px-4">
      <div className="bg-white p-8 rounded-2xl shadow-lg border border-gray-100">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-sky-600 mb-2 flex items-center">
            {/* Icono: Academic Cap (simula "chalkboard-teacher") */}
            <svg xmlns="http://www.w3.org/2000/svg" className="mr-3" width="24" height="24" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10 2L1 7l9 5 9-5-9-5z" />
              <path d="M1 7l9 5 9-5" />
              <path d="M1 13l9 5 9-5" />
            </svg>
            Generador de Contenido Audio Educativo
          </h1>
          <p className="text-gray-600">
            Herramienta para crear materiales auditivos con múltiples estilos de habla
          </p>
        </header>

        <div className="bg-sky-50 p-6 rounded-xl mb-8">
          <div className="flex items-start">
            {/* Icono: Lightbulb */}
            <svg xmlns="http://www.w3.org/2000/svg" className="mr-3 text-2xl text-sky-600 mt-1" width="24" height="24" viewBox="0 0 352 512" fill="currentColor">
              <path d="M96 0C43 0 0 43 0 96c0 41.7 25.3 77.2 60.3 91.6 3.8 1.8 6.7 5.2 7.4 9.3L80 232c0 13.3 10.7 24 24 24h48v64H88c-13.3 0-24 10.7-24 24v24h128v-24c0-13.3-10.7-24-24-24H112v-64h48c13.3 0 24-10.7 24-24l-12.3-34.1c-.7-4.1 3.6-7.5 7.4-9.3C326.7 173.2 352 137.7 352 96c0-53-43-96-96-96H96z" />
            </svg>
            <div>
              <h3 className="text-lg font-semibold mb-3 text-sky-800">Consejos Pedagógicos</h3>
              <pre className="whitespace-pre-wrap text-sm bg-white p-4 rounded-lg border border-sky-100">
{`- Utilice diferentes tonos para enfatizar conceptos clave
- Varíe la velocidad en secciones importantes
- Añada pausas estratégicas para reflexión
- Use efectos de volumen para mantener atención
- Combine estilos para dinámicas interactivas`}
              </pre>
            </div>
          </div>
        </div>

        <section className="space-y-6 mb-8">
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
        </section>

        <button
          onClick={handleAddSpeechType}
          className="w-full bg-emerald-100 hover:bg-emerald-200 text-emerald-700 py-3 rounded-xl font-semibold transition-all flex items-center justify-center"
        >
          {/* Icono: Plus Circle */}
          <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
            <path d="M256 8C119.033 8 8 119.033 8 256s111.033 248 248 248 248-111.033 248-248S392.967 8 256 8zm124 276h-100v100c0 13.255-10.745 24-24 24s-24-10.745-24-24V284H132c-13.255 0-24-10.745-24-24s10.745-24 24-24h100V136c0-13.255 10.745-24 24-24s24 10.745 24 24v100h100c13.255 0 24 10.745 24 24s-10.745 24-24 24z"/>
          </svg>
          Añadir Nuevo Estilo de Voz
        </button>

        <section className="mt-10">
          <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              {/* Icono: Pencil Alt */}
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
                <path d="M497.9 74.1l-60.1-60.1c-18.7-18.7-49.1-18.7-67.9 0L182.3 259.7l-23.3 92.9c-2.4 9.7 2.3 19.9 11.2 24.2 9.1 4.3 19.6 1.3 26.1-6.3L416 190.1l-60.1-60.1 141.9-141.9c18.8-18.8 18.8-49.2 0-68zM106.2 351.1l92.9-23.3 189.3-189.3-69.3-69.3L129.9 258.5l-23.7 92.6zM0 464c0 26.5 21.5 48 48 48h416c26.5 0 48-21.5 48-48v-16H0v16z"/>
              </svg>
              Composición del Material
            </h3>
            <textarea
              value={generationText}
              onChange={(e) => setGenerationText(e.target.value)}
              className="w-full h-48 p-4 border-2 border-gray-200 rounded-lg focus:border-sky-500 focus:ring-2 focus:ring-sky-200 resize-none"
              placeholder="Ej: {Regular} Bienvenidos a la clase de hoy. {Entusiasta} ¡Hoy aprenderemos algo increíble!"
            />
          </div>
        </section>

        <section className="mt-8 bg-orange-50 p-6 rounded-xl border border-orange-200">
          <h3 className="text-xl font-semibold text-orange-800 mb-5 flex items-center">
            {/* Icono: Sliders / Ajustes */}
            <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="21" x2="4" y2="14"></line>
              <line x1="4" y1="10" x2="4" y2="3"></line>
              <line x1="12" y1="21" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12" y2="3"></line>
              <line x1="20" y1="21" x2="20" y2="16"></line>
              <line x1="20" y1="12" x2="20" y2="3"></line>
              <line x1="1" y1="14" x2="7" y2="14"></line>
              <line x1="9" y1="8" x2="15" y2="8"></line>
              <line x1="17" y1="16" x2="23" y2="16"></line>
            </svg>
            Configuración Avanzada
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={removeSilence}
                  onChange={(e) => setRemoveSilence(e.target.checked)}
                  className="w-5 h-5 text-sky-600 rounded border-gray-300"
                />
                <span className="text-gray-700 font-medium">Eliminar silencios</span>
              </label>
              <p className="text-sm text-gray-500 ml-8">Optimiza el tiempo eliminando pausas</p>
            </div>

            <div className="space-y-4">
              <label className="block">
                <span className="text-gray-700 font-medium">Velocidad: {speedChange.toFixed(1)}x</span>
                <input
                  type="range"
                  step="0.1"
                  min="0.3"
                  max="2.0"
                  value={speedChange}
                  onChange={(e) => setSpeedChange(parseFloat(e.target.value))}
                  className="w-full mt-2 range-slider"
                />
              </label>
            </div>

            <div className="space-y-4">
              <label className="block">
                <span className="text-gray-700 font-medium">Transición: {crossFadeDuration.toFixed(2)}s</span>
                <input
                  type="range"
                  step="0.05"
                  min="0"
                  max="1"
                  value={crossFadeDuration}
                  onChange={(e) => setCrossFadeDuration(parseFloat(e.target.value))}
                  className="w-full mt-2 range-slider"
                />
              </label>
            </div>
          </div>
        </section>

        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={`w-full py-4 text-lg font-bold rounded-xl mt-8 transition-all ${
            isGenerating 
              ? 'bg-gray-400 cursor-not-allowed' 
              : 'bg-sky-600 hover:bg-sky-700 text-white'
          } flex items-center justify-center`}
        >
          {isGenerating ? (
            <>
              {/* Icono: Spinner */}
              <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Generando Material...
            </>
          ) : (
            <>
              {/* Icono: Magic Wand */}
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
                <path d="M510.4 17.7l-18.1-18.1c-8.5-8.5-22.4-8.5-30.9 0l-55.1 55.1c-15.5-6.3-32.2-9.8-49.5-9.8-58.6 0-106 47.4-106 106 0 17.3 3.5 33.9 9.8 49.5l-55.1 55.1c-8.5 8.5-8.5 22.4 0 30.9l18.1 18.1c8.5 8.5 22.4 8.5 30.9 0l55.1-55.1c15.5 6.3 32.2 9.8 49.5 9.8 58.6 0 106-47.4 106-106 0-17.3-3.5-33.9-9.8-49.5l55.1-55.1c8.5-8.5 8.5-22.4 0-30.9zM250.5 400.5l-70-70c-5.8-5.8-5.8-15.2 0-21l70-70c5.8-5.8 15.2-5.8 21 0l70 70c5.8 5.8 5.8 15.2 0 21l-70 70c-5.8 5.8-15.2 5.8-21 0z"/>
              </svg>
              Crear Contenido Multimedia
            </>
          )}
        </button>

        {generatedAudios.length > 0 && (
          <section className="mt-10">
            <h3 className="text-xl font-semibold mb-5 text-gray-800 flex items-center">
              {/* Icono: History */}
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
                <path d="M504 256c0 136.967-111.033 248-248 248S8 392.967 8 256 119.033 8 256 8c76.034 0 142.021 38.596 183.293 96h-79.293v56h136v-136h-56v79.293C415.404 113.979 471.106 192.3 504 256zM256 464c114.875 0 208-93.125 208-208S370.875 48 256 48 48 141.125 48 256s93.125 208 208 208zM272 144v128h-64v64h128V144h-64z"/>
              </svg>
              Historial de Generaciones
            </h3>
            <div className="space-y-4">
              {generatedAudios.map((audioPath, index) => (
                <div key={index} className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(audioPath)}`} />
                    <span className="text-sm text-gray-600 font-mono">{audioPath.split('/').pop()}</span>
                  </div>
                  <button
                    onClick={() => handleDeleteAudio(audioPath)}
                    className="px-3 py-1 bg-red-100 hover:bg-red-200 text-red-600 rounded-lg transition-colors flex items-center"
                  >
                    {/* Icono: Trash Alt */}
                    <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 448 512" fill="currentColor">
                      <path d="M135.2 17.7L144 0h160l8.8 17.7L416 32H32l103.2-14.3zM400 96v368c0 26.5-21.5 48-48 48H96c-26.5 0-48-21.5-48-48V96h352z"/>
                    </svg>
                    Eliminar
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        {generatedAudio && (
          <div className="mt-8">
            <h3 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
              {/* Icono: Volume Up */}
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <line x1="23" y1="9" x2="23" y2="15"></line>
                <path d="M16 9.35a8 8 0 0 1 0 5.3"></path>
              </svg>
              Audio Generado Principal
            </h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(generatedAudio)}`} />
            {!analyzeDone && (
              <button
                onClick={handleAnalyzeAudio}
                disabled={isAnalyzing}
                className={`mt-4 px-6 py-2 rounded-lg text-white transition ${
                  isAnalyzing
                    ? 'bg-purple-400 cursor-not-allowed'
                    : 'bg-purple-600 hover:bg-purple-700'
                } flex items-center`}
              >
                {/* Icono: Microscope */}
                <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 576 512" fill="currentColor">
                  <path d="M564.8 82.2c-12.1-12.1-31.8-12.1-43.9 0L448 155.1V72c0-13.3-10.7-24-24-24h-48c-13.3 0-24 10.7-24 24v83.1L195.1 82.2c-12.1-12.1-31.8-12.1-43.9 0L5.4 228.9c-12.1 12.1-12.1 31.8 0 43.9l60.1 60.1c12.1 12.1 31.8 12.1 43.9 0l89.3-89.3v140.2c0 13.3 10.7 24 24 24h48c13.3 0 24-10.7 24-24V242.7l89.3 89.3c12.1 12.1 31.8 12.1 43.9 0l60.1-60.1c12.1-12.1 12.1-31.8 0-43.9L564.8 82.2z"/>
                </svg>
                {isAnalyzing ? 'Analizando...' : 'Personalizar Audio'}
              </button>
            )}
          </div>
        )}

        {transcriptionData && transcriptionData.success && (
          <ProsodyModifier transcriptionData={transcriptionData} generatedAudio={generatedAudio} />
        )}

        {modifiedAudio && (
          <div className="mt-8">
            <h3 className="text-xl font-semibold mb-4 text-gray-800 flex items-center">
              {/* Icono: Waveform Lines */}
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 640 512" fill="currentColor">
                <path d="M64 96v320c0 17.7 14.3 32 32 32h32c17.7 0 32-14.3 32-32V96c0-17.7-14.3-32-32-32H96C78.3 64 64 78.3 64 96zM256 160v192c0 17.7 14.3 32 32 32h32c17.7 0 32-14.3 32-32V160c0-17.7-14.3-32-32-32h-32c-17.7 0-32 14.3-32 32zM448 224v128c0 17.7 14.3 32 32 32h32c17.7 0 32-14.3 32-32V224c0-17.7-14.3-32-32-32h-32c-17.7 0-32 14.3-32 32z"/>
              </svg>
              Audio con Prosodia Aplicada
            </h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(modifiedAudio)}`} />
            <button
              onClick={() => handleDeleteAudio(modifiedAudio)}
              className="mt-4 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-600 rounded-lg transition-colors flex items-center"
            >
              {/* Icono: Trash Alt */}
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 448 512" fill="currentColor">
                <path d="M135.2 17.7L144 0h160l8.8 17.7L416 32H32l103.2-14.3zM400 96v368c0 26.5-21.5 48-48 48H96c-26.5 0-48-21.5-48-48V96h352z"/>
              </svg>
              Eliminar Versión
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default MultiSpeechGenerator;
