// src/components/MultiSpeechGenerator.jsx

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import SpeechTypeInput from './SpeechTypeInput';
import AudioPlayer from './AudioPlayer';
import ProsodyModifier from './ProsodyModifier';

const MAX_SPEECH_TYPES = 100;
const GUIDE_TEXT = `La Revolución Científica del siglo 17 transformó nuestra comprensión del universo. Figuras como Galileo Galilei, nacido en 1564, demostraron que la Tierra orbita alrededor del Sol, 
desafiando teorías geocéntricas. Isaac Newton formuló las tres leyes del movimiento y la ley de gravitación universal. Estos avances sentaron las bases para la física moderna y el método científico experimental.`;
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
  const [pendingUpload, setPendingUpload] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeDone, setAnalyzeDone] = useState(false);

  const [editableTranscription, setEditableTranscription] = useState('');
  const [isProsodyGenerating, setIsProsodyGenerating] = useState(false);
  const [modifiedAudio, setModifiedAudio] = useState(null);

  const refTextRefs = useRef({});
  const [audioData, setAudioData] = useState({
    regular: { audio: null, refText: '' }
  });

  // Cambiar a useRef para los chunks de grabación
  const recordedChunksRef = useRef([]);
  const [isRecording, setIsRecording] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [recordTimer, setRecordTimer] = useState(0);
  const recordTimerRef = useRef(null);
  const customRefProvidedRef = useRef(false);

  // Estado para manejar la lista de audios generados
  const [generatedAudios, setGeneratedAudios] = useState([]);

// Agrega este estado al componente
const [showRecordingText, setShowRecordingText] = useState(false);

const latestAudioDataRef = useRef(audioData);
useEffect(() => {
  latestAudioDataRef.current = audioData;
}, [audioData]);


const handleSetGuideTextForType = (id) => {
  setAudioData(prev => ({
    ...prev,
    [id]: {
      ...prev[id],
      refText: GUIDE_TEXT
    }
  }));
  if (refTextRefs.current[id]) {
    refTextRefs.current[id].value = GUIDE_TEXT;
  }
  customRefProvidedRef.current = true; // Marcamos que se proporcionó texto custom
  toast.success("Texto guía agregado como referencia");
};


const handleConfirmUpload = async () => {
  if (!pendingUpload) return;
  
  const { speechTypeId, file } = pendingUpload;
  // Leer el valor actual del textarea usando la ref y aplicar trim
  let currentRefText = refTextRefs.current[speechTypeId]?.value || '';
  currentRefText = currentRefText.trim();
  
  // Si el texto ingresado es el mismo que el texto guía o está vacío, forzamos que se envíe vacío para transcribir
  const refText = (currentRefText === GUIDE_TEXT.trim() || currentRefText === "") 
                    ? "" 
                    : currentRefText;
  
  const styleName = findSpeechTypeNameById(speechTypeId);
  
  await handleAudioUpload(speechTypeId, file, refText, styleName);
  
  // Actualizar el estado para guardar el texto (vacío o el custom)
  setAudioData(prev => ({
    ...prev,
    [speechTypeId]: {
      ...prev[speechTypeId],
      refText: refText
    }
  }));
  
  setPendingUpload(null);
};

// ----------------------------------------------------------------
// Función para iniciar grabación (versión completa actualizada)
// ----------------------------------------------------------------
const startRecording = async (speechTypeId) => {
  customRefProvidedRef.current = false;

  try {
    // Validación de nombre para nuevos tipos de habla
    if (speechTypeId !== 'regular') {
      const currentType = speechTypes.find(type => type.id === speechTypeId);
      if (!currentType?.name?.trim()) {
        toast.error('Debe ingresar un nombre para el estilo antes de grabar');
        return;
      }
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      toast.error('La API de grabación no está soportada en este navegador.');
      return;
    }


    // Iniciar flujo de grabación
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 44100,
        channelCount: 1,
        noiseSuppression: true,
        echoCancellation: true
      }
    });

    // Mostrar texto guía
    setShowRecordingText(true);
    setRecordTimer(0);

    const options = { 
      mimeType: 'audio/webm;codecs=opus',
      audioBitsPerSecond: 128000
    };
    
    const recorder = new MediaRecorder(stream, options);
    recordedChunksRef.current = [];

    // Configurar handlers
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        recordedChunksRef.current.push(e.data);
      }
    };

    recorder.onstop = async () => {
      setShowRecordingText(false);
      
      const blob = new Blob(recordedChunksRef.current, { 
        type: 'audio/webm;codecs=opus' 
      });
      
      const convertedBlob = await convertWebmToWav(blob);
      
      const file = new File([convertedBlob], 'grabacion.wav', {
        type: 'audio/wav'
      });
      
      // En lugar de subir, guardamos el archivo y el id en pendingUpload
      setPendingUpload({
        speechTypeId,
        file
      });
      
      recordedChunksRef.current = [];
      stream.getTracks().forEach(track => track.stop());
    };
    

    // Iniciar temporizador y grabación
    recorder.start(1000);
    setMediaRecorder(recorder);
    setIsRecording(speechTypeId);
    setIsPaused(false);

    recordTimerRef.current = setInterval(() => {
      setRecordTimer((prev) => prev + 1);
    }, 1000);

    toast.success('Grabación iniciada - Lea el texto en pantalla');

  } catch (error) {
    console.error('Error en grabación:', error);
    setShowRecordingText(false);
    toast.error('Error al iniciar la grabación');
    if (mediaRecorder) {
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
  }
};

// Función para convertir WebM a WAV
const convertWebmToWav = async (webmBlob) => {
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const arrayBuffer = await webmBlob.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const wavBuffer = audioBufferToWav(audioBuffer);
    return new Blob([wavBuffer], { type: 'audio/wav' });
  } finally {
    audioContext.close();
  }
};

// Función para convertir AudioBuffer a WAV
const audioBufferToWav = (buffer) => {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 3; // Float32
  const bitDepth = 32;

  const bufferHeader = new ArrayBuffer(44);
  const view = new DataView(bufferHeader);

  // Encabezado WAV
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + buffer.length * numChannels * 4, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, format, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 4, true);
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  view.setUint16(34, bitDepth, true);
  writeString(view, 36, 'data');
  view.setUint32(40, buffer.length * numChannels * 4, true);

  // Datos de audio intercalados
  const interleaved = new Float32Array(buffer.length * numChannels);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < buffer.length; i++) {
      interleaved[i * numChannels + channel] = channelData[i];
    }
  }

  // Combinar header y datos
  const wavBuffer = new Uint8Array(bufferHeader.byteLength + interleaved.byteLength);
  wavBuffer.set(new Uint8Array(bufferHeader), 0);
  wavBuffer.set(new Uint8Array(interleaved.buffer), bufferHeader.byteLength);
  
  return wavBuffer;
};

const writeString = (view, offset, string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};
  
  // --------------------------------------------------------------
  // Función para pausar y reanudar grabación
  // --------------------------------------------------------------
  const pauseRecording = () => {
    if (!mediaRecorder) return;
    
    if (!isPaused) {
      mediaRecorder.pause();
      setIsPaused(true);
      toast('Grabación en pausa');
    } else {
      mediaRecorder.resume();
      setIsPaused(false);
      toast('Grabación reanudada');
    }
  };

  // --------------------------------
  // Función para detener la grabación
  // --------------------------------
  const stopRecording = () => {
    if (!mediaRecorder) return;

    mediaRecorder.stop();
    clearInterval(recordTimerRef.current);
    recordTimerRef.current = null;
    setRecordTimer(0);
    setIsRecording(null);
    setIsPaused(false);
    setMediaRecorder(null);
  };


  // ------------------------------------------------------
  // Encuentra el nombre (p.e. "Regular") de un speechType
  // ------------------------------------------------------
  const findSpeechTypeNameById = (id) => {
    const st = speechTypes.find((t) => t.id === id);
    return st ? st.name : '';
  };

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
      // Revisar si para el tipo "regular" (o el que corresponda) se tiene un texto de referencia custom
      const customText = audioData["regular"]?.refText || "";
      if (customText.trim() === "") {
        // No se proporcionó texto custom, se usa la transcripción
        setEditableTranscription(transcriptionData.transcription);
      } else {
        // Se proporcionó un texto custom; se usará ese valor
        setEditableTranscription(customText.trim());
      }
      setModifiedAudio(null);
    }
  }, [transcriptionData, audioData]);

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
        end_time: 9999.0,
        pitch_shift: initialMarks.pitch_shift,
        volume_change: initialMarks.volume_change,
        speed_change: initialMarks.speed_change
      });
      accumulatedStart = 9999.0;
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
        // Agregar un segmento "sin modificar" antes de esta palabra si hay espacio
        if (start_time > accumulatedStart && accumulatedStart < 9999.0) {
          modifications.push({
            start_time: accumulatedStart,
            end_time: start_time,
            pitch_shift:0,
            volume_change:0,
            speed_change:1.0
          });
        }

        // Segmento con modificación
        const end_time = (i < matches.length - 1) ? matches[i+1].time : 9999.0;
        modifications.push({
          start_time,
          end_time,
          pitch_shift: m.pitch_shift,
          volume_change: m.volume_change,
          speed_change: m.speed_change
        });
        accumulatedStart = end_time;
      }
      // Si no hay marcas en el segmento, no añadimos nada
    }

    // Si no hay marcas de tiempo pero sí de prosodia, aplicar al audio completo
    if (matches.length === 0 && hasMarks(text)) {
      const m = extractMarks(text);
      modifications = [{
        start_time: 0.0,
        end_time: 9999.0,
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
        setGeneratedAudios(prev => [...prev, response.data.output_audio_path]);
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

  // Eliminar audios generados
  const handleDeleteAudio = async (audioPath) => {
    const confirmDelete = window.confirm('¿Estás seguro de que deseas eliminar este audio?');

    if (!confirmDelete) return;

    try {
      const response = await axios.post('http://localhost:5000/api/delete_audio', {
        audio_path: audioPath
      });

      if (response.data.success) {
        setGeneratedAudios(prev => prev.filter(path => path !== audioPath));

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
            {/* Icono */}
            <svg xmlns="http://www.w3.org/2000/svg" className="mr-3" width="24" height="24" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10 2L1 7l9 5 9-5-9-5z" />
              <path d="M1 7l9 5 9-5" />
              <path d="M1 13l9 5 9-5" />
            </svg>
            Generador de contenido de audio educativo
          </h1>
          <p className="text-gray-600">
            Herramienta para crear materiales auditivos con múltiples estilos de habla
          </p>
        </header>

        <div className="bg-sky-50 p-6 rounded-xl mb-8">
          <div className="flex items-start">
            {/* Icono */}
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

        {/* LISTA DE SpeechTypes */}
        <section className="space-y-6 mb-8">
          {speechTypes.map((type) => (
            type.isVisible && (
              <div key={type.id} className="p-4 border border-gray-200 rounded-lg bg-gray-50">
                <div className="flex justify-between items-center">
                  <div className="flex-1">
                    <label className="block text-sm font-semibold text-gray-700 mb-1">Nombre del Estilo</label>
                    <input
                      type="text"
                      className="w-full p-2 border border-gray-300 rounded"
                      value={type.name}
                      onChange={(e) => handleNameUpdate(type.id, e.target.value)}
                      disabled={type.id === 'regular'}
                    />
                  </div>
                  {type.id !== 'regular' && (
                    <button
                      onClick={() => handleDeleteSpeechType(type.id)}
                      className="ml-4 px-3 py-2 text-red-500 bg-red-100 hover:bg-red-200 rounded-lg transition-colors"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 448 512" fill="currentColor">
                        <path d="M135.2 17.7L144 0h160l8.8 17.7L416 32H32l103.2-14.3zM400 96v368c0 26.5-21.5 48-48 48H96c-26.5 0-48-21.5-48-48V96h352z"/>
                      </svg>
                    </button>
                  )}
                </div>

                <div className="mt-4">
                  <div className="flex items-center justify-between">
                    <label className="block text-sm font-semibold text-gray-700 mb-1">
                      Texto de Referencia
                    </label>
                    <button
                      onClick={() => handleSetGuideTextForType(type.id)}
                      className="ml-2 px-2 py-1 bg-blue-100 hover:bg-blue-200 text-blue-600 rounded-lg text-xs"
                    >
                      Agregar texto guía
                    </button>
                  </div>
                  <textarea
                    ref={(el) => (refTextRefs.current[type.id] = el)}
                    className="w-full p-2 border border-gray-300 rounded"
                    rows="2"
                    value={audioData[type.id]?.refText || ''}
                    onChange={(e) =>
                      setAudioData({
                        ...audioData,
                        [type.id]: {
                          ...audioData[type.id],
                          refText: e.target.value
                        }
                      })
                    }
                  />
                </div>

                <div className="mt-4 flex items-center space-x-3">
                  <div className="flex flex-col">
                    <label className="block text-sm font-semibold text-gray-700 mb-1">Archivo de Audio</label>
                    <input
                      type="file"
                      onChange={(e) => {
                        if (e.target.files && e.target.files.length > 0) {
                          handleAudioUpload(
                            type.id,
                            e.target.files[0],
                            audioData[type.id]?.refText || '',
                            type.name
                          );
                        }
                      }}
                    />
                  </div>

                  {/* Botón "Grabar voz" en lugar de la bola verde */}
                  <div className="flex flex-col items-center">
                    {
                      (!isRecording || isRecording !== type.id) ? (
                        <button
                          onClick={() => startRecording(type.id)}
                          className={`rounded-full text-white px-4 py-2 flex items-center ${
                            type.id !== 'regular' && !type.name.trim() ? 
                              'bg-gray-400 cursor-not-allowed' : 
                              'bg-green-500 hover:bg-green-600'
                          }`}
                          title="Iniciar Grabación"
                          disabled={type.id !== 'regular' && !type.name.trim()}
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="16" height="16" viewBox="0 0 512 512" fill="currentColor">
                            <path d="M192 352v-96c0-53 43-96 96-96s96 43 96 96v96c0 53-43 96-96 96s-96-43-96-96zM416 208c0-77.4-62.6-140-140-140s-140 62.6-140 140v96c0 77.4 62.6 140 140 140s140-62.6 140-140v-96z"/>
                          </svg>
                          Grabar voz
                        </button>
                      ) : (
                        <div className="flex space-x-2">
                          {/* Botón Pausar/Reanudar */}
                          <button
                            onClick={pauseRecording}
                            className="bg-yellow-400 hover:bg-yellow-500 text-white px-3 py-2 rounded-lg flex items-center"
                          >
                            {isPaused ? (
                              <>
                                <svg xmlns="http://www.w3.org/2000/svg" className="mr-1" width="16" height="16" fill="currentColor" viewBox="0 0 512 512">
                                  <path d="M96 64l320 192-320 192z"/>
                                </svg>
                                Reanudar
                              </>
                            ) : (
                              <>
                                <svg xmlns="http://www.w3.org/2000/svg" className="mr-1" width="16" height="16" fill="currentColor" viewBox="0 0 448 512">
                                  <path d="M144 63h-32C50.1 63 0 113.1 0 175v162c0 61.9 50.1 112 112 112h32c61.9 0 112-50.1 112-112V175c0-61.9-50.1-112-112-112zM368 63h-32c-61.9 0-112 50.1-112 112v162c0 61.9 50.1 112 112 112h32c61.9 0 112-50.1 112-112V175c0-61.9-50.1-112-112-112z"/>
                                </svg>
                                Pausar
                              </>
                            )}
                          </button>
                          {/* Botón Detener */}
                          <button
                            onClick={stopRecording}
                            className="bg-red-500 hover:bg-red-600 text-white px-3 py-2 rounded-lg flex items-center"
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" className="mr-1" width="16" height="16" viewBox="0 0 448 512" fill="currentColor">
                              <path d="M400 32H48C21.49 32 0 53.49 0 80v352c0 26.5 21.49 48 48 48h352c26.51 0 48-21.5 48-48V80C448 53.49 426.51 32 400 32z"/>
                            </svg>
                            Detener
                          </button>
                        </div>
                      )
                    }
                    {/* Contador de tiempo */}
                    {isRecording === type.id && (
                      <div className="text-sm text-gray-500 mt-1">
                        Grabando: {recordTimer}s
                      </div>
                    )}
                  </div>
                  
                  {/* Insertar etiqueta en el texto principal */}
                  <button
                    onClick={() => handleInsertSpeechType(type.name)}
                    className="ml-auto px-3 py-2 bg-blue-100 hover:bg-blue-200 text-blue-600 rounded-lg transition-colors"
                  >
                    Insertar
                  </button>
                </div>

                {/* Mostrar si el audio está subido */}
                {audioData[type.id]?.audio && (
                  <p className="mt-2 text-sm text-green-600">
                    Audio subido: {audioData[type.id].audio}
                  </p>
                )}
              </div>
            )
          ))}
        </section>

        <button
          onClick={handleAddSpeechType}
          className="w-full bg-emerald-100 hover:bg-emerald-200 text-emerald-700 py-3 rounded-xl font-semibold transition-all flex items-center justify-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
            <path d="M256 8C119.033 8 8 119.033 8 256s111.033 248 248 248 248-111.033 248-248S392.967 8 256 8zm124 276h-100v100c0 13.255-10.745 24-24 24s-24-10.745-24-24V284H132c-13.255 0-24-10.745-24-24s10.745-24 24-24h100V136c0-13.255 10.745-24 24-24s24 10.745 24 24v100h100c13.255 0 24 10.745 24 24s-10.745 24-24 24z"/>
          </svg>
          Añadir Nuevo Estilo de Voz
        </button>

        <section className="mt-10">
          <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
            <h3 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
                <path d="M497.9 74.1l-60.1-60.1c-18.7-18.7-49.1-18.7-67.9 0L182.3 259.7l-23.3 92.9c-2.4 9.7 2.3 19.9 11.2 24.2 9.1 4.3 19.6 1.3 26.1-6.3L416 190.1l-60.1-60.1 141.9-141.9c18.8-18.8 18.8-49.2 0-68zM106.2 351.1l92.9-23.3 189.3-189.3-69.3-69.3L129.9 258.5l-23.7 92.6zM0 464c0 26.5 21.5 48 48 48h416c26.5 0 48-21.5 48-48v-16H0v16z"/>
              </svg>
              Composición del Material
            </h3>
            <h3 className="whitespace-pre-wrap text-sm bg-white p-4 rounded-lg border border-sky-100">
              (Para agregar silencios solo agregue puntos suspensivos ...)
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
                <span className="text-gray-700 font-medium">Velocidad: {speedChange.toFixed(1)}x (normal: 1.0)</span>
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
                <span className="text-gray-700 font-medium">Transición: {crossFadeDuration.toFixed(2)}s (normal: 0.15s)</span>
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
              <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10"
                  stroke="currentColor" strokeWidth="4" fill="none"
                />
                <path className="opacity-75" fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 
                  5.373 0 12h4zm2 5.291A7.962 7.962 0 
                  014 12H0c0 3.042 1.135 5.824 3 
                  7.938l3-2.647z"
                />
              </svg>
              Generando Material...
            </>
          ) : (
            <>
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
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 512 512" fill="currentColor">
                <path d="M504 256c0 136.967-111.033 248-248 248S8 392.967 8 256 119.033 8 
                256 8c76.034 0 142.021 38.596 183.293 96h-79.293v56h136v-136h-56v79.293C415.404 
                113.979 471.106 192.3 504 256zM256 464c114.875 0 208-93.125 208-208S370.875 
                48 256 48 48 141.125 48 256s93.125 208 208 208zM272 144v128h-64v64h128V144h-64z"/>
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
                    <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 448 512" fill="currentColor">
                      <path d="M135.2 17.7L144 0h160l8.8 17.7L416 32H32l103.2-14.3zM400 
                      96v368c0 26.5-21.5 48-48 48H96c-26.5 0-48-21.5-48-48V96h352z"/>
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
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24"
                viewBox="0 0 24 24" fill="none" stroke="currentColor"
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
                <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24"
                  viewBox="0 0 576 512" fill="currentColor">
                  <path d="M564.8 82.2c-12.1-12.1-31.8-12.1-43.9 
                  0L448 155.1V72c0-13.3-10.7-24-24-24h-48c-13.3 
                  0-24 10.7-24 24v83.1L195.1 82.2c-12.1-12.1-31.8-12.1-43.9 
                  0L5.4 228.9c-12.1 12.1-12.1 31.8 0 43.9l60.1 60.1c12.1 12.1 
                  31.8 12.1 43.9 0l89.3-89.3v140.2c0 13.3 10.7 24 24 24h48c13.3 
                  0 24-10.7 24-24V242.7l89.3 89.3c12.1 12.1 31.8 12.1 43.9 
                  0l60.1-60.1c12.1-12.1 12.1-31.8 0-43.9L564.8 82.2z"/>
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
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24"
                height="24" viewBox="0 0 640 512" fill="currentColor">
                <path d="M64 96v320c0 17.7 14.3 32 32 32h32c17.7 
                0 32-14.3 32-32V96c0-17.7-14.3-32-32-32H96C78.3 
                64 64 78.3 64 96zM256 160v192c0 17.7 14.3 
                32 32 32h32c17.7 0 32-14.3 32-32V160c0-17.7-14.3-32-32-32h-32c-17.7 
                0-32 14.3-32 32zM448 224v128c0 17.7 14.3 32 
                32 32h32c17.7 0 32-14.3 32-32V224c0-17.7-14.3-32-32-32h-32c-17.7 
                0-32 14.3-32 32z"/>
              </svg>
              Audio con Prosodia Aplicada
            </h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(modifiedAudio)}`} />
            <button
              onClick={() => handleDeleteAudio(modifiedAudio)}
              className="mt-4 px-4 py-2 bg-red-100 hover:bg-red-200 text-red-600 rounded-lg transition-colors flex items-center"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24"
                height="24" viewBox="0 0 448 512" fill="currentColor">
                <path d="M135.2 17.7L144 
                0h160l8.8 17.7L416 32H32l103.2-14.3zM400 
                96v368c0 26.5-21.5 48-48 48H96c-26.5 
                0-48-21.5-48-48V96h352z"/>
              </svg>
              Eliminar Versión
            </button>
          </div>
        )}
        {showRecordingText && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white p-8 rounded-lg max-w-2xl max-h-[90vh] overflow-auto">
            <h3 className="text-xl font-bold mb-4 text-blue-600">
              Texto para leer durante la grabación
            </h3>
            <p className="text-lg leading-relaxed text-gray-800 whitespace-pre-line">
              "La Revolución Científica del siglo 17 transformó nuestra comprensión del universo. Figuras como Galileo Galilei, nacido en 1564, demostraron que la Tierra orbita alrededor del Sol, desafiando teorías geocéntricas. Isaac Newton formuló las tres leyes del movimiento y la ley de gravitación universal. Estos avances sentaron las bases para la física moderna y el método científico experimental."
            </p>
            <div className="mt-6 flex justify-between items-center">
              <p className="text-sm text-gray-500">
                Tiempo de grabación: {recordTimer}s
              </p>
              <button
                onClick={() => setShowRecordingText(false)}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg"
              >
                Cerrar Texto (la grabación continuará)
              </button>
            </div>
          </div>
        </div>
      )}
        {pendingUpload && (
          <div className="fixed bottom-4 right-4 p-4 bg-white rounded shadow-lg border border-gray-300">
            <p className="mb-2 text-sm text-gray-700">
              Audio grabado pendiente de subir. Revise y, si desea, agregue o modifique el texto de referencia.
            </p>
            <button
              onClick={handleConfirmUpload}
              className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded"
            >
              Confirmar audio
            </button>
          </div>
        )}
      </div>
    </div>
    
  );
}

export default MultiSpeechGenerator;
