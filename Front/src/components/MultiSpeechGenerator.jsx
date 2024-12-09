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

  // Nuevo estado para controlar el botón de "Personalizar Audio"
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeDone, setAnalyzeDone] = useState(false);

  const [audioData, setAudioData] = useState({
    regular: { audio: null, refText: '' }
  });

  const handleAddSpeechType = () => {
    if (speechTypes.length < MAX_SPEECH_TYPES) {
      const newId = `speech-type-${speechTypes.length}`;
      setSpeechTypes([
        ...speechTypes,
        { id: newId, name: '', isVisible: true }
      ]);
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
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setAudioData({
        ...audioData,
        [id]: { 
          audio: response.data.filepath,
          refText: refText,
          speechType: speechType
        }
      });

      toast.success('Audio cargado correctamente');
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

      const mentionedTypes = [...generationText.matchAll(/\{([^}]+)\}/g)]
        .map(match => match[1]);
      
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

      setGeneratedAudio(response.data.audio_path);
      setTranscriptionData(null);
      setAnalyzeDone(false); // Permitir personalizar si se desea
      toast.success('Audio generado correctamente');
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
        setAnalyzeDone(true); // Ya se obtuvo la transcripción, ocultar el botón
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

  return (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          Generación de Múltiples Tipos de Habla
        </h2>
        
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Ejemplos de Formato:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 p-4 rounded">
              <p className="font-medium mb-2">Ejemplo 1:</p>
              <pre className="whitespace-pre-wrap text-sm">
{`{Regular} Hola, me gustaría pedir un sándwich, por favor.
{Sorprendido} ¿Qué quieres decir con que no tienen pan?
{Triste} Realmente quería un sándwich...
{Enojado} ¡Sabes qué, maldición a ti y a tu pequeña tienda!
{Susurro} Solo volveré a casa y lloraré ahora.
{Gritando} ¿Por qué yo?!`}
              </pre>
            </div>
            
            <div className="bg-gray-50 p-4 rounded">
              <p className="font-medium mb-2">Ejemplo 2:</p>
              <pre className="whitespace-pre-wrap text-sm">
{`{Speaker1_Feliz} Hola, me gustaría pedir un sándwich, por favor.
{Speaker2_Regular} Lo siento, nos hemos quedado sin pan.
{Speaker1_Triste} Realmente quería un sándwich...
{Speaker2_Susurro} Te daré el último que estaba escondiendo.`}
              </pre>
            </div>
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
            placeholder="Ingresa el guion con los tipos de habla entre llaves..."
          />
        </div>

        <div className="mt-6 bg-gray-100 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-4">Configuraciones Avanzadas</h3>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Texto de Referencia
            </label>
            <textarea
              value={''}
              onChange={() => {}}
              className="w-full h-24 p-3 border rounded-md"
              placeholder="Deja en blanco para transcribir automáticamente el audio de referencia..."
              disabled
            />
            <p className="text-xs text-gray-500 mt-1">
              Deja en blanco para transcribir automáticamente el audio de referencia...
            </p>
          </div>

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
              Ajusta la velocidad del audio. (1.0 = normal)
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
              Establece la duración del cross-fade entre clips de audio.
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

        {generatedAudio && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Audio Generado:</h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${encodeURIComponent(generatedAudio)}`} />
            {/* Mostrar el botón "Personalizar Audio" solo si no se ha terminado el análisis */}
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
            <h3 className="text-lg font-medium mb-2">Transcripción Generada:</h3>
            <p>{transcriptionData.transcription}</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default MultiSpeechGenerator;
