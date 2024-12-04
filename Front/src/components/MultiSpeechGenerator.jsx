// src/components/MultiSpeechGenerator.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';
import SpeechTypeInput from './SpeechTypeInput';
import AudioPlayer from './AudioPlayer';
import ProsodyModifier from './ProsodyModifier'; // Importar el subcomponente

const MAX_SPEECH_TYPES = 100;

function MultiSpeechGenerator() {
  // Estados para los tipos de habla
  const [speechTypes, setSpeechTypes] = useState([
    { id: 'regular', name: 'Regular', isVisible: true }
  ]);
  const [generationText, setGenerationText] = useState('');
  const [removeSilence, setRemoveSilence] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudio, setGeneratedAudio] = useState(null);
  const [modifiedAudio, setModifiedAudio] = useState(null); // Nuevo estado para audio modificado

  // Estado para almacenar los datos de audio de referencia
  const [audioData, setAudioData] = useState({
    regular: { audio: null, refText: '' }
  });

  // Estado para almacenar modificaciones de prosodia
  const [prosodyModifications, setProsodyModifications] = useState([]);

  // Función para agregar un nuevo tipo de habla
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

  // Función para eliminar un tipo de habla
  const handleDeleteSpeechType = (idToDelete) => {
    setSpeechTypes(speechTypes.filter(type => type.id !== idToDelete));
    const newAudioData = { ...audioData };
    delete newAudioData[idToDelete];
    setAudioData(newAudioData);
  };

  // Función para actualizar el nombre de un tipo de habla
  const handleNameUpdate = (id, newName) => {
    setSpeechTypes(speechTypes.map(type => 
      type.id === id ? { ...type, name: newName } : type
    ));
  };

  // Función para manejar la carga de archivos de audio
  const handleAudioUpload = async (id, file, refText, speechType) => {
    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('speechType', speechType);
  
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

  // Función para insertar el tipo de habla en el texto
  const handleInsertSpeechType = (name) => {
    setGenerationText(prev => `${prev}{${name}} `);
  };

  // Función para generar el audio multi-estilo
  const handleGenerate = async () => {
    try {
      setIsGenerating(true);

      // Validar que todos los tipos de habla mencionados tengan audio
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

      // Preparar datos para la API
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
        remove_silence: removeSilence
      });

      setGeneratedAudio(response.data.audio_path);
      setModifiedAudio(null); // Resetear el audio modificado
      toast.success('Audio generado correctamente');
    } catch (error) {
      toast.error('Error al generar el audio');
      console.error('Error:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  // Función para manejar las modificaciones de prosodia
  const handleApplyProsody = async (modifications) => {
    if (!generatedAudio) {
      toast.error('No hay audio generado para modificar');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/api/modify_prosody', {
        audio_path: generatedAudio,
        modifications: modifications.map(mod => ({
          start_time: mod.start_time,
          end_time: mod.end_time,
          pitch_shift: mod.pitch_shift,
          volume_change: mod.volume_change,
          speed_change: mod.speed_change
        }))
      });

      setModifiedAudio(response.data.output_audio_path);
      toast.success('Prosodia modificada correctamente');
    } catch (error) {
      toast.error('Error al modificar la prosodia');
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

        {/* Tipos de habla */}
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

        {/* Botón para agregar tipo de habla */}
        <button
          onClick={handleAddSpeechType}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Agregar Tipo de Habla
        </button>

        {/* Área de texto para generación */}
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

        {/* Configuraciones avanzadas */}
        <div className="mt-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={removeSilence}
              onChange={(e) => setRemoveSilence(e.target.checked)}
              className="rounded border-gray-300"
            />
            <span className="text-sm text-gray-700">Eliminar Silencios</span>
          </label>
        </div>

        {/* Botón de generación */}
        <button
          onClick={handleGenerate}
          disabled={isGenerating}
          className={`mt-6 px-6 py-3 bg-green-600 text-white rounded-lg font-medium
            ${isGenerating ? 'opacity-50 cursor-not-allowed' : 'hover:bg-green-700'} 
            transition`}
        >
          {isGenerating ? 'Generando...' : 'Generar Habla Multi-Estilo'}
        </button>

        {/* Reproductor de audio generado */}
        {generatedAudio && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Audio Generado:</h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${generatedAudio}`} />
          </div>
        )}

        {/* Componente para Modificar Prosodia */}
        {generatedAudio && (
          <ProsodyModifier onAddModification={handleApplyProsody} />
        )}

        {/* Reproductor de audio modificado */}
        {modifiedAudio && (
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Audio Modificado:</h3>
            <AudioPlayer audioUrl={`http://localhost:5000/api/get_audio/${modifiedAudio}`} />
          </div>
        )}
      </div>
    </div>
  );
}

export default MultiSpeechGenerator;
