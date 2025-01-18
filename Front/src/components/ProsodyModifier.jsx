// src/components/ProsodyModifier.jsx
import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import axios from 'axios';
import toast from 'react-hot-toast';

function ProsodyModifier({ transcriptionData, generatedAudio }) {
  const [words, setWords] = useState([]);
  const [globalEffects, setGlobalEffects] = useState({ speed: null, pitch: null, volume: null });
  const [isProsodyGenerating, setIsProsodyGenerating] = useState(false);
  const [modifiedAudio, setModifiedAudio] = useState(null);

  useEffect(() => {
    if (transcriptionData && transcriptionData.transcription) {
      let wordArray = [];
      if (Array.isArray(transcriptionData.transcription)) {
        wordArray = transcriptionData.transcription;
      } else {
        // Suponemos que la transcripción es una cadena con marcadores de tiempo entre paréntesis
        const splitTokens = transcriptionData.transcription.split(' ');
        for (let i = 0; i < splitTokens.length; i++) {
          // Detectar tokens que representan tiempos, por ejemplo "(0.00)"
          if (/^\(\d+(\.\d+)?\)$/.test(splitTokens[i])) {
            const start_time = parseFloat(splitTokens[i].slice(1, -1));
            const word = splitTokens[i + 1] || '';  // La palabra debería seguir al tiempo
            let end_time = start_time + 0.5;  // Valor predeterminado en caso de no encontrar siguiente tiempo
            // Si el token después de la palabra es otro marcador de tiempo, usarlo como end_time
            if (i + 2 < splitTokens.length && /^\(\d+(\.\d+)?\)$/.test(splitTokens[i + 2])) {
              end_time = parseFloat(splitTokens[i + 2].slice(1, -1));
            }
            wordArray.push({ word, start_time, end_time });
            i++;  // Saltar el siguiente token, ya que fue utilizado como palabra
          }
        }
      }
      // Añadir campo para efectos en cada palabra
      const wordsWithEffects = wordArray.map(obj => ({
        ...obj,
        effects: { speed: null, pitch: null, volume: null }
      }));
      setWords(wordsWithEffects);
    }
  }, [transcriptionData]);

  const handleCheckboxChange = (rowIndex, effectType) => {
    let currentValue = words[rowIndex].effects[effectType];
    if (currentValue === null) {
      const input = prompt(`Ingrese valor para ${effectType} en la palabra "${words[rowIndex].word}"`, "0");
      if (input !== null) {
        const num = parseFloat(input);
        if (!isNaN(num)) {
          updateWordEffect(rowIndex, effectType, num);
        }
      }
    } else {
      updateWordEffect(rowIndex, effectType, null);
    }
  };

  const updateWordEffect = (rowIndex, effectType, value) => {
    setWords(prevWords => {
      const newWords = [...prevWords];
      newWords[rowIndex] = {
        ...newWords[rowIndex],
        effects: {
          ...newWords[rowIndex].effects,
          [effectType]: value
        }
      };
      return newWords;
    });
  };

  const handleGlobalCheckboxChange = (effectType) => {
    let currentValue = globalEffects[effectType];
    if (currentValue === null) {
      const input = prompt(`Ingrese valor para ${effectType} para todo el audio`, "0");
      if (input !== null) {
        const num = parseFloat(input);
        if (!isNaN(num)) {
          setGlobalEffects(prev => ({ ...prev, [effectType]: num }));
        }
      }
    } else {
      setGlobalEffects(prev => ({ ...prev, [effectType]: null }));
    }
  };

  const compileModifications = () => {
    let modifications = [];
    if (globalEffects.speed !== null || globalEffects.pitch !== null || globalEffects.volume !== null) {
      modifications.push({
        start_time: 0.0,
        end_time: 9999.0,
        speed_change: globalEffects.speed !== null ? globalEffects.speed : 1.0,
        pitch_shift: globalEffects.pitch !== null ? globalEffects.pitch : 0.0,
        volume_change: globalEffects.volume !== null ? globalEffects.volume : 0.0
      });
    }
    words.forEach(wordObj => {
      const { start_time, end_time, effects } = wordObj;
      const { speed, pitch, volume } = effects;
      if (speed !== null || pitch !== null || volume !== null) {
        modifications.push({
          start_time,
          end_time,
          speed_change: speed !== null ? speed : 1.0,
          pitch_shift: pitch !== null ? pitch : 0.0,
          volume_change: volume !== null ? volume : 0.0
        });
      }
    });
    return modifications;
  };

  const handleGenerarProsodia = async () => {
    if (!generatedAudio) {
      toast.error('No hay audio para generar prosodia');
      return;
    }

    const modifications = compileModifications();
    if (modifications.length === 0) {
      toast.error('No se seleccionaron modificaciones.');
      return;
    }

    setIsProsodyGenerating(true);
    try {
      const response = await axios.post('http://localhost:5000/api/modify_prosody', {
        audio_path: generatedAudio,
        modifications
      });
      if (response.data.output_audio_path) {
        setModifiedAudio(response.data.output_audio_path);
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

  return (
    <div className="p-4 bg-gray-50 rounded-lg">
      <Toaster position="top-right" />
      <h3 className="text-2xl font-bold mb-4">Transcripción Generada</h3>
      
      {/* Sección para modificaciones globales */}
      <div className="mb-4">
        <h4 className="text-xl font-semibold">Modificar audio completo</h4>
        <div className="flex space-x-4">
          <label>
            <input
              type="checkbox"
              checked={globalEffects.speed !== null}
              onChange={() => handleGlobalCheckboxChange('speed')}
            />
            <span className="ml-2">Velocidad</span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={globalEffects.pitch !== null}
              onChange={() => handleGlobalCheckboxChange('pitch')}
            />
            <span className="ml-2">Pitch</span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={globalEffects.volume !== null}
              onChange={() => handleGlobalCheckboxChange('volume')}
            />
            <span className="ml-2">Volumen</span>
          </label>
        </div>
      </div>
      
      {/* Tabla de palabras */}
      <table className="min-w-full divide-y divide-gray-200">
        <thead>
          <tr>
            <th className="px-4 py-2 text-left">Palabra</th>
            <th className="px-4 py-2 text-left">Inicio</th>
            <th className="px-4 py-2 text-left">Fin</th>
            <th className="px-4 py-2 text-center">Velocidad</th>
            <th className="px-4 py-2 text-center">Pitch</th>
            <th className="px-4 py-2 text-center">Volumen</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {words.map((wordObj, index) => (
            <tr key={index}>
              <td className="px-4 py-2">{wordObj.word}</td>
              <td className="px-4 py-2">{wordObj.start_time.toFixed(2)}</td>
              <td className="px-4 py-2">{wordObj.end_time.toFixed(2)}</td>
              <td className="px-4 py-2 text-center">
                <input
                  type="checkbox"
                  checked={wordObj.effects.speed !== null}
                  onChange={() => handleCheckboxChange(index, 'speed')}
                />
              </td>
              <td className="px-4 py-2 text-center">
                <input
                  type="checkbox"
                  checked={wordObj.effects.pitch !== null}
                  onChange={() => handleCheckboxChange(index, 'pitch')}
                />
              </td>
              <td className="px-4 py-2 text-center">
                <input
                  type="checkbox"
                  checked={wordObj.effects.volume !== null}
                  onChange={() => handleCheckboxChange(index, 'volume')}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <button
        onClick={handleGenerarProsodia}
        disabled={isProsodyGenerating}
        className={`mt-4 px-4 py-2 rounded text-white transition ${
          isProsodyGenerating ? 'bg-green-400 cursor-not-allowed' : 'bg-green-600 hover:bg-green-700'
        }`}
      >
        {isProsodyGenerating ? 'Generando Prosodia...' : 'Generar Prosodia'}
      </button>

      {modifiedAudio && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold mb-2">Audio con Prosodia Aplicada:</h4>
          <audio controls className="w-full">
            <source src={`http://localhost:5000/api/get_audio/${encodeURIComponent(modifiedAudio)}`} type="audio/wav" />
            Tu navegador no soporta el elemento de audio.
          </audio>
        </div>
      )}
    </div>
  );
}

export default ProsodyModifier;
