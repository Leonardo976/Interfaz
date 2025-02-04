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
        const splitTokens = transcriptionData.transcription.split(' ');
        for (let i = 0; i < splitTokens.length; i++) {
          if (/^\(\d+(\.\d+)?\)$/.test(splitTokens[i])) {
            const start_time = parseFloat(splitTokens[i].slice(1, -1));
            const word = splitTokens[i + 1] || '';
            let end_time = start_time + 0.5;
            if (i + 2 < splitTokens.length && /^\(\d+(\.\d+)?\)$/.test(splitTokens[i + 2])) {
              end_time = parseFloat(splitTokens[i + 2].slice(1, -1));
            }
            wordArray.push({ word, start_time, end_time });
            i++;
          }
        }
      }
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
    <div className="p-6 bg-white rounded-xl shadow-lg">
      <Toaster position="top-right" />
      <div className="mb-8">
        <h3 className="text-2xl font-bold text-gray-800 mb-2 flex items-center">
          {/* Icono Edit (FiEdit) */}
          <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-blue-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
          </svg>
          Editor de Prosodia
        </h3>
        <p className="text-gray-600">Modifica los parámetros de velocidad, tono y volumen por palabra o para todo el audio.</p>
      </div>

      <div className="mb-8 bg-blue-50 p-4 rounded-lg">
        <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          {/* Icono Activity (FiActivity) */}
          <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-blue-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
          </svg>
          Ajustes Globales
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {['speed', 'pitch', 'volume'].map((effectType) => (
            <button
              key={effectType}
              onClick={() => handleGlobalCheckboxChange(effectType)}
              className={`p-3 rounded-lg flex items-center justify-center transition-all ${
                globalEffects[effectType] !== null
                  ? 'bg-blue-100 border-2 border-blue-300'
                  : 'bg-white border-2 border-gray-200 hover:border-blue-200'
              }`}
            >
              {effectType === 'speed' && (
                // Icono Clock (FiClock)
                <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-blue-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
              )}
              {effectType === 'pitch' && (
                // Icono Volume2 (FiVolume2)
                <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-blue-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
                </svg>
              )}
              {effectType === 'volume' && (
                // Icono Activity (FiActivity)
                <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-blue-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
              )}
              <span className="capitalize">
                {effectType === 'speed' ? 'Velocidad' : 
                 effectType === 'pitch' ? 'Tono' : 'Volumen'}
              </span>
              {globalEffects[effectType] !== null && (
                <span className="ml-2 text-blue-600">({globalEffects[effectType]})</span>
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-8">
        <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
          {/* Icono CheckCircle (FiCheckCircle) */}
          <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-blue-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 12l2 2l4-4"></path>
            <circle cx="12" cy="12" r="10"></circle>
          </svg>
          Ajustes por Palabra
        </h4>
        <div className="overflow-x-auto rounded-lg border border-gray-200">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Palabra</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Inicio (s)</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">Fin (s)</th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">
                  {/* Icono Clock (FiClock) */}
                  <svg xmlns="http://www.w3.org/2000/svg" className="inline mr-1" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                  </svg>
                  Velocidad
                </th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">
                  {/* Icono Volume2 (FiVolume2) */}
                  <svg xmlns="http://www.w3.org/2000/svg" className="inline mr-1" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                    <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                    <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
                  </svg>
                  Tono
                </th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">
                  {/* Icono Activity (FiActivity) */}
                  <svg xmlns="http://www.w3.org/2000/svg" className="inline mr-1" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                  </svg>
                  Volumen
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {words.map((wordObj, index) => (
                <tr key={index} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3 text-sm font-medium text-gray-900">{wordObj.word}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{wordObj.start_time.toFixed(2)}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{wordObj.end_time.toFixed(2)}</td>
                  {['speed', 'pitch', 'volume'].map((effectType) => (
                    <td key={effectType} className="px-4 py-3 text-center">
                      <button
                        onClick={() => handleCheckboxChange(index, effectType)}
                        className={`p-2 rounded-full transition-all ${
                          wordObj.effects[effectType] !== null
                            ? 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                            : 'bg-gray-100 text-gray-400 hover:bg-gray-200'
                        }`}
                      >
                        {wordObj.effects[effectType] !== null ? wordObj.effects[effectType] : '+'}
                      </button>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="flex justify-center">
        <button
          onClick={handleGenerarProsodia}
          disabled={isProsodyGenerating}
          className={`px-6 py-3 rounded-xl font-medium flex items-center transition-all ${
            isProsodyGenerating
              ? 'bg-blue-300 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          {isProsodyGenerating ? (
            <>
              <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Procesando...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="mr-2" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 12l2 2l4-4"></path>
                <circle cx="12" cy="12" r="10"></circle>
              </svg>
              Generar Audio Modificado
            </>
          )}
        </button>
      </div>

      {modifiedAudio && (
        <div className="mt-8 bg-green-50 p-4 rounded-lg">
          <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="mr-2 text-green-600" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
              <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
            </svg>
            Resultado Final
          </h4>
          <div className="bg-white p-4 rounded-lg shadow">
            <audio controls className="w-full">
              <source src={`http://localhost:5000/api/get_audio/${encodeURIComponent(modifiedAudio)}`} type="audio/wav" />
              Tu navegador no soporta el elemento de audio.
            </audio>
          </div>
        </div>
      )}
    </div>
  );
}

export default ProsodyModifier;
