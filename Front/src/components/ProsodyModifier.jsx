import React, { useState, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

function ProsodyModifier({ onAddModification, generatedAudio, transcriptionData, refText }) {
  const [modifications, setModifications] = useState([
    { id: 1, start_time: 0, end_time: 5, pitch_shift: 0, volume_change: 0, speed_change: 1.0 }
  ]);

  const [alignedWords, setAlignedWords] = useState([]);

  // Función de Levenshtein
  function levenshtein(a, b) {
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = a[i - 1] === b[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
      }
    }
    return dp[m][n];
  }

  function normalize(word) {
    return word.toLowerCase().replace(/[^\wñáéíóúü]+/g, '');
  }

  // Alinear palabras originales con reconocidas
  useEffect(() => {
    if (transcriptionData && refText) {
      const originalText = refText;
      const originalWords = originalText.match(/\S+|\n/g) || [];

      const recognizedWords = [];
      if (transcriptionData?.segments && Array.isArray(transcriptionData.segments)) {
        for (const seg of transcriptionData.segments) {
          if (seg.words) {
            for (const w of seg.words) {
              recognizedWords.push(w);
            }
          }
        }
      }

      const aligned = [];
      let previousEnd = 0; // Usar el tiempo final de la palabra anterior
      for (const ow of originalWords) {
        const ow_norm = normalize(ow);
        if (!ow_norm) {
          aligned.push({ word: ow, start: null, end: null });
          continue;
        }

        let best_match = null;
        let best_dist = Infinity;
        for (const rw of recognizedWords) {
          const rw_norm = normalize(rw.word);
          const dist = levenshtein(ow_norm, rw_norm);
          if (dist < best_dist) {
            best_dist = dist;
            best_match = rw;
          }
        }

        if (best_match && best_dist <= Math.max(ow_norm.length / 2, 2)) {
          aligned.push({ word: ow, start: best_match.start, end: best_match.end });
          previousEnd = best_match.end; // Actualizar el tiempo de la palabra final
        } else {
          // Si no se encuentra, usar interpolación entre el tiempo anterior y predecir el siguiente
          const interpolatedStart = previousEnd + 0.1; // Tiempo anterior + 100ms
          const interpolatedEnd = interpolatedStart + 0.3; // Duración por defecto de 300ms
          aligned.push({ word: ow, start: interpolatedStart, end: interpolatedEnd });
          previousEnd = interpolatedEnd;
        }
      }

      setAlignedWords(aligned);
    }
  }, [transcriptionData, refText]);

  const handleChange = (id, field, value) => {
    setModifications(modifications.map(mod => 
      mod.id === id ? { ...mod, [field]: value } : mod
    ));
  };

  const handleAddModification = () => {
    setModifications([
      ...modifications,
      { id: modifications.length + 1, start_time: 0, end_time: 5, pitch_shift: 0, volume_change: 0, speed_change: 1.0 }
    ]);
  };

  const handleRemoveModification = (id) => {
    setModifications(modifications.filter(mod => mod.id !== id));
  };

  const handleSubmit = () => {
    for (let mod of modifications) {
      if (mod.start_time < 0 || mod.end_time < 0) {
        toast.error('Los tiempos de inicio y fin no pueden ser negativos');
        return;
      }
      if (mod.start_time >= mod.end_time) {
        toast.error('El tiempo de inicio debe ser menor que el tiempo de fin');
        return;
      }
      if (mod.speed_change <= 0) {
        toast.error('El factor de cambio de velocidad debe ser mayor que 0');
        return;
      }
    }

    onAddModification(modifications);
  };

  return (
    <div className="mt-8 bg-gray-50 p-6 rounded-lg shadow">
      <h3 className="text-xl font-semibold mb-4">Modificar Prosodia</h3>

      {alignedWords.length > 0 && (
        <div className="mb-8 bg-white p-4 rounded-md shadow">
          <h4 className="text-lg font-medium mb-2">Texto Original con Timestamps</h4>
          <div className="flex flex-wrap mt-2">
            {alignedWords.map((w, idx) => (
              <span key={idx} className="mr-2 mb-2 p-1 bg-gray-200 rounded text-sm">
                ({w.start !== null ? w.start.toFixed(2)+'s' : 'null'}) {w.word}
              </span>
            ))}
          </div>
        </div>
      )}

      {modifications.map(mod => (
        <div key={mod.id} className="mb-4 border-b pb-4">
          <div className="flex justify-between items-center">
            <h4 className="text-lg font-medium">Modificación {mod.id}</h4>
            {modifications.length > 1 && (
              <button onClick={() => handleRemoveModification(mod.id)} className="text-red-500 hover:text-red-700">
                Eliminar
              </button>
            )}
          </div>
        </div>
      ))}

      <button onClick={handleAddModification} className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition">
        Agregar Modificación
      </button>

      <button onClick={handleSubmit} className="mt-6 px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition">
        Aplicar Modificaciones
      </button>
    </div>
  );
}

export default ProsodyModifier;
