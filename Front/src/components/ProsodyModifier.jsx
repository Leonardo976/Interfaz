// src/components/ProsodyModifier.jsx
import React, { useState } from 'react';

const ProsodyModifier = ({ onAddModification }) => {
  const [modifications, setModifications] = useState([]);

  const addModification = () => {
    setModifications([
      ...modifications,
      {
        id: Date.now(),
        start_time: 0,
        end_time: 5,
        pitch_shift: 0,
        volume_change: 0,
        speed_change: 1.0,
      },
    ]);
  };

  const updateModification = (id, field, value) => {
    setModifications(
      modifications.map(mod =>
        mod.id === id ? { ...mod, [field]: value } : mod
      )
    );
  };

  const removeModification = id => {
    setModifications(modifications.filter(mod => mod.id !== id));
  };

  const handleSubmit = () => {
    onAddModification(modifications);
  };

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-4">Modificar Prosodia</h3>
      {modifications.map(mod => (
        <div key={mod.id} className="border p-4 rounded mb-4">
          <div className="flex justify-between items-center mb-2">
            <span>Segmento</span>
            <button
              onClick={() => removeModification(mod.id)}
              className="text-red-500 hover:text-red-700"
            >
              Eliminar
            </button>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Tiempo de Inicio (s)
              </label>
              <input
                type="number"
                step="0.1"
                value={mod.start_time}
                onChange={e =>
                  updateModification(mod.id, 'start_time', parseFloat(e.target.value))
                }
                className="mt-1 block w-full border rounded-md p-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Tiempo de Fin (s)
              </label>
              <input
                type="number"
                step="0.1"
                value={mod.end_time}
                onChange={e =>
                  updateModification(mod.id, 'end_time', parseFloat(e.target.value))
                }
                className="mt-1 block w-full border rounded-md p-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Cambio de Tono (semitonos)
              </label>
              <input
                type="number"
                step="1"
                value={mod.pitch_shift}
                onChange={e =>
                  updateModification(mod.id, 'pitch_shift', parseFloat(e.target.value))
                }
                className="mt-1 block w-full border rounded-md p-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Cambio de Volumen (dB)
              </label>
              <input
                type="number"
                step="0.1"
                value={mod.volume_change}
                onChange={e =>
                  updateModification(mod.id, 'volume_change', parseFloat(e.target.value))
                }
                className="mt-1 block w-full border rounded-md p-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Cambio de Velocidad (Factor)
              </label>
              <input
                type="number"
                step="0.1"
                value={mod.speed_change}
                onChange={e =>
                  updateModification(mod.id, 'speed_change', parseFloat(e.target.value))
                }
                className="mt-1 block w-full border rounded-md p-2"
              />
            </div>
          </div>
        </div>
      ))}
      <button
        onClick={addModification}
        className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition"
      >
        Añadir Modificación de Prosodia
      </button>
      {modifications.length > 0 && (
        <button
          onClick={handleSubmit}
          className="ml-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Aplicar Modificaciones
        </button>
      )}
    </div>
  );
};

export default ProsodyModifier;
