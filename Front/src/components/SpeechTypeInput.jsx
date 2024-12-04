// src/components/SpeechTypeInput.jsx
import React, { useState, useEffect } from 'react';

function SpeechTypeInput({
  id,
  name,
  isRegular,
  onNameChange,
  onDelete,
  onAudioUpload,
  onInsert,
  uploadedAudio,
  uploadedRefText
}) {
  const [refText, setRefText] = useState(uploadedRefText || '');
  const [audioFile, setAudioFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [hasUploadedAudio, setHasUploadedAudio] = useState(!!uploadedAudio);

  useEffect(() => {
    setRefText(uploadedRefText || '');
    setHasUploadedAudio(!!uploadedAudio);
  }, [uploadedAudio, uploadedRefText]);

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('audio/')) {
        alert('Por favor, seleccione un archivo de audio válido.');
        e.target.value = '';
        return;
      }
      setAudioFile(file);
      setHasUploadedAudio(false);
    }
  };

  const handleSubmit = async () => {
    if (!name.trim()) {
      alert('Por favor, ingrese un nombre para el tipo de habla.');
      return;
    }

    if (!audioFile && !hasUploadedAudio) {
      alert('Por favor, seleccione un archivo de audio.');
      return;
    }

    try {
      setIsUploading(true);
      if (audioFile) {
        // Enviamos el refText incluso si está vacío
        await onAudioUpload(audioFile, refText.trim());
        setHasUploadedAudio(true);
        setAudioFile(null);
        // Resetear el input de archivo
        const fileInput = document.querySelector(`#audio-input-${id}`);
        if (fileInput) fileInput.value = '';
      }
    } catch (error) {
      console.error('Error al cargar el audio:', error);
      alert('Error al cargar el audio. Por favor, intente nuevamente.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="p-4 border rounded-lg bg-gray-50">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Nombre del Tipo de Habla {isRegular && <span className="text-blue-600">(Principal)</span>}
          </label>
          <div className="flex space-x-2">
            <input
              type="text"
              value={name}
              onChange={(e) => onNameChange(e.target.value)}
              disabled={isRegular}
              className={`flex-1 p-2 border rounded ${
                isRegular ? 'bg-gray-100' : 'hover:border-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500'
              }`}
              placeholder="Ejemplo: Regular, Feliz, Triste..."
            />
            <button
              onClick={() => onInsert(name)}
              disabled={!name.trim()}
              className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Insertar
            </button>
            {!isRegular && (
              <button
                onClick={onDelete}
                className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
              >
                Eliminar
              </button>
            )}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Audio de Referencia {hasUploadedAudio && <span className="text-green-600">(Cargado)</span>}
          </label>
          <input
            id={`audio-input-${id}`}
            type="file"
            accept="audio/*"
            onChange={handleAudioChange}
            className="block w-full text-sm text-gray-500 
              file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 
              file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 
              hover:file:bg-blue-100 transition-all cursor-pointer"
          />
          {(audioFile || hasUploadedAudio) && (
            <p className="mt-1 text-sm text-gray-500">
              {hasUploadedAudio ? 
                "✓ Audio cargado correctamente" : 
                `Archivo seleccionado: ${audioFile.name}`}
            </p>
          )}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Texto de Referencia (Opcional)
          </label>
          <textarea
            value={refText}
            onChange={(e) => setRefText(e.target.value)}
            className="w-full p-2 border rounded hover:border-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            rows="2"
            placeholder="Opcional: Ingrese el texto que corresponde al audio..."
          />
        </div>

        <div className="flex items-end">
          <button
            onClick={handleSubmit}
            disabled={isUploading}
            className={`w-full px-4 py-2 rounded transition-colors ${
              isUploading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : hasUploadedAudio
                ? 'bg-green-600 text-white hover:bg-green-700'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            } flex items-center justify-center`}
          >
            {isUploading ? (
              <>
                <span className="animate-spin mr-2">⭮</span>
                Cargando...
              </>
            ) : hasUploadedAudio ? (
              <>
                <span className="mr-2">✓</span>
                Audio Cargado
              </>
            ) : (
              'Cargar Audio'
            )}
          </button>
        </div>
      </div>
    </div>
  );

}

export default SpeechTypeInput;
