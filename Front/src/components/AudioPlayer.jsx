// src/components/AudioPlayer.jsx
import React from 'react';

function AudioPlayer({ audioUrl, filename }) {
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = audioUrl;
    link.download = filename || 'audio_generado.wav';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="bg-white p-4 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {/* Icono de auriculares (SVG) */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-blue-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"
            />
          </svg>
          {filename && (
            <span className="text-sm font-medium text-gray-700 truncate max-w-[200px]">
              {filename}
            </span>
          )}
        </div>

        {/* Botón de descarga con icono SVG */}
        <button
          onClick={handleDownload}
          className="p-2 rounded-full hover:bg-blue-50 transition-colors text-blue-600 hover:text-blue-700"
          title="Descargar audio"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
        </button>
      </div>

      <div className="relative">
        <audio
          controls
          className="w-full rounded-lg [&::-webkit-media-controls-panel]:bg-gray-50"
        >
          <source src={audioUrl} type="audio/wav" />
          <source src={audioUrl} type="audio/mpeg" />
          Tu navegador no soporta el elemento de audio.
        </audio>

        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
          {!audioUrl && (
            <div className="flex flex-col items-center text-gray-400">
              {/* Icono de música (SVG) */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-8 w-8"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
                />
              </svg>
              <span className="text-sm">Previsualización no disponible</span>
            </div>
          )}
        </div>
      </div>

      {audioUrl && (
        <div className="mt-3 text-xs text-gray-500 flex items-center space-x-2">
          {/* Icono de música pequeño (SVG) */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"
            />
          </svg>
          <span>Formato: WAV • Calidad: 44.1kHz • 16-bit</span>
        </div>
      )}
    </div>
  );
}

export default AudioPlayer;