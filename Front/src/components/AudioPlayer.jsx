import React from 'react';

function AudioPlayer({ audioUrl, filename }) {
  return (
    <div className="rounded-lg bg-gray-50 p-4">
      {filename && (
        <p className="text-sm text-gray-600 mb-2">Archivo: {filename}</p>
      )}
      <audio controls data-testid="audio-player" className="w-full">
        <source src={audioUrl} type="audio/wav" />
        <source src={audioUrl} type="audio/mpeg" />
        Tu navegador no soporta el elemento de audio.
      </audio>
    </div>
  );
}

export default AudioPlayer;
