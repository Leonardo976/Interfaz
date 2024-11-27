// src/components/AudioPlayer.jsx
import React from 'react';

function AudioPlayer({ audioUrl }) {
  return (
    <div className="rounded-lg bg-gray-50 p-4">
      <audio controls className="w-full">
        <source src={audioUrl} type="audio/wav" />
        Tu navegador no soporta el elemento de audio.
      </audio>
    </div>
  );
}

export default AudioPlayer;
