// src/components/Spectrogram.jsx
import React from 'react';

function Spectrogram({ spectrogramUrl, altText }) {
  return (
    <div className="rounded-lg bg-gray-50 p-4">
      <img
        src={spectrogramUrl}
        alt={altText || "Espectrograma del audio"}
        className="w-full h-auto"
      />
    </div>
  );
}

export default Spectrogram;
