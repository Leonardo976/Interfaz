// src/components/__tests__/AudioPlayer.test.jsx

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import AudioPlayer from '../AudioPlayer';

describe('AudioPlayer Component', () => {
  test('renderiza audio con la url y filename', () => {
    const url = 'http://localhost:5000/api/get_audio/test.wav';
    render(<AudioPlayer audioUrl={url} filename="test.wav" />);

    // Muestra "Archivo: test.wav"
    expect(screen.getByText(/Archivo: test.wav/i)).toBeInTheDocument();

    // Usa data-testid="audio-player" en el <audio>
    const audioElement = screen.getByTestId('audio-player');
    expect(audioElement).toBeInTheDocument();

    const sources = audioElement.querySelectorAll('source');
    expect(sources).toHaveLength(2);
    expect(sources[0]).toHaveAttribute('src', url);
    expect(sources[0]).toHaveAttribute('type', 'audio/wav');
    expect(sources[1]).toHaveAttribute('src', url);
    expect(sources[1]).toHaveAttribute('type', 'audio/mpeg');
  });

  test('no muestra el texto de archivo si no se pasa filename', () => {
    const url = 'http://localhost:5000/api/get_audio/test.wav';
    render(<AudioPlayer audioUrl={url} />);

    // No debe salir "Archivo:"
    expect(screen.queryByText(/Archivo:/i)).not.toBeInTheDocument();
  });
});
