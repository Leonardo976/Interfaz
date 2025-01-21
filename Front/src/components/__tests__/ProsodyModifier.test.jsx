// src/components/__tests__/ProsodyModifier.test.jsx

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';

// Mock consistente con "import toast, { Toaster } from 'react-hot-toast';"
jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    error: jest.fn(),
    success: jest.fn(),
  },
  Toaster: () => <div data-testid="toaster"></div>,
}));

jest.mock('axios');

import ProsodyModifier from '../ProsodyModifier';

describe('ProsodyModifier Component', () => {
  const mockTranscriptionData = {
    // 2 palabras => "Hola" y "mundo"
    transcription: [
      { word: 'Hola', start_time: 0.0, end_time: 0.5 },
      { word: 'mundo', start_time: 0.5, end_time: 1.0 }
    ],
    success: true
  };

  const defaultProps = {
    transcriptionData: mockTranscriptionData,
    generatedAudio: 'path/to/generatedAudio.wav'
  };

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renderiza la transcripción y los checkboxes para cada palabra', () => {
    render(<ProsodyModifier {...defaultProps} />);

    // 3 checkboxes globales (speed, pitch, volume) + 2 palabras * 3 checkboxes c/u => 3 + (2 * 3) = 9
    const checkboxes = screen.getAllByRole('checkbox');
    expect(checkboxes.length).toBe(9);

    // Verificamos que aparezcan "Hola" y "mundo"
    expect(screen.getByText('Hola')).toBeInTheDocument();
    expect(screen.getByText('mundo')).toBeInTheDocument();
  });

  test('marca global de velocidad, pitch y volumen', () => {
    render(<ProsodyModifier {...defaultProps} />);
    // El primer checkbox corresponde a "Velocidad" global (según el orden en el JSX)
    // El segundo "Pitch" global, el tercero "Volumen" global, y así sucesivamente.
    const checkboxes = screen.getAllByRole('checkbox');
    // Chequeamos que al inicio no estén marcados
    expect(checkboxes[0]).not.toBeChecked(); // Global velocidad
    expect(checkboxes[1]).not.toBeChecked(); // Global pitch
    expect(checkboxes[2]).not.toBeChecked(); // Global volumen

    // Mock de prompt para introducir un valor (ej: 1.5)
    window.prompt = jest.fn().mockReturnValue('1.5');
    // Marcamos el primer checkbox (Velocidad global)
    fireEvent.click(checkboxes[0]);

    // Ahora debe estar marcado
    expect(checkboxes[0]).toBeChecked();
    expect(window.prompt).toHaveBeenCalled();
  });

  test('genera prosodia al hacer click con algún checkbox marcado', async () => {
    render(<ProsodyModifier {...defaultProps} />);
    window.prompt = jest.fn().mockReturnValue('1.5');

    // 9 checkboxes totales => [0..8]
    //   0: Velocidad (global)
    //   1: Pitch (global)
    //   2: Volumen (global)
    //   3: Velocidad (palabra 1)
    //   4: Pitch (palabra 1)
    //   5: Volumen (palabra 1)
    //   6: Velocidad (palabra 2)
    //   7: Pitch (palabra 2)
    //   8: Volumen (palabra 2)
    const checkboxes = screen.getAllByRole('checkbox');
    // Activamos la velocidad de la primera palabra: checkboxes[3]
    fireEvent.click(checkboxes[3]);

    // Mock de axios
    axios.post.mockResolvedValue({ data: { output_audio_path: 'path/to/modified.wav' } });

    const generateButton = screen.getByRole('button', { name: /Generar Prosodia/i });
    fireEvent.click(generateButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:5000/api/modify_prosody',
        expect.any(Object)
      );
    });
  });

  test('muestra error si no hay audio para generar prosodia', async () => {
    render(<ProsodyModifier {...defaultProps} generatedAudio={null} />);

    const generateButton = screen.getByRole('button', { name: /Generar Prosodia/i });
    fireEvent.click(generateButton);

    const { default: toastMock } = require('react-hot-toast');
    expect(toastMock.error).toHaveBeenCalledWith('No hay audio para generar prosodia');
  });

  test('muestra error si no hay modificaciones seleccionadas', async () => {
    render(<ProsodyModifier {...defaultProps} />);
    // No marcamos ningún checkbox => compileModifications() estará vacío

    const generateButton = screen.getByRole('button', { name: /Generar Prosodia/i });
    fireEvent.click(generateButton);

    const { default: toastMock } = require('react-hot-toast');
    expect(toastMock.error).toHaveBeenCalledWith('No se seleccionaron modificaciones.');
  });
});
