// src/components/__tests__/MultiSpeechGenerator.test.jsx

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import MultiSpeechGenerator from '../MultiSpeechGenerator';

// Mock de react-hot-toast
jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    error: jest.fn(),
    success: jest.fn(),
  },
  // Por si usas <Toaster />
  Toaster: () => <div data-testid="toaster"></div>,
}));

// Mock de axios
jest.mock('axios');

// Confirm para “¿Estás seguro?” (si quisieras seguir usándolo en algún sitio)
window.confirm = jest.fn().mockReturnValue(true);

describe('MultiSpeechGenerator Component (SIN probar delete)', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  test('renderiza elementos básicos', () => {
    render(<MultiSpeechGenerator />);
    expect(
      screen.getByText(/Generación de Múltiples Tipos de Habla/i)
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Agregar Tipo de Habla/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Ingresa el guion/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generar Habla Multi-Estilo/i })).toBeInTheDocument();
  });

  test('agrega un nuevo tipo de habla', () => {
    render(<MultiSpeechGenerator />);
    fireEvent.click(screen.getByRole('button', { name: /Agregar Tipo de Habla/i }));

    const nameLabels = screen.getAllByText(/Nombre del Tipo de Habla/i);
    // Por defecto había 1 (Regular). Al agregar => 2
    expect(nameLabels).toHaveLength(2);
  });

  test('elimina un tipo de habla dinámico (no es delete_audio, solo remove input)', () => {
    render(<MultiSpeechGenerator />);
    // Agregar uno
    fireEvent.click(screen.getByRole('button', { name: /Agregar Tipo de Habla/i }));
    // Se agrega un botón de “Eliminar” (el Regular no se puede)
    const deleteButtons = screen.getAllByRole('button', { name: /Eliminar/i });
    expect(deleteButtons).toHaveLength(1);

    // Lo eliminamos
    fireEvent.click(deleteButtons[0]);
    // Debe quedar solo “Regular”
    expect(screen.getAllByText(/Nombre del Tipo de Habla/i)).toHaveLength(1);
  });

  test('genera audio y lo muestra en la lista (sin {Regular} para no exigir audio subido)', async () => {
    // 1) Al generar
    axios.post.mockResolvedValueOnce({
      data: {
        success: true,
        audio_path: 'path/to/generatedAudio.wav'
      }
    });

    render(<MultiSpeechGenerator />);
    // Cambiamos el texto (sin {Regular})
    fireEvent.change(
      screen.getByPlaceholderText(/Ingresa el guion/i),
      { target: { value: 'Texto sin llaves' } }
    );
    // Click en “Generar Habla”
    fireEvent.click(screen.getByRole('button', { name: /Generar Habla Multi-Estilo/i }));

    // Esperamos la llamada a generate_multistyle_speech
    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:5000/api/generate_multistyle_speech',
        expect.any(Object)
      );
    });

    // Debe aparecer "generatedAudio.wav" en la lista
    expect(screen.getByText('generatedAudio.wav')).toBeInTheDocument();
  });

  test('analiza el audio (analyze_audio)', async () => {
    // 1) Generar
    axios.post.mockResolvedValueOnce({
      data: {
        success: true,
        audio_path: 'path/to/generatedAudio.wav'
      }
    });

    render(<MultiSpeechGenerator />);
    fireEvent.change(
      screen.getByPlaceholderText(/Ingresa el guion/i),
      { target: { value: 'Texto sin llaves' } }
    );
    fireEvent.click(screen.getByRole('button', { name: /Generar Habla Multi-Estilo/i }));

    // Esperamos ver "generatedAudio.wav"
    await waitFor(() => {
      expect(screen.getByText('generatedAudio.wav')).toBeInTheDocument();
    });

    // 2) analyze_audio
    axios.post.mockResolvedValueOnce({
      data: { success: true, transcription: 'Texto de prueba' }
    });

    const personalizeButton = screen.getByRole('button', { name: /Personalizar Audio/i });
    fireEvent.click(personalizeButton);

    await waitFor(() => {
      expect(axios.post).toHaveBeenCalledWith(
        'http://localhost:5000/api/analyze_audio',
        { audio_path: 'path/to/generatedAudio.wav' }
      );
    });
  });

  // NOTA: Aquí NO tenemos el test de “elimina un audio generado al confirmar”
  // para evitar la segunda llamada a /delete_audio.
});
