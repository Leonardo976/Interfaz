// src/components/__tests__/SpeechTypeInput.test.jsx

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SpeechTypeInput from '../SpeechTypeInput';

// Mock de window.alert para evitar que rompa
global.alert = jest.fn();

describe('SpeechTypeInput Component', () => {
  const defaultProps = {
    id: 'test-id',
    name: 'Regular',
    isRegular: false,
    onNameChange: jest.fn(),
    onDelete: jest.fn(),
    onAudioUpload: jest.fn(),
    onInsert: jest.fn(),
    uploadedAudio: null,
    uploadedRefText: ''
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renderiza correctamente los campos iniciales', () => {
    render(<SpeechTypeInput {...defaultProps} />);
    
    // "Nombre del Tipo de Habla"
    const nameInput = screen.getByLabelText(/Nombre del Tipo de Habla/i);
    expect(nameInput).toBeInTheDocument();

    // "Audio de Referencia"
    const fileInput = screen.getByLabelText(/Audio de Referencia/i);
    expect(fileInput).toBeInTheDocument();

    // Botón Insertar
    expect(screen.getByRole('button', { name: /insertar/i })).toBeInTheDocument();

    // Botón Eliminar (si !isRegular)
    expect(screen.getByRole('button', { name: /eliminar/i })).toBeInTheDocument();
  });

  test('deshabilita el campo de nombre si isRegular es true', () => {
    render(<SpeechTypeInput {...defaultProps} isRegular />);
    const nameInput = screen.getByLabelText(/Nombre del Tipo de Habla/i);
    expect(nameInput).toBeDisabled();
  });

  test('llama a onNameChange al cambiar texto del nombre', () => {
    render(<SpeechTypeInput {...defaultProps} />);
    const nameInput = screen.getByLabelText(/Nombre del Tipo de Habla/i);

    fireEvent.change(nameInput, { target: { value: 'Triste' } });
    expect(defaultProps.onNameChange).toHaveBeenCalledWith('Triste');
  });

  test('llama a onInsert al presionar "Insertar"', () => {
    render(<SpeechTypeInput {...defaultProps} />);
    const insertButton = screen.getByRole('button', { name: /insertar/i });
    fireEvent.click(insertButton);

    expect(defaultProps.onInsert).toHaveBeenCalledWith('Regular');
  });

  test('llama a onDelete al presionar "Eliminar"', () => {
    render(<SpeechTypeInput {...defaultProps} />);
    const deleteButton = screen.getByRole('button', { name: /eliminar/i });
    fireEvent.click(deleteButton);

    expect(defaultProps.onDelete).toHaveBeenCalledTimes(1);
  });

  test('muestra alerta si se sube un archivo no-audio', async () => {
    render(<SpeechTypeInput {...defaultProps} />);
    const fileInput = screen.getByLabelText(/Audio de Referencia/i);

    const fakeFile = new File(['(⌐□_□)'], 'fake.txt', { type: 'text/plain' });
    fireEvent.change(fileInput, { target: { files: [fakeFile] } });

    expect(global.alert).toHaveBeenCalledWith('Por favor, seleccione un archivo de audio válido.');
    expect(fileInput.value).toBe('');
  });

  test('asigna archivo de audio válido sin alert', async () => {
    render(<SpeechTypeInput {...defaultProps} />);
    const fileInput = screen.getByLabelText(/Audio de Referencia/i);

    const audioFile = new File(['test'], 'audioTest.wav', { type: 'audio/wav' });
    fireEvent.change(fileInput, { target: { files: [audioFile] } });

    expect(global.alert).not.toHaveBeenCalled();
  });

  test('muestra alerta si nombre está vacío al presionar "Cargar Audio"', async () => {
    const props = { ...defaultProps, name: '' };
    render(<SpeechTypeInput {...props} />);
    const button = screen.getByRole('button', { name: /cargar audio/i });
    fireEvent.click(button);

    expect(global.alert).toHaveBeenCalledWith('Por favor, ingrese un nombre para el tipo de habla.');
  });

  test('muestra alerta si no hay archivo y no hay audio subido antes de presionar "Cargar Audio"', async () => {
    render(<SpeechTypeInput {...defaultProps} name="Feliz" />);
    const button = screen.getByRole('button', { name: /cargar audio/i });
    fireEvent.click(button);

    expect(global.alert).toHaveBeenCalledWith('Por favor, seleccione un archivo de audio o grabe uno.');
  });

  test('llama a onAudioUpload cuando hay un archivo y se presiona "Cargar Audio"', async () => {
    const props = { ...defaultProps, name: 'Feliz' };
    render(<SpeechTypeInput {...props} />);

    const fileInput = screen.getByLabelText(/Audio de Referencia/i);
    const audioFile = new File(['audio data'], 'audioTest.wav', { type: 'audio/wav' });
    fireEvent.change(fileInput, { target: { files: [audioFile] } });

    const button = screen.getByRole('button', { name: /cargar audio/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(props.onAudioUpload).toHaveBeenCalledWith(expect.any(File), '');
    });
  });

  test('comienza y detiene la grabación (mock de getUserMedia)', async () => {
    const mockStream = {};
    const mockMediaRecorderInstance = {
      start: jest.fn(),
      stop: jest.fn(),
      ondataavailable: null,
      onstop: null,
    };
    global.navigator.mediaDevices = {
      getUserMedia: jest.fn().mockResolvedValue(mockStream),
    };
    window.MediaRecorder = jest.fn().mockImplementation(() => mockMediaRecorderInstance);

    render(<SpeechTypeInput {...defaultProps} />);

    const recordButton = screen.getByRole('button', { name: /grabar audio/i });
    fireEvent.click(recordButton);

    await waitFor(() => {
      expect(global.navigator.mediaDevices.getUserMedia).toHaveBeenCalledWith({ audio: true });
      expect(mockMediaRecorderInstance.start).toHaveBeenCalled();
    });

    const stopButton = screen.getByRole('button', { name: /detener grabación/i });
    fireEvent.click(stopButton);

    expect(mockMediaRecorderInstance.stop).toHaveBeenCalled();
  });
});
