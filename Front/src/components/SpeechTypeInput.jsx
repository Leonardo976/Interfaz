import React, { useState, useEffect, useRef } from 'react';

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
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

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

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioFile(new File([audioBlob], 'recorded_audio.wav', { type: 'audio/wav' }));
        setHasUploadedAudio(false);
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error al acceder al micrófono:', error);
      alert('No se pudo acceder al micrófono. Por favor, verifica los permisos.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleSubmit = async () => {
    if (!name.trim()) {
      alert('Por favor, ingrese un nombre para el tipo de habla.');
      return;
    }

    if (!audioFile && !hasUploadedAudio) {
      alert('Por favor, seleccione un archivo de audio o grabe uno.');
      return;
    }

    try {
      setIsUploading(true);
      if (audioFile) {
        await onAudioUpload(audioFile, refText.trim());
        setHasUploadedAudio(true);
        setAudioFile(null);
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
    <div className="bg-white p-6 rounded-xl border-2 border-dashed border-gray-200 hover:border-solid hover:border-sky-300 transition-all">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sección Nombre del Estilo */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <i className="fas fa-tag mr-2 text-sky-600"></i>
            Nombre del Estilo
          </label>
          <div className="flex gap-2">
            <input
              value={name}
              onChange={(e) => onNameChange(e.target.value)}
              disabled={isRegular}
              className={`flex-1 p-2.5 border-2 rounded-lg ${
                isRegular 
                  ? 'bg-gray-100 border-gray-200' 
                  : 'border-sky-200 focus:border-sky-500 focus:ring-2 focus:ring-sky-200'
              }`}
              placeholder="Ej: Entusiasta, Serio..."
            />
            <button
              onClick={() => onInsert(name)}
              disabled={!name.trim()}
              className="px-4 bg-sky-100 hover:bg-sky-200 text-sky-700 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <i className="fas fa-plus mr-2"></i>Insertar
            </button>
            {!isRegular && (
              <button
                onClick={onDelete}
                className="px-4 bg-red-100 hover:bg-red-200 text-red-600 rounded-lg"
              >
                <i className="fas fa-trash"></i>
              </button>
            )}
          </div>
        </div>

        {/* Sección Subida de Audio */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <i className="fas fa-file-upload mr-2 text-sky-600"></i>
            Archivo de Audio
          </label>
          <div className="flex items-center gap-3">
            <label className="flex-1 cursor-pointer">
              <input
                id={`audio-input-${id}`}
                type="file"
                accept="audio/*"
                onChange={handleAudioChange}
                className="hidden"
              />
              <div className="w-full p-2.5 border-2 border-sky-200 rounded-lg hover:bg-sky-50 text-center text-sky-700">
                <i className="fas fa-upload mr-2"></i>
                {audioFile ? audioFile.name : 'Seleccionar archivo'}
              </div>
            </label>
            <button
              onClick={!isRecording ? startRecording : stopRecording}
              className={`p-2.5 rounded-lg ${
                isRecording 
                  ? 'bg-red-500 hover:bg-red-600 text-white' 
                  : 'bg-emerald-500 hover:bg-emerald-600 text-white'
              }`}
            >
              <i className={`fas ${isRecording ? 'fa-stop' : 'fa-microphone'}`}></i>
            </button>
          </div>
          {(audioFile || hasUploadedAudio) && (
            <p className="mt-2 text-sm text-gray-500">
              <i className="fas fa-info-circle mr-2"></i>
              {hasUploadedAudio 
                ? "Audio cargado correctamente" 
                : `Listo para subir: ${audioFile?.name}`}
            </p>
          )}
        </div>

        {/* Sección Texto de Referencia */}
        <div className="lg:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <i className="fas fa-align-left mr-2 text-sky-600"></i>
            Texto de Referencia
          </label>
          <textarea
            value={refText}
            onChange={(e) => setRefText(e.target.value)}
            className="w-full p-2.5 border-2 border-sky-200 rounded-lg focus:border-sky-500 focus:ring-2 focus:ring-sky-200"
            rows="3"
            placeholder="Texto correspondiente al audio (ej: 'Buenos días estudiantes')..."
          />
        </div>

        {/* Botón de Subida */}
        <div className="lg:col-span-2">
          <button
            onClick={handleSubmit}
            disabled={isUploading}
            className={`w-full p-3 rounded-lg font-medium transition-all ${
              isUploading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : hasUploadedAudio
                ? 'bg-green-100 hover:bg-green-200 text-green-700'
                : 'bg-sky-600 hover:bg-sky-700 text-white'
            }`}
          >
            {isUploading ? (
              <>
                <i className="fas fa-spinner fa-spin mr-2"></i> Cargando...
              </>
            ) : hasUploadedAudio ? (
              <>
                <i className="fas fa-check-circle mr-2"></i> Audio Listo
              </>
            ) : (
              <>
                <i className="fas fa-cloud-upload-alt mr-2"></i> Subir Audio
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default SpeechTypeInput;