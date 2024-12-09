import React, { useState, useEffect } from 'react';
import { Toaster } from 'react-hot-toast';

function ProsodyModifier({ transcriptionData }) {
    const [transcription, setTranscription] = useState('');

    useEffect(() => {
        if (transcriptionData && transcriptionData.transcription) {
            // Asignamos directamente la transcripción que viene del backend
            setTranscription(transcriptionData.transcription);
        }
    }, [transcriptionData]);

    return (
        <div className="p-4 bg-gray-50 rounded-lg">
            <Toaster position="top-right" />
            <h3 className="text-2xl font-bold mb-4">Transcripción Generada</h3>
            
            {/* Como se asume que siempre habrá transcripción, la mostramos directamente */}
            <div className="bg-gray-50 p-4 rounded border border-gray-300">
                <pre className="whitespace-pre-wrap leading-relaxed text-gray-800">
                    {transcription}
                </pre>
            </div>
        </div>
    );
}

export default ProsodyModifier;
