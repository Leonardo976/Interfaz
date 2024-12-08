import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast, Toaster } from 'react-hot-toast';

function ProsodyModifier({
    generatedAudio,  // Audio file path or blob
    transcriptionData,  // Data from backend transcription
    refText = ''  // Optional reference text with default empty string
}) {
    const [transcriptionDetails, setTranscriptionDetails] = useState({
        timestampedText: '',
        wordsWithTimestamps: [],
        audioDuration: 0,
        originalText: ''
    });
    const [loading, setLoading] = useState(false);

    // Effect to handle transcription data from props
    useEffect(() => {
        if (transcriptionData) {
            setTranscriptionDetails({
                timestampedText: transcriptionData.timestamped_text || '',
                wordsWithTimestamps: transcriptionData.words_with_timestamps || [],
                audioDuration: transcriptionData.audio_duration || 0,
                originalText: transcriptionData.original_text || ''
            });
        }
    }, [transcriptionData]);

    // Effect to automatically analyze audio when generatedAudio changes
    useEffect(() => {
        if (generatedAudio) {
            handleAnalyzeAudio();
        }
    }, [generatedAudio]);

    const handleAnalyzeAudio = async () => {
        if (!generatedAudio) {
            toast.error('No audio file provided');
            return;
        }

        try {
            setLoading(true);
            // Send audio and optional reference text to backend
            const response = await axios.post('/api/analyze_audio', {
                audio_path: generatedAudio,
                ref_text: refText
            });

            // Update state with comprehensive transcription details
            setTranscriptionDetails({
                timestampedText: response.data.timestamped_text,
                wordsWithTimestamps: response.data.words_with_timestamps,
                audioDuration: response.data.audio_duration,
                originalText: response.data.original_text
            });

            toast.success('Audio analyzed successfully');
        } catch (error) {
            console.error('Audio analysis error:', error);
            toast.error('Failed to analyze audio');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="prosody-modifier relative p-4 bg-gray-50 rounded-lg shadow-md">
            <Toaster position="top-right" />

            {/* Audio Analysis Section */}
            <div className="bg-white p-6 rounded-lg shadow-inner">
                <h3 className="text-2xl font-bold mb-4 text-gray-800">Audio Transcription</h3>

                {/* Loading Indicator */}
                {loading && (
                    <div className="absolute top-4 right-4 text-blue-600 font-semibold">
                        Analyzing audio...
                    </div>
                )}

                {/* Comprehensive Transcription Details */}
                <div className="space-y-4">
                    {/* Original Text */}
                    {transcriptionDetails.originalText && (
                        <div>
                            <h4 className="font-semibold text-lg text-gray-700 mb-2">Original Text</h4>
                            <p className="text-gray-600 italic">{transcriptionDetails.originalText}</p>
                        </div>
                    )}

                    {/* Timestamped Text */}
                    {transcriptionDetails.timestampedText && (
                        <div>
                            <h4 className="font-semibold text-lg text-gray-700 mb-2">Timestamped Text</h4>
                            <p className="text-base text-gray-800">{transcriptionDetails.timestampedText}</p>
                        </div>
                    )}

                    {/* Detailed Word Timestamps */}
                    {transcriptionDetails.wordsWithTimestamps.length > 0 && (
                        <div>
                            <h4 className="font-semibold text-lg text-gray-700 mb-2">Detailed Word Timestamps</h4>
                            <div className="overflow-x-auto">
                                <table className="w-full border-collapse">
                                    <thead>
                                        <tr className="bg-gray-200">
                                            <th className="p-2 text-left">Word</th>
                                            <th className="p-2 text-left">Start Time (s)</th>
                                            <th className="p-2 text-left">End Time (s)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {transcriptionDetails.wordsWithTimestamps.map((wordData, index) => (
                                            <tr 
                                                key={index} 
                                                className="border-b hover:bg-gray-100 transition-colors"
                                            >
                                                <td className="p-2">{wordData.word}</td>
                                                <td className="p-2">{wordData.start.toFixed(2)}</td>
                                                <td className="p-2">{wordData.end.toFixed(2)}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* Audio Duration */}
                    {transcriptionDetails.audioDuration > 0 && (
                        <div className="mt-4">
                            <strong className="text-gray-700">Audio Duration:</strong> 
                            <span className="ml-2 text-gray-600">
                                {transcriptionDetails.audioDuration.toFixed(2)} seconds
                            </span>
                        </div>
                    )}
                </div>

                {/* Manual Trigger Button (Optional) */}
                {generatedAudio && !loading && (
                    <button
                        onClick={handleAnalyzeAudio}
                        className="mt-4 w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                    >
                        Reanalyze Audio
                    </button>
                )}
            </div>
        </div>
    );
}

export default ProsodyModifier;