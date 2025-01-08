// src/services/apiService.js

import config from '../config';

export const modifyAudio = async (audio, modifications) => {
  const response = await fetch(`${config.API_URL}/modify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ audio_path: audio, modifications }),
  });

  if (!response.ok) {
    throw new Error('Error al modificar el audio');
  }

  return response.json();
};
