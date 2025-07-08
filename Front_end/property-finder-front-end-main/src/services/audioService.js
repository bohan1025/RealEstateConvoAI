// services/audioService.js

export async function startRecording(setRecorder, setAudioChunks, onAudioReady) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  
  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
    ? 'audio/webm;codecs=opus' 
    : MediaRecorder.isTypeSupported('audio/webm') 
    ? 'audio/webm' 
    : 'audio/ogg;codecs=opus';
  
  console.log('The audio format used:', mimeType);
  
  const mediaRecorder = new MediaRecorder(stream, { mimeType });

  let chunks = [];

  mediaRecorder.ondataavailable = (e) => {
    chunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    // Ensure the audio data is ready
    console.log('Audio chunks ready, count:', chunks.length);
    setAudioChunks(chunks);
    
    // Process the audio directly
    const audioBlob = new Blob(chunks, { type: 'audio/webm' });
    console.log('Audio ready, size:', audioBlob.size);
    
    // Call the callback function to process the audio
    if (onAudioReady) {
      onAudioReady(audioBlob);
    }
  };

  mediaRecorder.start();
  setRecorder(mediaRecorder);
}

export function stopRecording(recorder) {
  if (recorder && recorder.state === 'recording') {
    recorder.stop();
  }
}

export async function sendAudioToBackend(audioBlob) {
  const formData = new FormData();
  formData.append('file', audioBlob, 'userAudio.webm');

  console.log('Sending the audio file, size:', audioBlob.size, 'bytes');

  const response = await fetch('http://localhost:8000/process-audio', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    console.error("❌ Backend processing failed, status:", response.status);
    const errorText = await response.text();
    console.error("Error details:", errorText);
    throw new Error(`Backend error: ${response.status}`);
  }

  const result = await response.json();
  console.log("✅ Backend response:", result);

  return result; // Return the full backend response
}

// Play audio from a given URL once and resolve only after playback completes
export async function playAudioFromUrl(audioUrl) {
  return new Promise((resolve, reject) => {
    const audio = new Audio(audioUrl);

    audio.onended = () => {
      console.log("✅ Audio playback finished.");
      resolve(); // Only then continue to next recording cycle
    };

    audio.onerror = (e) => {
      console.error("❌ Audio load error:", e);
      reject(new Error("Failed to load audio"));
    };

    // Attempt to play audio
    audio.play().catch((err) => {
      console.error("❌ Playback failed:", err);
      reject(err);
    });
  });
}

// Get and play welcome audio from backend
export async function playWelcomeAudio() {
  try {
    console.log("🎵 Fetching welcome audio...");
    
    const response = await fetch('http://localhost:8000/welcome-audio', {
      method: 'GET',
    });

    if (!response.ok) {
      console.error("❌ Failed to get welcome audio, status:", response.status);
      throw new Error(`Backend error: ${response.status}`);
    }

    const result = await response.json();
    console.log("✅ Welcome audio response:", result);

    if (result.success && result.audio_url) {
      console.log("🎵 Playing welcome audio...");
      await playAudioFromUrl(`http://localhost:8000${result.audio_url}`);
      console.log("✅ Welcome audio playback completed");
    }

    return result;
  } catch (error) {
    console.error("❌ Failed to play welcome audio:", error);
    throw error;
  }
}

