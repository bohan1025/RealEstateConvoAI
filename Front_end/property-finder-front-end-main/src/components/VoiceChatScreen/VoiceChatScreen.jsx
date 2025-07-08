import React, { useState, useEffect } from 'react';
import './VoiceChatScreen.css';
import {
  startRecording,
  stopRecording,
  sendAudioToBackend,
  playAudioFromUrl,
  playWelcomeAudio,
} from '../../services/audioService';
import { FaMicrophone, FaStop } from 'react-icons/fa';

function VoiceChatScreen({ onEndChat }) {
  const [recorder, setRecorder] = useState(null);
  const [audioChunks, setAudioChunks] = useState([]);
  const [isBotReplying, setIsBotReplying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [propertyPrompt, setPropertyPrompt] = useState("Tap the mic and ask me about a property 🏡");
  const [errorMessage, setErrorMessage] = useState('');
  
  // Record duration control
  const [canStopRecording, setCanStopRecording] = useState(false);
  const MIN_RECORDING_DURATION = 2000; // Minimum record duration 2 seconds

  // Play welcome audio when component mounts
  const [hasPlayedWelcome, setHasPlayedWelcome] = useState(false);

  useEffect(() => {
    if (hasPlayedWelcome) return; // prevent duplicate playback
    
    const playWelcome = async () => {
      try {
        setHasPlayedWelcome(true);
        setIsBotReplying(true);
        setPropertyPrompt("🎵 Welcome! Listen to my introduction...");
        await playWelcomeAudio();
        setIsBotReplying(false);
        setPropertyPrompt("Tap the mic and ask me about a property 🏡");
      } catch (error) {
        setHasPlayedWelcome(false); // reset when failed
        console.error("Failed to play welcome audio:", error);
        setIsBotReplying(false);
        setPropertyPrompt("Tap the mic and ask me about a property 🏡");
      }
    };

    playWelcome();
  }, [hasPlayedWelcome]); // Empty dependency array means this runs once when component mounts

  // Auto-send audio when chunks are ready and recording is stopped
  useEffect(() => {
    if (audioChunks.length > 0 && !isRecording && !isBotReplying) {
      const sendAudio = async () => {
        setIsBotReplying(true);
        setPropertyPrompt("Processing your question... 🤖");
        
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log('audioBlob:', audioBlob, 'size:', audioBlob.size);
        
        try {
          const result = await sendAudioToBackend(audioBlob);
          console.log("🎙️ Transcription:", result.recognized_text);
          console.log("📦 Extracted Info:", result.client_info);
          console.log("🗣️ Bot Reply:", result.property_info);

          if (result.audio_url) {
            await playAudioFromUrl(`http://localhost:8000${result.audio_url}`);
          }

          setIsBotReplying(false);
          setPropertyPrompt("Ask another question or explore more 🏘️");
          setAudioChunks([]); // Clear audio data
        } catch (error) {
          console.error("Failed to send audio:", error);
          setErrorMessage("⚠️ " + error.message);
          setPropertyPrompt("⚠️ Backend unavailable. Try again later.");
          setIsBotReplying(false);
          setAudioChunks([]); // Clear audio data
        }
      };
      
      sendAudio();
    }
  }, [audioChunks, isRecording, isBotReplying]);

  const handleStartRecording = async () => {
    await startRecording(setRecorder, setAudioChunks);
    setIsRecording(true);
    setCanStopRecording(false);
    setErrorMessage('');
    
    // Allow record in 2 seconds
    setTimeout(() => {
      setCanStopRecording(true);
    }, MIN_RECORDING_DURATION);
  };

  const handleStopRecording = () => {
    if (!canStopRecording) {
      setErrorMessage("Speak at least two seconds");
      return;
    }
    
    stopRecording(recorder);
    setIsRecording(false);
    setCanStopRecording(false);
  };

  const handleMicClick = async () => {
    if (!isRecording) {
      handleStartRecording();
    } else {
      handleStopRecording();
      // Audio sending is handled by useEffect when audioChunks are ready
    }
  };

  return (
    <div className={`voice-chat-screen ${isBotReplying ? 'blur-background' : ''}`}>
      <div className="chat-header">
        <h2>��️ Real Estate Voice Assistant</h2>
        <p className="chat-subtitle">Let's find your dream property — just talk to me!</p>
      </div>

      <div className="chat-container">
        <p className="bot-response">
          {isBotReplying ? "🤖 Replying with the best property match..." : `�� ${propertyPrompt}`}
        </p>
      </div>

      <div className="mic-toggle">
        <button
          className={`mic-button ${isRecording ? "recording" : ""} ${isRecording && !canStopRecording ? "recording-waiting" : ""}`}
          onClick={handleMicClick}
        >
          {isRecording
            ? <FaStop size={22} color="white" />
            : <FaMicrophone size={22} color="#007bff" />
          }
        </button>
        <p className="mic-status">
          {isRecording 
            ? (canStopRecording ? "Recording... (Can realease the speak button)" : "Recording... (please contunue talking)")
            : "Tap to speak"
          }
        </p>
        {errorMessage && (
          <p className="error-message">{errorMessage}</p>
        )}
      </div>

      <div className="footer-note">�� Serving properties around your preferred suburb</div>

      <button
        className="end-chat-button"
        onClick={() => {
          handleStopRecording();
          onEndChat();
        }}
      >
        End Chat
      </button>
    </div>
  );
}

export default VoiceChatScreen;