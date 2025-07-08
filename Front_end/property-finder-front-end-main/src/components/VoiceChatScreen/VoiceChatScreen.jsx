import React, { useState } from 'react';
import './VoiceChatScreen.css';
import {
  startRecording,
  stopRecording,
  sendAudioToBackend,
  playAudioFromUrl,
} from '../../services/audioService';
import { FaMicrophone, FaStop } from 'react-icons/fa';

function VoiceChatScreen({ onEndChat }) {
  const [recorder, setRecorder] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isBotReplying, setIsBotReplying] = useState(false);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [propertyPrompt, setPropertyPrompt] = useState("Tap the mic and ask me about a property 🏡");
  const [errorMessage, setErrorMessage] = useState('');

  const handleStartRecording = async () => {
    await startRecording(setRecorder, () => {}, handleAudioReady);
    setIsRecording(true);
    setErrorMessage('');
  };

  const handleStopRecording = () => {
    stopRecording(recorder);
    setIsRecording(false);
  };

  const handleMicClick = async () => {
    if (!isRecording) {
      await handleStartRecording();
    } else {
      handleStopRecording();
      setPropertyPrompt("Processing your question... 🤖");
      setIsBotReplying(true);
    }
  };

  const handleAudioReady = async (audioBlob) => {
    if (!audioBlob || audioBlob.size === 0) {
      setErrorMessage("⚠️ Audio recording failed or was too short.");
      setPropertyPrompt("Please try again.");
      setIsBotReplying(false);
      return;
    }

    try {
      const response = await sendAudioToBackend(audioBlob);

      if (response.audio_url) {
        setIsBotSpeaking(true);
        await playAudioFromUrl(`http://localhost:8000${response.audio_url}`);
        setIsBotSpeaking(false);
      }

      setPropertyPrompt("Ask another question or explore more 🏘️");
    } catch (err) {
      setErrorMessage("⚠️ " + err.message);
      setPropertyPrompt("⚠️ Backend unavailable. Try again later.");
    }

    setIsBotReplying(false);
  };

  return (
    <div className={`voice-chat-screen ${isBotReplying ? 'blur-background' : ''}`}>
      <div className="chat-header">
        <h2>🏘️ Real Estate Voice Assistant</h2>
        <p className="chat-subtitle">Let’s find your dream property — just talk to me!</p>
      </div>

      <div className="chat-container">
        <p className="bot-response">
          {isBotReplying ? "🤖 Replying with the best property match..." : `🎧 ${propertyPrompt}`}
        </p>

        {isRecording && (
          <div className="listening-animation">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        )}

        {isBotSpeaking && (
          <div className="speaking-animation">
            <span className="bar"></span>
            <span className="bar"></span>
            <span className="bar"></span>
            <span className="bar"></span>
          </div>
        )}
      </div>

      <div className="mic-toggle">
        <button
          className={`mic-button ${isRecording ? "recording" : ""}`}
          onClick={handleMicClick}
        >
          {isRecording
            ? <FaStop size={22} color="white" />
            : <FaMicrophone size={22} color="#007bff" />
          }
        </button>
        <p className="mic-status">{isRecording ? "Recording..." : "Tap to speak"}</p>
        {errorMessage && <p className="error-message">{errorMessage}</p>}
      </div>

      <div className="footer-note">📍 Serving properties around your preferred suburb</div>

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
