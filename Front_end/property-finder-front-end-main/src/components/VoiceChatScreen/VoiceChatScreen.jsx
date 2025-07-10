import React, { useState, useEffect, useRef } from 'react';
import './VoiceChatScreen.css';

import {
  startRecording,
  stopRecording,
  sendAudioToBackend,
  playAudioFromUrl,
  playWelcomeAudio,
} from '../../services/audioService';
import { FaMicrophone, FaStop, FaUser, FaRobot } from 'react-icons/fa';

function VoiceChatScreen({ onEndChat }) {
  // Basic recording states
  const [recorder, setRecorder] = useState(null);
  const welcomePlayedRef = useRef(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isBotReplying, setIsBotReplying] = useState(false);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  
  // Multi-turn conversation states
  const [conversationHistory, setConversationHistory] = useState([]);
  const [currentStage, setCurrentStage] = useState('greeting');
  const [clientProfile, setClientProfile] = useState({});
  const [hasPlayedWelcome, setHasPlayedWelcome] = useState(false);
  const [isConversationComplete, setIsConversationComplete] = useState(false);

  // Stage display texts for user guidance
  const stageTexts = {
    greeting: "👋 Welcome! Let me understand your needs",
    basic_info: "📝 Please provide your basic information",
    preferences: "🏠 Tell me about your property preferences",
    property_search: "🔍 Searching for suitable properties",
    details: "🏡 Here are the property details",
    booking: "📅 Would you like to schedule a viewing?",
    complete: "✅ Conversation completed! Thank you!"
  };

  // Play welcome audio when component mounts
  useEffect(() => {
  const playWelcome = async () => {
    if (welcomePlayedRef.current) return;

    try {
      welcomePlayedRef.current = true;
      setIsBotReplying(true);
      setIsBotSpeaking(true);
      await playWelcomeAudio(); // plays once
      setIsBotSpeaking(false);
      setIsBotReplying(false);
    } catch (error) {
      console.error("Failed to play welcome audio:", error);
      setIsBotReplying(false);
      setIsBotSpeaking(false);
      welcomePlayedRef.current = false; // reset on failure
    }
  };

  playWelcome();
}, []);

  const handleStartRecording = async () => {
    if (isConversationComplete) return; // Prevent recording if conversation is complete
    
    await startRecording(setRecorder, () => {}, handleAudioReady);
    setIsRecording(true);
    setErrorMessage('');
  };

  const handleStopRecording = () => {
    stopRecording(recorder);
    setIsRecording(false);
  };

  const handleMicClick = async () => {
    if (isConversationComplete) return; // Prevent new recording if complete
    
    if (!isRecording) {
      await handleStartRecording();
    } else {
      handleStopRecording();
      setIsBotReplying(true);
    }
  };

  const handleAudioReady = async (audioBlob) => {
    if (!audioBlob || audioBlob.size === 0) {
      setErrorMessage("⚠️ Recording failed or too short");
      setIsBotReplying(false);
      return;
    }

    try {
      const response = await sendAudioToBackend(audioBlob);
      console.log("Backend response:", response);

      // Update conversation history with new messages
      setConversationHistory(prev => [
        ...prev,
        { 
          role: 'user', 
          content: response.recognized_text || "Audio message",
          timestamp: new Date().toLocaleTimeString()
        },
        { 
          role: 'assistant', 
          content: response.ai_response || response.property_info || "Processing...",
          timestamp: new Date().toLocaleTimeString()
        }
      ]);

      // Update conversation stage and client profile from backend
      if (response.conversation_stage) {
        setCurrentStage(response.conversation_stage);
      }
      
      if (response.client_profile) {
        setClientProfile(response.client_profile);
      }

      // Check if conversation is complete
      if (response.conversation_complete) {
        console.log("🎉 Conversation completed!");
        
        setIsConversationComplete(true);
        setCurrentStage("complete");
        
        // Add completion message to conversation history
        setConversationHistory(prev => [
          ...prev,
          { 
            role: 'system', 
            content: "✅ Conversation completed! Thank you for using our service!",
            timestamp: new Date().toLocaleTimeString()
          }
        ]);
        
        // Play the final response first
        if (response.audio_url) {
          setIsBotSpeaking(true);
          await playAudioFromUrl(`http://localhost:8000${response.audio_url}`);
          setIsBotSpeaking(false);
        }
        
        // Show completion status
        setErrorMessage("✅ Conversation completed successfully! Redirecting in 5 seconds...");
        
        // Auto-end chat after 5 seconds
        setTimeout(() => {
          console.log("Auto-ending chat...");
          onEndChat();
        }, 5000);
        
        setIsBotReplying(false);
        return; // Exit early, don't continue with normal flow
      }

      // Normal flow: Play AI response audio
      if (response.audio_url) {
        setIsBotSpeaking(true);
        await playAudioFromUrl(`http://localhost:8000${response.audio_url}`);
        setIsBotSpeaking(false);
      }

    } catch (err) {
      setErrorMessage("⚠️ " + err.message);
      console.error('Conversation processing failed:', err);
    }

    setIsBotReplying(false);
  };

  return (
    <div className={"voice-chat-screen"}>
      {/* Header with stage indicator */}
      <div className="chat-header">
        <h2>🏘️ Real Estate Voice Assistant</h2>
        <div className="conversation-stage">
          <span className="stage-indicator">
            {stageTexts[currentStage] || "💬 Conversation in progress..."}
          </span>
        </div>
      </div>

      {/* Conversation history display */}
      {conversationHistory.length > 0 && (
        <div className="conversation-history">
          <div className="history-container">
            {conversationHistory.slice(-4).map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                {msg.role !== 'system' && (
                  <div className="message-icon">
                    {msg.role === 'user' ? <FaUser /> : <FaRobot />}
                  </div>
                )}
                <div className="message-content">
                  <div className="message-text">{msg.content}</div>
                  <div className="message-time">{msg.timestamp}</div>
                </div>
              </div>
            ))}

          </div>
        </div>
      )}

      {/* Client information display */}
      {Object.keys(clientProfile).length > 0 && (
        <div className="client-info">
          <h4>📋 Collected Information:</h4>
          <div className="info-grid">
            {clientProfile.name && <span>👤 {clientProfile.name}</span>}
            {clientProfile.phone && <span>📞 {clientProfile.phone}</span>}
            {clientProfile.budget_range && <span>💰 {clientProfile.budget_range}</span>}
            {clientProfile.property_type && <span>🏠 {clientProfile.property_type}</span>}
            {clientProfile.location_preferences && <span>📍 {clientProfile.location_preferences}</span>}
            {clientProfile.search_intent && <span>🎯 {clientProfile.search_intent}</span>}
            {clientProfile.viewing_time && <span>📅 {clientProfile.viewing_time}</span>}
          </div>
        </div>
      )}

      {/* Main chat container */}
      <div className="chat-container">
        <p className="bot-response">
          {isBotSpeaking
            ? "🔊 Playing response..."
            : isBotReplying
            ? "🤖 Thinking..."
            : currentStage === "complete"
            ? "✅ Conversation completed! Ending chat..."
            : "🎤 Click microphone to start conversation"}
        </p>

        {/* Recording animation */}
        {isRecording && (
          <div className="listening-animation">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
          </div>
        )}

        {/* Unified circular animation with dynamic color */}
        {(isBotReplying || isBotSpeaking) && (
          <div className={`bot-circle-animation ${isBotSpeaking ? 'speaking' : 'thinking'}`}>
            <div className="pulse-ring"></div>
            <div className="pulse-ring delay"></div>
          </div>
        )}



        {/* Completion animation */}
        {currentStage === "complete" && (
          <div className="completion-animation">
            <span className="checkmark">✅</span>
            <span className="completion-text">Thank you!</span>
          </div>
        )}
      </div>

      {/* Microphone button */}
      <div className="mic-toggle">
        <button
          className={`mic-button ${isRecording ? "recording" : ""} ${isConversationComplete ? "disabled" : ""}`}
          onClick={handleMicClick}
          disabled={isBotReplying || isBotSpeaking || isConversationComplete}
        >
          {isRecording
            ? <FaStop size={22} color="white" />
            : <FaMicrophone size={22} color={isConversationComplete ? "#ccc" : "#007bff"} />
          }
        </button>
        <p className="mic-status">
          {isConversationComplete ? "🎉 Conversation completed!" :
           isRecording ? "🎙️ Recording..." : 
           isBotReplying ? "⏳ Processing..." :
           isBotSpeaking ? "🔊 Playing..." : "Click to start conversation"}
        </p>
        {errorMessage && <p className="error-message">{errorMessage}</p>}
      </div>

      {/* Footer note */}
      <div className="footer-note">
        {isConversationComplete ? 
          "🎉 Thank you for using our service! We'll contact you soon." :
          "💡 Tip: Say your name, phone number, and property needs for personalized recommendations"
        }
      </div>

      {/* End chat button */}
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