// BeginChatScreen.jsx
import React from 'react';
import './BeginChatScreen.css'; // custom styles

// This component shows a welcome screen with a button to start the chat
function BeginChatScreen({ onStart }) {
  return (
    <div className="begin-chat-screen">
      <h1>🎤 Welcome to Voice Assistant</h1>
      <p>Click the button below to start your conversation</p>
      <button className="start-button" onClick={onStart}>
        Begin Chat
      </button>
    </div>
  );
}

export default BeginChatScreen;
