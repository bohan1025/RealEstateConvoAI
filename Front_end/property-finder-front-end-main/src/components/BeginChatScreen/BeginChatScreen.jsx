// BeginChatScreen.jsx
import React from 'react';
import './BeginChatScreen.css';

function BeginChatScreen({ onStart }) {
  return (
    <div className="begin-chat-screen">
      <div className="floating-icon"></div>
      <div className="floating-icon"></div>
      <div className="floating-icon"></div>

      <div className="background-logo" />
      <div className="floating-bubbles" />
      <div className="glass-card">
        <h1 className="animated-heading shimmer-text">Welcome to Your AI Real Estate Assistant</h1>
        <p className="animated-subtext">Tap below and start speaking. We'll handle the rest.</p>
        <button className="start-button" onClick={onStart}>🎙️ Begin Chat</button>
      </div>
    </div>
  );
}

export default BeginChatScreen;
