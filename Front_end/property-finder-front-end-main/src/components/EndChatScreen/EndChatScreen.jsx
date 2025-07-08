// EndChatScreen.jsx
import React, { useEffect, useState } from 'react';
import './EndChatScreen.css';

function EndChatScreen({ onRestart }) {
  const [userData, setUserData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/user-latest-data') // Adjust endpoint if needed
      .then((res) => res.json())
      .then((data) => setUserData(data))
      .catch((err) => {
        console.error("Failed to fetch user data:", err);
        setUserData(null);
      });
  }, []);

  return (
    <div className="end-chat-screen">
      <h2>🎉 Thank you for chatting with us!</h2>

      {userData ? (
        <div className="user-summary">
          <p><strong>Name:</strong> {userData.name}</p>
          <p><strong>Phone:</strong> {userData.phone}</p>
          <p><strong>Email:</strong> {userData.email}</p>
          <p><strong>Interested In:</strong> {userData.interested_property}</p>
        </div>
      ) : (
        <p>Loading your submitted data...</p>
      )}

      <button className="restart-button" onClick={onRestart}>
        🔁 Start New Chat
      </button>
    </div>
  );
}

export default EndChatScreen;
