import React, { useState } from 'react';
import BeginChatScreen from './components/BeginChatScreen/BeginChatScreen';
import VoiceChatScreen from './components/VoiceChatScreen/VoiceChatScreen';
import EndChatScreen from './components/EndChatScreen/EndChatScreen';

function App() {
  const [screen, setScreen] = useState('begin');

  const handleRestart = () => {
    setScreen('begin'); // Restarts at the welcome screen
  };

  return (
    <>
      {screen === 'begin' && <BeginChatScreen onStart={() => setScreen('chat')} />}
      {screen === 'chat' && <VoiceChatScreen onEndChat={() => setScreen('end')} />}
      {screen === 'end' && <EndChatScreen onRestart={handleRestart} />}
    </>
  );
}

export default App;  // ✅ This was missing
