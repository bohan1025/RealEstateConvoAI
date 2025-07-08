// Import core React libraries
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

// Import the main App component
import App from './App';

// Get the root DOM element from index.html where React will render
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the App component into the root element
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
