import React from 'react';
import { createRoot } from 'react-dom/client';
// Self-hosted fonts (no network needed): Source Sans 3 for reading and the interface, JetBrains Mono for code.
import '@fontsource-variable/source-sans-3/index.css';
import '@fontsource-variable/source-sans-3/wght-italic.css';
import '@fontsource-variable/jetbrains-mono/index.css';
import './fonts.css';
import './sidebar.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(<App />);
