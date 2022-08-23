import React from 'react';
import { createRoot } from 'react-dom/client';

import "normalize.css";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";

import App from './App';


const root = createRoot(document.getElementById('root'))
root.render(<App />)
