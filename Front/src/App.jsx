// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import MultiSpeechGenerator from './components/MultiSpeechGenerator';
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-8">

          </h1>
          <p className="text-gray-600 mb-8">

          </p>
          <Switch>
            <Route exact path="/" component={MultiSpeechGenerator} />
          </Switch>
        </div>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;
