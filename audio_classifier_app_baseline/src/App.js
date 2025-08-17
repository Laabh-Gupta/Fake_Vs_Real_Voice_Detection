import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError("Please select a file first.");
      return;
    }

    setIsLoading(true);
    setPrediction(null); // Clear previous results
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok.');
      }

      const data = await response.json();
      setPrediction(data); // Store the entire response object
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎙️ Real vs. Fake Voice Classifier</h1>
        <p>Upload an audio file (.wav or .mp3) to see if it's real or AI-generated.</p>
        
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleFileChange} accept=".wav,.mp3" />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Analyzing...' : 'Classify Audio'}
          </button>
        </form>

        {/* --- THIS IS THE CORRECTED LOGIC --- */}
        
        {/* 1. Show a loading message if processing */}
        {isLoading && <p className="loading">Loading...</p>}
        
        {/* 2. Show an error if the local error state is set OR if the API returned an error object */}
        { (error || (prediction && prediction.error)) && (
          <div className="error">
            Error: {error || prediction.error}
          </div>
        )}
        
        {/* 3. ONLY show the success result if a prediction exists AND it has a 'predicted_class' property */}
        { prediction && prediction.predicted_class && (
          <div className="result">
            <h2>Prediction Result:</h2>
            <p><strong>Filename:</strong> {prediction.filename}</p>
            <p className={`prediction-${prediction.predicted_class}`}>
              <strong>Predicted Class:</strong> {prediction.predicted_class.toUpperCase()}
            </p>
            <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(2)}%</p>
          </div>
        )}

      </header>
    </div>
  );
}

export default App;