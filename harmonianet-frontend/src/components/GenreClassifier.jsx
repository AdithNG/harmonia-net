import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import './GenreClassifier.css';

function GenreClassifier() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults(null);
    setError('');
    setFeedbackSent(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Server error');
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Error:', err);
      setError('Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = (correct) => {
    setFeedbackSent(true);
    alert(correct ? 'Thanks for confirming!' : 'Thanks for your feedback!');
  };

  const reset = () => {
    setFile(null);
    setResults(null);
    setError('');
    setFeedbackSent(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <motion.div
      className="genre-classifier"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          ref={fileInputRef}
          accept="audio/*"
          onChange={handleFileChange}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !file}>
          {loading ? 'Analyzing...' : 'Predict Genre'}
        </button>
      </form>

      {loading && (
        <div className="progress-bar">
          <div className="progress-fill" />
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {results && Object.keys(results).length > 0 && (() => {
        const sorted = Object.entries(results).sort((a, b) => b[1] - a[1]);
        const [topGenre, topProb] = sorted[0];

        return (
          <motion.div
            className="results-wrapper"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <div className="top-prediction">
              🎵 <strong>Predicted Genre:</strong>{' '}
              <span>{topGenre}</span> ({(topProb * 100).toFixed(2)}%)
            </div>

            <div className="results">
              <h3>Full Prediction Breakdown:</h3>
              <ul>
                {sorted.map(([genre, prob]) => (
                  <li key={genre}>
                    <strong>{genre}</strong>: {(prob * 100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>

            {!feedbackSent && (
              <div className="feedback-section">
                <p>Was this classification correct?</p>
                <button onClick={() => handleFeedback(true)}>👍 Yes</button>
                <button onClick={() => handleFeedback(false)}>👎 No</button>
              </div>
            )}

            <button className="reset-button" onClick={reset}>
              🔁 Choose Another File
            </button>
          </motion.div>
        );
      })()}
    </motion.div>
  );
}

export default GenreClassifier;
