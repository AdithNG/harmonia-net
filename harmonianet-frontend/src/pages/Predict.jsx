import React from 'react';
import GenreClassifier from '../components/GenreClassifier';

function Predict() {
  return (
    <div className="predict-page">
      <h2 className="page-title">Upload your track</h2>
      <GenreClassifier />
    </div>
  );
}

export default Predict;
