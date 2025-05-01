import React from 'react';
import './Home.css';

function Home() {
  return (
    <div className="home-page">
      <h1>Welcome to HarmoniaNet</h1>
      <p>
        Upload music. Get predictions. Understand your genre with the power of deep learning.
      </p>
      <a href="/predict" className="home-button">Try It Now</a>
    </div>
  );
}

export default Home;
