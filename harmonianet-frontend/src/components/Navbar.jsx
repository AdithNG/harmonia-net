import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';
import logo from '../assets/logo.png';

function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-logo">
        <img src={logo} alt="HarmoniaNet" />
        HarmoniaNet
      </Link>
      <div className="navbar-links">
        <Link to="/">Home</Link>
        <Link to="/predict">Try the Classifier</Link>
      </div>
    </nav>
  );
}

export default Navbar;
