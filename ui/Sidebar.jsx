import React from 'react';
import { Link } from 'react-router-dom';

export default function Sidebar() {
  return (
    <nav className="sidebar">
      <ul>
        <li><Link to="/">Dashboard</Link></li>
        <li><Link to="/beat-lab">Beat Lab</Link></li>
        <li><Link to="/marketplace">Marketplace</Link></li>
        <li><Link to="/collab">Collab Universe</Link></li>
        <li><Link to="/profile">Profile</Link></li>
        <li><Link to="/settings">Settings</Link></li>
        <li><Link to="/help">Help & Assistant</Link></li>
      </ul>
    </nav>
  );
}