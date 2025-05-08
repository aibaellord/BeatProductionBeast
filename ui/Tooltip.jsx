import React from 'react';

export default function Tooltip({ text, children }) {
  return (
    <span className="tooltip-container">
      {children}
      <span className="tooltip-text">{text}</span>
    </span>
  );
}
