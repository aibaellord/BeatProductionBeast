import React from 'react';

export default function ConsciousnessSlider({ value, onChange }) {
  return (
    <div className="consciousness-slider">
      <h4>Consciousness/Energy Level</h4>
      <input type="range" min={1} max={10} value={value} onChange={e => onChange(Number(e.target.value))} />
      <span>{value}</span>
    </div>
  );
}
