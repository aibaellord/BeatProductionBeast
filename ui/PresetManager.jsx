import React from 'react';
export default function PresetManager({ presets, onSelect }) {
  return (
    <div>
      <h3>Presets</h3>
      <ul>
        {presets.map((preset, idx) => (
          <li key={idx} onClick={() => onSelect(preset)}>{preset.name}</li>
        ))}
      </ul>
    </div>
  );
}
