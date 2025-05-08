import React, { useState } from 'react';

export default function MoodToMusic({ onGenerate }) {
  const [mood, setMood] = useState('');
  return (
    <div className="mood-to-music">
      <h4>Mood-to-Music</h4>
      <input placeholder="Describe your mood or scene..." value={mood} onChange={e => setMood(e.target.value)} />
      <button onClick={() => onGenerate(mood)}>Generate</button>
    </div>
  );
}
