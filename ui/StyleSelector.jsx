import React from 'react';

const styles = [
  "Trap", "Lofi", "Ambient", "Drill", "Cyberpunk", "Meditation", "Jazz", "Funk", "Synthwave", "Pop", "EDM", "Classical", "World", "Chillhop", "Reggaeton", "Afrobeats", "Rock", "Blues", "Country", "Experimental"
];

export default function StyleSelector({ onSelect }) {
  return (
    <div className="style-selector">
      <h4>Select a Style/Genre</h4>
      <select onChange={e => onSelect(e.target.value)}>
        {styles.map((style, idx) => <option key={idx} value={style}>{style}</option>)}
      </select>
    </div>
  );
}
