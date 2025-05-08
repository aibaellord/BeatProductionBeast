import React, { useState } from 'react';

const genres = [
  "Trap", "Lofi", "Ambient", "Drill", "Cyberpunk", "Jazz", "Funk", "Synthwave", "Pop", "EDM", "Classical", "World", "Chillhop", "Reggaeton", "Afrobeats", "Rock", "Blues", "Country", "Experimental"
];

export default function GenreFusion({ onFuse }) {
  const [selected, setSelected] = useState([]);
  const toggleGenre = genre => setSelected(selected.includes(genre) ? selected.filter(g => g !== genre) : [...selected, genre]);
  return (
    <div className="genre-fusion">
      <h4>Genre Fusion</h4>
      <div>
        {genres.map((genre, idx) => (
          <label key={idx}>
            <input type="checkbox" checked={selected.includes(genre)} onChange={() => toggleGenre(genre)} />
            {genre}
          </label>
        ))}
      </div>
      <button onClick={() => onFuse(selected)}>Fuse Selected Genres</button>
    </div>
  );
}
