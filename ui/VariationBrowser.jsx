import React from 'react';
export default function VariationBrowser({ variations, onPreview }) {
  return (
    <div>
      <h3>Variation Browser</h3>
      <ul>
        {variations.map((variation, idx) => (
          <li key={idx}>
            Variation {idx + 1}
            <button onClick={() => onPreview(variation)}>Preview</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
