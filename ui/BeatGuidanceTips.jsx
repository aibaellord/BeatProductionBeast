import React from 'react';

const tips = [
  "Try combining two genres for a unique fusion.",
  "Use the 'Mood-to-Music' feature to generate tracks based on your feelings.",
  "Experiment with the 'Consciousness Slider' for different energy levels.",
  "Batch-generate variations for A/B testing and playlists.",
  "Leverage the 'AI Collab' to expand your own melodies or beats.",
  "Use presets for quick style changes and inspiration.",
  "Mint your best tracks as NFTs with one click.",
  "Check the revenue dashboard for growth tips and payout history.",
  "Share your creations directly to social media for more reach.",
  "Explore the 'Remix Trending' feature to ride viral waves."
];

export default function BeatGuidanceTips() {
  return (
    <div className="beat-guidance-tips">
      <h4>Pro Tips & Guidance</h4>
      <ul>
        {tips.map((tip, idx) => <li key={idx}>{tip}</li>)}
      </ul>
    </div>
  );
}
