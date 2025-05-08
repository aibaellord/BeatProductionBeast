import React from 'react';

export default function OnboardingModal({ onClose }) {
  return (
    <div className="onboarding-modal">
      <h2>Welcome to BeatProductionBeast!</h2>
      <p>Get started by generating your first beat, exploring mood-to-music, or fusing genres. Use the dashboard to track your revenue, manage your account, and more. Tooltips and guidance are available throughout the app for every feature.</p>
      <button onClick={onClose}>Get Started</button>
    </div>
  );
}
