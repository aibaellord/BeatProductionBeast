import React from 'react';
import OnboardingModal from './OnboardingModal';
import SmartAssistant from './SmartAssistant';
import Tooltip from './Tooltip';

export default function HelpCenter() {
  return (
    <div className="help-center-page">
      <h2>Help & Assistant</h2>
      <OnboardingModal />
      <SmartAssistant />
      <section>
        <h3>Frequently Asked Questions</h3>
        <ul>
          <li>How do I generate a beat?</li>
          <li>How does one-click automation work?</li>
          <li>How do I join a remix challenge?</li>
          <li>How do I license or sell my beats?</li>
          <li>How do I use the Smart Assistant?</li>
          {/* Add more FAQ items as needed */}
        </ul>
      </section>
      <section>
        <h3>Guided Onboarding</h3>
        <p>Step-by-step walkthroughs and tooltips are available throughout the app. Look for the <b>?</b> icon for contextual help.</p>
      </section>
      <Tooltip text="Get help, onboarding, and answers to all your questions here." />
    </div>
  );
}