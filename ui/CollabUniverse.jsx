import React from 'react';
import RemixChallenge from './RemixChallenge';
import InfluencerCollab from './InfluencerCollab';
import QuantumUniverseExplorer from './QuantumUniverseExplorer';
import Badges from './Badges';
import FeedbackForm from './FeedbackForm';
import SmartAssistant from './SmartAssistant';
import Tooltip from './Tooltip';

export default function CollabUniverse() {
  return (
    <div className="collab-universe-page">
      <h2>Collab & Quantum Universe</h2>
      <RemixChallenge />
      <InfluencerCollab />
      <QuantumUniverseExplorer />
      <Badges />
      <FeedbackForm />
      <SmartAssistant />
      <Tooltip text="Join remix challenges, collaborate, and explore the quantum universe of music." />
    </div>
  );
}