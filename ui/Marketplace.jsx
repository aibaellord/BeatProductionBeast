import React from 'react';
import Marketplace from './Marketplace';
import SyncMarketplace from './SyncMarketplace';
import Badges from './Badges';
import QuantumUniverseExplorer from './QuantumUniverseExplorer';
import FeedbackForm from './FeedbackForm';
import SmartAssistant from './SmartAssistant';
import Tooltip from './Tooltip';

export default function MarketplacePage() {
  return (
    <div className="marketplace-page">
      <h2>Marketplace</h2>
      <Marketplace />
      <SyncMarketplace />
      <Badges />
      <QuantumUniverseExplorer />
      <FeedbackForm />
      <SmartAssistant />
      <Tooltip text="Buy, sell, license, and explore beats, presets, and collaborations." />
    </div>
  );
}
