import React from 'react';
import OneClickBeatButton from './OneClickBeatButton';
import PresetManager from './PresetManager';
import ProcessVisualizer from './ProcessVisualizer';
import VariationBrowser from './VariationBrowser';
import AnalyticsDashboard from './AnalyticsDashboard';
import RevenueDashboard from './RevenueDashboard';
import BatchRelease from './BatchRelease';
import AutoCuration from './AutoCuration';
import RemixChallenge from './RemixChallenge';
import Marketplace from './Marketplace';
import BeatGuidanceTips from './BeatGuidanceTips';
import MoodToMusic from './MoodToMusic';
import GenreFusion from './GenreFusion';
import ConsciousnessSlider from './ConsciousnessSlider';
import StyleSelector from './StyleSelector';
import SmartAssistant from './SmartAssistant';
import FeedbackForm from './FeedbackForm';
import Badges from './Badges';
import QuantumUniverseExplorer from './QuantumUniverseExplorer';
import Tooltip from './Tooltip';

export default function BeatLab() {
  return (
    <div className="beat-lab-page">
      <h2>Beat Lab</h2>
      <StyleSelector />
      <ConsciousnessSlider />
      <OneClickBeatButton />
      <PresetManager />
      <ProcessVisualizer />
      <VariationBrowser />
      <MoodToMusic />
      <GenreFusion />
      <BatchRelease />
      <AutoCuration />
      <RemixChallenge />
      <BeatGuidanceTips />
      <FeedbackForm />
      <Badges />
      <SmartAssistant />
      <Tooltip text="Explore advanced beat creation, remix, and automation tools here." />
    </div>
  );
}