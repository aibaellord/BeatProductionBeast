import React, { useState } from 'react';
import { View, Text, Button } from 'react-native';
import MonetizationOnboarding from './MonetizationOnboarding';
import { useTrackOnboarding } from '../components/AnalyticsTracker';

// TODO: Integrate onboarding progress with backend API
// TODO: Track analytics events for onboarding steps
// TODO: Add localization for onboarding text
// TODO: Add unit and integration tests for OnboardingScreen

export default function OnboardingScreen({ navigation }) {
  const [showMonetization, setShowMonetization] = useState(false);
  useTrackOnboarding();
  if (showMonetization) {
    return <MonetizationOnboarding onNext={() => navigation.navigate('Generate')} />;
  }
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold' }}>Welcome to BeatProductionBeast!</Text>
      <Text style={{ marginVertical: 16 }}>Create, remix, and license beats with AI-powered automation.</Text>
      <Button title="Get Started" onPress={() => setShowMonetization(true)} />
    </View>
  );
}
