import React from 'react';
import { View, Text, Button } from 'react-native';

export default function MonetizationOnboarding({ onNext }: { onNext: () => void }) {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 }}>
      <Text style={{ fontSize: 22, fontWeight: 'bold', marginBottom: 16 }}>Earn with BeatProductionBeast</Text>
      <Text style={{ marginBottom: 16 }}>Unlock premium, refer friends, and join the leaderboard to maximize your income. Get paid for your creativity!</Text>
      <Button title="Next" onPress={onNext} />
    </View>
  );
}
