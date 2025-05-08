import React from 'react';
import { View, Text, Button } from 'react-native';

export default function PremiumUpsell({ onUpgrade }: { onUpgrade: () => void }) {
  return (
    <View style={{ padding: 16, backgroundColor: '#ffeedd', borderRadius: 8, margin: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Unlock Premium</Text>
      <Text style={{ marginVertical: 8 }}>Get unlimited beat generation, exclusive sounds, and higher earnings.</Text>
      <Text style={{ color: '#d84315', fontWeight: 'bold', marginTop: 8 }}>🔥 Limited Time: 50% off your first month!</Text>
      <Button title="Upgrade Now" onPress={onUpgrade} />
    </View>
  );
}
