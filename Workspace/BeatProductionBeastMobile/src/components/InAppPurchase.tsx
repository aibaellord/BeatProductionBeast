import React from 'react';
import { View, Text, Button } from 'react-native';
import { trackEvent } from './AnalyticsTracker';

export default function InAppPurchase({ onPurchase }: { onPurchase: () => void }) {
  return (
    <View style={{ padding: 16, backgroundColor: '#fff3e0', borderRadius: 8, margin: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>In-App Purchases</Text>
      <Text style={{ marginVertical: 8 }}>Unlock exclusive packs, features, and more with a single tap.</Text>
      <Text style={{ color: '#388e3c', fontWeight: 'bold', marginTop: 8 }}>🎁 Buy now and get a free exclusive sound pack!</Text>
      <Button title="Buy Now" onPress={() => { onPurchase(); trackEvent('inapp_purchase_click'); }} />
    </View>
  );
}
