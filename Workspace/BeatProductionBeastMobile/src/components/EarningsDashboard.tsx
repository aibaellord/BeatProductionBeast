import React from 'react';
import { View, Text } from 'react-native';

export default function EarningsDashboard({ earnings }: { earnings: number }) {
  return (
    <View style={{ padding: 16, backgroundColor: '#e0ffe0', borderRadius: 8, margin: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Your Earnings</Text>
      <Text style={{ marginVertical: 8, fontSize: 16 }}>${earnings.toFixed(2)}</Text>
      {/* Add payout, referral, and analytics info here */}
    </View>
  );
}
