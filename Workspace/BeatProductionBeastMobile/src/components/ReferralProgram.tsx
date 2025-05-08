import React from 'react';
import { View, Text, Button, Clipboard } from 'react-native';

export default function ReferralProgram({ referralLink }: { referralLink: string }) {
  const copyToClipboard = () => {
    Clipboard.setString(referralLink);
  };
  return (
    <View style={{ padding: 16, backgroundColor: '#e0f7fa', borderRadius: 8, margin: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Referral Program</Text>
      <Text style={{ marginVertical: 8 }}>Share your link and earn passive income for every new user or sale!</Text>
      <Text selectable style={{ marginBottom: 8 }}>{referralLink}</Text>
      <Button title="Copy Link" onPress={copyToClipboard} />
    </View>
  );
}
