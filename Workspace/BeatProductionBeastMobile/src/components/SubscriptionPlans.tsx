import React, { useState } from 'react';
import { View, Text, Button } from 'react-native';
import { trackEvent } from './AnalyticsTracker';

const plans = [
  {
    name: 'Free',
    price: 0,
    features: [
      'Limited beat generation',
      'Marketplace browsing',
      'Community chat',
    ],
  },
  {
    name: 'Pro',
    price: 9.99,
    features: [
      'Unlimited beat generation',
      'Premium sound packs',
      'Priority support',
      'Access to remix challenges',
      'Higher earnings split',
    ],
  },
  {
    name: 'Elite',
    price: 29.99,
    features: [
      'All Pro features',
      'AI-powered recommendations',
      'Batch scheduling',
      'NFT minting',
      'Custom beat requests',
      'White-label/enterprise options',
    ],
  },
];

export default function SubscriptionPlans({ onSubscribe }: { onSubscribe: (plan: string) => void }) {
  const [selected, setSelected] = useState<string>('Free');
  return (
    <View style={{ padding: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 22, marginBottom: 16 }}>Choose Your Plan</Text>
      {plans.map(plan => (
        <View key={plan.name} style={{ marginBottom: 24, padding: 16, backgroundColor: selected === plan.name ? '#ffe082' : '#f5f5f5', borderRadius: 8, borderWidth: selected === plan.name ? 2 : 0, borderColor: '#ffb300' }}>
          <Text style={{ fontSize: 18, fontWeight: 'bold' }}>{plan.name} {plan.price > 0 ? `- $${plan.price}/mo` : '(Free)'}</Text>
          {plan.features.map((f, i) => (
            <Text key={i} style={{ marginLeft: 8, marginVertical: 2 }}>• {f}</Text>
          ))}
          <Button title={plan.price > 0 ? `Subscribe to ${plan.name}` : 'Start Free'} onPress={() => { setSelected(plan.name); onSubscribe(plan.name); trackEvent('subscribe_click', { plan: plan.name }); }} />
        </View>
      ))}
      <Text style={{ marginTop: 12, color: '#888' }}>Special: Upgrade to Pro or Elite in the next 24h and get a bonus sound pack!</Text>
    </View>
  );
}
