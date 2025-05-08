import React, { useEffect, useState } from 'react';
import { View, Text, FlatList } from 'react-native';
import { trackEvent } from './AnalyticsTracker';

export default function Leaderboard() {
  const [leaders, setLeaders] = useState<{user: string, earnings: number}[]>([]);
  useEffect(() => {
    // TODO: Fetch from backend
    setLeaders([
      { user: 'Alice', earnings: 1200 },
      { user: 'Bob', earnings: 950 },
      { user: 'You', earnings: 800 },
    ]);
  }, []);
  return (
    <View style={{ padding: 16, backgroundColor: '#f3e5f5', borderRadius: 8, margin: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Top Earners Leaderboard</Text>
      <FlatList
        data={leaders}
        keyExtractor={item => item.user}
        renderItem={({ item, index }) => (
          <Text style={{ marginVertical: 2 }}>{index + 1}. {item.user} - ${item.earnings}</Text>
        )}
      />
      <Text style={{ marginTop: 12, color: '#1976d2', fontWeight: 'bold' }} onPress={() => trackEvent('leaderboard_prize_click')}>Compete weekly for cash prizes and exclusive features!</Text>
    </View>
  );
}
