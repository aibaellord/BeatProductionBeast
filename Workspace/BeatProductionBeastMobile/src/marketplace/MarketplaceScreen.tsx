import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, Share, Alert, Button } from 'react-native';
import { useTrackMarketplace } from '../components/AnalyticsTracker';
import { fetchMarketplaceListings } from '../components/APIService';

export default function MarketplaceScreen() {
  const [listings, setListings] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<any | null>(null);
  useTrackMarketplace();

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchMarketplaceListings()
      .then((data: any) => setListings(data))
      .catch(() => setError('Failed to load marketplace.'))
      .finally(() => setLoading(false));
  }, []);

  const handlePreview = (item: any) => {
    // TODO: Implement audio preview (e.g., using Expo AV or react-native-sound)
    Alert.alert('Preview', `Audio preview for ${item.title} would play here.`);
  };

  const handleShare = async (item: any) => {
    try {
      await Share.share({ message: `Check out this beat: ${item.title} - ${item.url}` });
    } catch (e) {
      setError('Failed to share beat.');
    }
  };

  const handleBuy = (item: any) => {
    // TODO: Integrate with in-app purchase/payment logic
    Alert.alert('Buy/License', `Purchase flow for ${item.title} would start here.`);
  };

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: 'bold' }}>Sync Marketplace</Text>
      <Text style={{ color: '#888', marginBottom: 12 }}>Browse, preview, share, or license beats with a single tap.</Text>
      {loading && <Text>Loading...</Text>}
      {error && <Text style={{ color: 'red' }}>{error}</Text>}
      <FlatList
        data={listings}
        keyExtractor={item => item.id}
        renderItem={({ item }) => (
          <View style={{ marginVertical: 8, backgroundColor: '#f5f5f5', borderRadius: 8, padding: 12 }}>
            <Text style={{ fontSize: 16, fontWeight: 'bold' }}>{item.title} - ${item.price}</Text>
            <View style={{ flexDirection: 'row', marginTop: 8 }}>
              <TouchableOpacity onPress={() => handlePreview(item)} accessibilityLabel={`Preview ${item.title}`} style={{ marginRight: 16 }}>
                <Text style={{ color: '#1976d2' }}>▶️ Preview</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => handleShare(item)} accessibilityLabel={`Share ${item.title}`} style={{ marginRight: 16 }}>
                <Text style={{ color: '#ff9800' }}>🔗 Share</Text>
              </TouchableOpacity>
              <TouchableOpacity onPress={() => handleBuy(item)} accessibilityLabel={`Buy or license ${item.title}`}>
                <Text style={{ color: '#388e3c' }}>💸 Buy/License</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      />
      <Text style={{ marginTop: 24, color: '#888', fontSize: 12 }}>
        Tip: Tap any beat to preview, share, or license it instantly.
      </Text>
    </View>
  );
}
