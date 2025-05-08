import React, { useState } from 'react';
import { View, Text, Button, ActivityIndicator, TouchableOpacity, Share, Alert } from 'react-native';
import { announceForAccessibility } from '../components/AccessibilityHelper';
import { useTrackBeatGen } from '../components/AnalyticsTracker';
import { generateBeat } from '../components/APIService';

export default function BeatGenScreen() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ beatUrl?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useTrackBeatGen();

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res: any = await generateBeat();
      if (res.success) {
        setResult(res);
        announceForAccessibility('Your beat is ready!');
      } else {
        setError('Beat generation failed.');
      }
    } catch (e) {
      setError('Failed to generate beat. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handlePreview = () => {
    if (result?.beatUrl) {
      // TODO: Implement audio preview (e.g., using Expo AV or react-native-sound)
      Alert.alert('Preview', 'Audio preview would play here.');
    }
  };

  const handleShare = async () => {
    if (result?.beatUrl) {
      try {
        await Share.share({ message: `Check out my new beat! ${result.beatUrl}` });
      } catch (e) {
        setError('Failed to share beat.');
      }
    }
  };

  const handleDownload = () => {
    if (result?.beatUrl) {
      // TODO: Implement download logic (platform-specific)
      Alert.alert('Download', 'Beat download would start here.');
    }
  };

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 }}></View>
      <Text style={{ fontSize: 20, marginBottom: 8 }}>One-Click Beat Generation</Text>
      <Text style={{ color: '#888', marginBottom: 16 }}>Tap below to instantly generate a new beat with AI.</Text>
      <Button title="Generate Beat" onPress={handleGenerate} disabled={loading} accessibilityLabel="Generate a new beat with one click" />
      {loading && <ActivityIndicator style={{ margin: 16 }} accessibilityLabel="Generating beat, please wait" />}
      {error && <Text style={{ color: 'red', marginTop: 8 }}>{error}</Text>}
      {result?.beatUrl && (
        <View style={{ marginTop: 24, alignItems: 'center' }}></View>
          <Text style={{ fontWeight: 'bold', marginBottom: 8 }}>Your Beat is Ready!</Text>
          <TouchableOpacity onPress={handlePreview} style={{ marginBottom: 8 }} accessibilityLabel="Preview your generated beat">
            <Text style={{ color: '#1976d2' }}>▶️ Preview</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={handleDownload} style={{ marginBottom: 8 }} accessibilityLabel="Download your beat">
            <Text style={{ color: '#388e3c' }}>⬇️ Download</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={handleShare} accessibilityLabel="Share your beat">
            <Text style={{ color: '#ff9800' }}>🔗 Share</Text>
          </TouchableOpacity>
        </View>
      )}
      <Text style={{ marginTop: 32, color: '#888', fontSize: 12 }}></Text>
        Tip: You can preview, download, or share your beat with a single tap.
      </Text>
    </View>
  );
}
