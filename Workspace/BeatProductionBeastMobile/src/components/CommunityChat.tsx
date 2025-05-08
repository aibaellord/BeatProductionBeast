import React, { useState } from 'react';
import { View, Text, TextInput, Button, FlatList } from 'react-native';

export default function CommunityChat() {
  const [messages, setMessages] = useState<{user: string, text: string}[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = () => {
    setLoading(true);
    setError(null);
    try {
      if (input.trim()) {
        setMessages([...messages, { user: 'You', text: input }]);
        setInput('');
        // TODO: Integrate with backend chat or real-time service
        // TODO: Track analytics event for chat message
      }
      setLoading(false);
    } catch (e) {
      setError('Failed to send message. Please try again.');
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontWeight: 'bold', fontSize: 18 }}>Community Chat</Text>
      <FlatList
        data={messages}
        keyExtractor={(_, idx) => idx.toString()}
        renderItem={({ item }) => (
          <Text style={{ marginVertical: 2 }}><Text style={{ fontWeight: 'bold' }}>{item.user}:</Text> {item.text}</Text>
        )}
        style={{ marginVertical: 8, maxHeight: 200 }}
      />
      {loading && <Text>Sending...</Text>}
      {error && <Text style={{ color: 'red', marginBottom: 8 }}>{error}</Text>}
      <TextInput
        style={{ borderWidth: 1, borderColor: '#ccc', padding: 8, marginBottom: 8 }}
        value={input}
        onChangeText={setInput}
        placeholder="Type your message..."
      />
      <Button title="Send" onPress={sendMessage} />
    </View>
  );
}

// TODO: Add unit and integration tests for CommunityChat
