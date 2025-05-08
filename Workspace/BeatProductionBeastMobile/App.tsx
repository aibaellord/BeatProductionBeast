import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import MainNavigator from './src/navigation/MainNavigator';
import { LocalizationProvider } from './src/components/LocalizationProvider';
import { useDeepLinks } from './src/components/DeepLinkHandler';

export default function App() {
  // Handle deep links for referrals, marketing, notifications
  useDeepLinks((url) => {
    // TODO: Parse URL and route to appropriate screen (e.g., referral, premium, chat)
    // Example: if (url.includes('ref')) { ... }
  });

  return (
    <LocalizationProvider>
      <NavigationContainer>
        <MainNavigator />
      </NavigationContainer>
    </LocalizationProvider>
  );
}
