import React from 'react';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import OnboardingScreen from '../onboarding/OnboardingScreen';
import BeatGenScreen from '../beatgen/BeatGenScreen';
import MarketplaceScreen from '../marketplace/MarketplaceScreen';
import AssistantScreen from '../assistant/AssistantScreen';
import PremiumUpsell from '../components/PremiumUpsell';
import EarningsDashboard from '../components/EarningsDashboard';
import ReferralProgram from '../components/ReferralProgram';
import InAppPurchase from '../components/InAppPurchase';
import CommunityChat from '../components/CommunityChat';
import Leaderboard from '../components/Leaderboard';
import SubscriptionPlans from '../components/SubscriptionPlans';

const Tab = createBottomTabNavigator();

export default function MainNavigator() {
  return (
    <Tab.Navigator initialRouteName="Onboarding">
      <Tab.Screen name="Onboarding" component={OnboardingScreen} />
      <Tab.Screen name="Generate" component={BeatGenScreen} />
      <Tab.Screen name="Marketplace" component={MarketplaceScreen} />
      <Tab.Screen name="Assistant" component={AssistantScreen} />
      <Tab.Screen name="Premium" children={() => <PremiumUpsell onUpgrade={() => {}} />} />
      <Tab.Screen name="Earnings" children={() => <EarningsDashboard earnings={800} />} />
      <Tab.Screen name="Referral" children={() => <ReferralProgram referralLink="https://beatbeast.com/ref/yourid" />} />
      <Tab.Screen name="Shop" children={() => <InAppPurchase onPurchase={() => {}} />} />
      <Tab.Screen name="Chat" component={CommunityChat} />
      <Tab.Screen name="Leaderboard" component={Leaderboard} />
      <Tab.Screen name="Plans" children={() => <SubscriptionPlans onSubscribe={plan => {/* TODO: Connect to in-app purchase logic */}} />} />
    </Tab.Navigator>
  );
}
