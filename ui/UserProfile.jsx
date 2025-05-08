import React from 'react';
import UserProfile from './UserProfile';
import Badges from './Badges';
import FeedbackForm from './FeedbackForm';
import SmartAssistant from './SmartAssistant';
import Tooltip from './Tooltip';

export default function UserProfilePage() {
  return (
    <div className="user-profile-page">
      <h2>Profile</h2>
      <UserProfile />
      <Badges />
      <FeedbackForm />
      <SmartAssistant />
      <Tooltip text="Manage your account, portfolio, and achievements here." />
    </div>
  );
}
