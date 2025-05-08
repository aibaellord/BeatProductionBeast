import React from 'react';
import { useLocation } from 'react-router-dom';

const actionsByPage = {
  '/': [
    { label: 'Generate Beat', action: () => window.scrollTo(0, 0) },
    { label: 'View Analytics', action: () => window.location.hash = '#analytics' },
  ],
  '/beat-lab': [
    { label: 'Generate Beat', action: () => document.querySelector('.one-click-beat-button')?.click() },
    { label: 'Batch Release', action: () => document.querySelector('.batch-release')?.scrollIntoView() },
  ],
  '/marketplace': [
    { label: 'List Beat', action: () => document.querySelector('.marketplace')?.scrollIntoView() },
    { label: 'Sync License', action: () => document.querySelector('.sync-marketplace')?.scrollIntoView() },
  ],
  '/collab': [
    { label: 'Join Challenge', action: () => document.querySelector('.remix-challenge')?.scrollIntoView() },
    { label: 'Start Collab', action: () => document.querySelector('.influencer-collab')?.scrollIntoView() },
  ],
  '/profile': [
    { label: 'Edit Profile', action: () => document.querySelector('.user-profile')?.scrollIntoView() },
  ],
  '/settings': [
    { label: 'Change Theme', action: () => document.querySelector('.settings-page')?.scrollIntoView() },
  ],
  '/help': [
    { label: 'Open FAQ', action: () => document.querySelector('.help-center-page')?.scrollIntoView() },
    { label: 'Contact Support', action: () => alert('Contact support coming soon!') },
  ],
};

export default function QuickActionsBar() {
  const location = useLocation();
  const actions = actionsByPage[location.pathname] || actionsByPage['/'];

  return (
    <div className="quick-actions-bar" role="toolbar" aria-label="Quick Actions">
      {actions.map((a, i) => (
        <button key={i} onClick={a.action} className="quick-action-btn">{a.label}</button>
      ))}
    </div>
  );
}