import React from 'react';

export default function AnalyticsDashboard({ analytics }) {
  return (
    <div className="analytics-dashboard">
      <h3>Analytics</h3>
      <div>Users: {analytics.users}</div>
      <div>Beats Generated: {analytics.beats_generated}</div>
      <div>Revenue: ${analytics.revenue}</div>
      <div>Active Sessions: {analytics.active_sessions}</div>
      <div>Top Styles: {analytics.top_styles?.join(', ')}</div>
      <div>Conversion Rate: {analytics.conversion_rate * 100}%</div>
    </div>
  );
}
