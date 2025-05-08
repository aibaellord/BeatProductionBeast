import React, { useEffect, useState } from 'react';

export default function MarketingDashboard() {
  const [analytics, setAnalytics] = useState({ reach: 0, engagement: 0, conversion: 0, roi: 0, top_channels: [] });

  useEffect(() => {
    fetch('/marketing-analytics/')
      .then(res => res.json())
      .then(data => setAnalytics(data));
  }, []);

  return (
    <div className="marketing-dashboard">
      <h3>Marketing Analytics</h3>
      <div>Reach: {analytics.reach}</div>
      <div>Engagement: {analytics.engagement}</div>
      <div>Conversion: {analytics.conversion}</div>
      <div>ROI: {analytics.roi}</div>
      <div>Top Channels: {analytics.top_channels?.join(', ')}</div>
    </div>
  );
}
