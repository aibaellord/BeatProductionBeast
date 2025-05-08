import React from 'react';
export default function RevenueDashboard({ dashboard }) {
  return (
    <div>
      <h3>Revenue Dashboard</h3>
      <div>Total Earned: ${dashboard.total_earned}</div>
      <div>NFT Sales: {dashboard.nft_sales}</div>
      <div>Subscriptions: {dashboard.subscriptions}</div>
      {/* Add more details as needed */}
    </div>
  );
}
