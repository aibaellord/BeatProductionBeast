import React, { useState } from 'react';
export default function InvestmentOptions({ onInvest }) {
  const [amount, setAmount] = useState('');
  const [userId, setUserId] = useState('');
  return (
    <div>
      <h3>Investment Options</h3>
      <input placeholder="Amount" value={amount} onChange={e => setAmount(e.target.value)} />
      <input placeholder="User ID" value={userId} onChange={e => setUserId(e.target.value)} />
      <button onClick={() => onInvest(amount, userId)}>Invest</button>
    </div>
  );
}
