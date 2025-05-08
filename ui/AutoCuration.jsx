import React, { useState } from 'react';
export default function AutoCuration() {
  const [userId, setUserId] = useState('');
  const [status, setStatus] = useState('');
  const curate = async () => {
    const res = await fetch('/auto-curation/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId }),
    });
    const data = await res.json();
    setStatus(data.status);
  };
  return (
    <div className="auto-curation">
      <h4>Auto-Curation</h4>
      <input placeholder="User ID" value={userId} onChange={e => setUserId(e.target.value)} />
      <button onClick={curate}>Curate Portfolio</button>
      <div>{status}</div>
    </div>
  );
}
