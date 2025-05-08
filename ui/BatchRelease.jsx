import React, { useState } from 'react';
export default function BatchRelease() {
  const [batch, setBatch] = useState('');
  const [status, setStatus] = useState('');
  const scheduleBatch = async () => {
    const items = batch.split(',').map(x => x.trim()).filter(Boolean);
    const res = await fetch('/batch-release/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ batch: items }),
    });
    const data = await res.json();
    setStatus(data.status + ' (' + data.count + ' items)');
  };
  return (
    <div className="batch-release">
      <h4>Batch Release</h4>
      <input placeholder="Comma-separated list of tracks" value={batch} onChange={e => setBatch(e.target.value)} />
      <button onClick={scheduleBatch}>Schedule Batch</button>
      <div>{status}</div>
    </div>
  );
}
