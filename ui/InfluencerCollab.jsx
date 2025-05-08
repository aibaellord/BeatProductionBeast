import React, { useState } from 'react';
export default function InfluencerCollab() {
  const [beatId, setBeatId] = useState('');
  const [influencer, setInfluencer] = useState('');
  const [status, setStatus] = useState('');
  const collab = async () => {
    const res = await fetch('/influencer-collab/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ beat_id: beatId, influencer_handle: influencer }),
    });
    const data = await res.json();
    setStatus(data.status);
  };
  return (
    <div className="influencer-collab">
      <h4>Influencer Collab</h4>
      <input placeholder="Beat ID" value={beatId} onChange={e => setBeatId(e.target.value)} />
      <input placeholder="Influencer Handle" value={influencer} onChange={e => setInfluencer(e.target.value)} />
      <button onClick={collab}>Initiate Collab</button>
      <div>{status}</div>
    </div>
  );
}
