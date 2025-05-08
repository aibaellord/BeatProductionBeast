import React, { useState } from 'react';
export default function RemixChallenge() {
  const [challenge, setChallenge] = useState('');
  const [source, setSource] = useState('');
  const [deadline, setDeadline] = useState('');
  const [status, setStatus] = useState('');
  const launchChallenge = async () => {
    const res = await fetch('/remix-challenge/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challenge_name: challenge, source_track: source, deadline }),
    });
    const data = await res.json();
    setStatus(data.status);
  };
  return (
    <div className="remix-challenge">
      <h4>Remix Challenge</h4>
      <input placeholder="Challenge Name" value={challenge} onChange={e => setChallenge(e.target.value)} />
      <input placeholder="Source Track URL" value={source} onChange={e => setSource(e.target.value)} />
      <input placeholder="Deadline" value={deadline} onChange={e => setDeadline(e.target.value)} />
      <button onClick={launchChallenge}>Launch Challenge</button>
      <div>{status}</div>
    </div>
  );
}
