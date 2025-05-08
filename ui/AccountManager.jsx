import React, { useState } from 'react';

export default function AccountManager({ user, onLogout }) {
  const [showCreate, setShowCreate] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [status, setStatus] = useState('');

  const handleCreateAccount = async () => {
    const res = await fetch('/create-account/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, email, password }),
    });
    const data = await res.json();
    setStatus(data.status);
    if (data.api_key) setApiKey(data.api_key);
  };

  const handleGenerateApiKey = async () => {
    const res = await fetch('/generate-api-key/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: user?.id }),
    });
    const data = await res.json();
    setApiKey(data.api_key);
    setStatus(data.status);
  };

  return (
    <div className="account-manager">
      <h3>Account</h3>
      <div>User: {user?.name || 'Guest'}</div>
      <button onClick={onLogout}>Logout</button>
      <button onClick={() => setShowCreate(!showCreate)}>{showCreate ? 'Cancel' : 'Create Account'}</button>
      {showCreate && (
        <div className="create-account-form">
          <input placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} />
          <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
          <input placeholder="Password" type="password" value={password} onChange={e => setPassword(e.target.value)} />
          <button onClick={handleCreateAccount}>Create</button>
        </div>
      )}
      <button onClick={handleGenerateApiKey}>Generate API Key</button>
      {apiKey && <div>API Key: <code>{apiKey}</code></div>}
      {status && <div>{status}</div>}
    </div>
  );
}
