import React, { useState } from 'react';
export default function SmartAssistant() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const sendMessage = async () => {
    const res = await fetch('/smart-assistant/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input }),
    });
    const data = await res.json();
    setResponse(data.response);
  };
  return (
    <div className="smart-assistant">
      <h4>Smart Assistant</h4>
      <input value={input} onChange={e => setInput(e.target.value)} placeholder="Ask for help or ideas..." />
      <button onClick={sendMessage}>Send</button>
      <div className="assistant-response">{response}</div>
    </div>
  );
}
