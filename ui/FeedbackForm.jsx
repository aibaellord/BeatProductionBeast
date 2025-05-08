import React, { useState } from 'react';
export default function FeedbackForm({ onSubmit }) {
  const [feedback, setFeedback] = useState('');
  return (
    <div>
      <h3>Feedback</h3>
      <textarea value={feedback} onChange={e => setFeedback(e.target.value)} />
      <button onClick={() => onSubmit(feedback)}>Submit</button>
    </div>
  );
}
