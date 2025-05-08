import React from 'react';
export default function ProcessVisualizer({ steps, currentStep }) {
  return (
    <div>
      <h3>Process Visualizer</h3>
      <ol>
        {steps.map((step, idx) => (
          <li key={idx} style={{ fontWeight: idx === currentStep ? 'bold' : 'normal' }}>{step}</li>
        ))}
      </ol>
    </div>
  );
}
