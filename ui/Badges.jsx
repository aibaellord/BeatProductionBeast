import React from 'react';
export default function Badges() {
  return (
    <div className="badges">
      <img src="https://img.shields.io/github/workflow/status/username/BeatProductionBeast/CI" alt="Build Status" />
      <img src="https://img.shields.io/codecov/c/github/username/BeatProductionBeast" alt="Coverage" />
      <img src="https://img.shields.io/github/license/username/BeatProductionBeast" alt="License" />
    </div>
  );
}
