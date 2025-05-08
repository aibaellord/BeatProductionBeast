import React, { useState, useEffect } from 'react';

export default function QuantumUniverseExplorer() {
  const [seeds, setSeeds] = useState([]);
  const [collabs, setCollabs] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [lineage, setLineage] = useState({ ancestry: [], descendants: [] });
  const [revenue, setRevenue] = useState({ contributors: [], revenue_split: {} });
  const [seedType, setSeedType] = useState('beat');
  const [seedData, setSeedData] = useState('');
  const [collabParents, setCollabParents] = useState('');
  const [status, setStatus] = useState('');

  useEffect(() => {
    fetch('/quantum-universe/explore/')
      .then(res => res.json())
      .then(data => {
        setSeeds(data.seeds || []);
        setCollabs(data.collabs || []);
      });
  }, [status]);

  const handleDropSeed = async () => {
    const res = await fetch('/quantum-universe/seed/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed_type: seedType, data: seedData }),
    });
    const data = await res.json();
    setStatus(data.status);
  };

  const handleCollab = async () => {
    const parentIds = collabParents.split(',').map(x => x.trim()).filter(Boolean);
    const res = await fetch('/quantum-universe/collab/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parent_ids: parentIds }),
    });
    const data = await res.json();
    setStatus(data.status);
  };

  const handleSelectNode = async (nodeId) => {
    setSelectedNode(nodeId);
    const lin = await fetch(`/quantum-universe/lineage/?node_id=${nodeId}`).then(res => res.json());
    setLineage(lin);
    const rev = await fetch(`/quantum-universe/revenue/?node_id=${nodeId}`).then(res => res.json());
    setRevenue(rev);
  };

  return (
    <div className="quantum-universe-explorer">
      <h3>Quantum Collab Universe</h3>
      <div>
        <h4>Drop a Seed</h4>
        <select value={seedType} onChange={e => setSeedType(e.target.value)}>
          <option value="beat">Beat</option>
          <option value="melody">Melody</option>
          <option value="vocal">Vocal</option>
          <option value="mood">Mood</option>
          <option value="idea">Idea</option>
        </select>
        <input placeholder="Seed Data (URL, text, etc.)" value={seedData} onChange={e => setSeedData(e.target.value)} />
        <button onClick={handleDropSeed}>Drop Seed</button>
      </div>
      <div>
        <h4>Trigger Collab/Evolution</h4>
        <input placeholder="Parent Node IDs (comma-separated)" value={collabParents} onChange={e => setCollabParents(e.target.value)} />
        <button onClick={handleCollab}>Collab/Evolve</button>
      </div>
      <div>
        <h4>Universe Graph</h4>
        <div>
          <strong>Seeds:</strong>
          <ul>
            {seeds.map(seed => (
              <li key={seed.seed_id} onClick={() => handleSelectNode(seed.seed_id)} style={{ cursor: 'pointer' }}>
                {seed.seed_id} ({seed.type})
              </li>
            ))}
          </ul>
          <strong>Collabs:</strong>
          <ul>
            {collabs.map(collab => (
              <li key={collab.collab_id} onClick={() => handleSelectNode(collab.collab_id)} style={{ cursor: 'pointer' }}>
                {collab.collab_id} (parents: {collab.parents?.join(', ')})
              </li>
            ))}
          </ul>
        </div>
      </div>
      {selectedNode && (
        <div>
          <h4>Lineage for {selectedNode}</h4>
          <div>Ancestry: {lineage.ancestry.join(', ')}</div>
          <div>Descendants: {lineage.descendants.join(', ')}</div>
          <h4>Revenue Split</h4>
          <div>Contributors: {revenue.contributors.join(', ')}</div>
          <div>Revenue Split: {JSON.stringify(revenue.revenue_split)}</div>
        </div>
      )}
      {status && <div>{status}</div>}
    </div>
  );
}
