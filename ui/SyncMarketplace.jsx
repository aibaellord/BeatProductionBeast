import React, { useState } from 'react';

export default function SyncMarketplace() {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState('');
  const [uploadStatus, setUploadStatus] = useState('');
  const [projectFile, setProjectFile] = useState(null);
  const [projectType, setProjectType] = useState('video');
  const [matchResults, setMatchResults] = useState([]);
  const [licenseStatus, setLicenseStatus] = useState('');
  const [listings, setListings] = useState([]);
  const [analytics, setAnalytics] = useState({ sales: 0, revenue: 0, top_beats: [] });

  // Upload beat for sync licensing
  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('beat_file', file);
    formData.append('metadata', metadata);
    const res = await fetch('/sync-marketplace/upload/', { method: 'POST', body: formData });
    const data = await res.json();
    setUploadStatus(data.status + ' ' + (data.license_url || ''));
  };

  // Match project to beats
  const handleMatch = async () => {
    const formData = new FormData();
    formData.append('project_file', projectFile);
    formData.append('project_type', projectType);
    const res = await fetch('/sync-marketplace/match/', { method: 'POST', body: formData });
    const data = await res.json();
    setMatchResults(data.matches || []);
  };

  // License a beat
  const handleLicense = async (beatId) => {
    const res = await fetch('/sync-marketplace/license/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ beat_id: beatId, buyer_info: { email: 'buyer@example.com' } }),
    });
    const data = await res.json();
    setLicenseStatus(data.status + ' ' + (data.license_url || ''));
  };

  // Fetch listings and analytics on mount
  React.useEffect(() => {
    fetch('/sync-marketplace/listings/').then(res => res.json()).then(data => setListings(data.listings || []));
    fetch('/sync-marketplace/analytics/').then(res => res.json()).then(setAnalytics);
  }, []);

  return (
    <div className="sync-marketplace">
      <h3>AI Sync Marketplace</h3>
      <div>
        <h4>Upload Beat for Sync Licensing</h4>
        <input type="file" onChange={e => setFile(e.target.files[0])} />
        <input placeholder="Metadata (JSON)" value={metadata} onChange={e => setMetadata(e.target.value)} />
        <button onClick={handleUpload}>Upload</button>
        <div>{uploadStatus}</div>
      </div>
      <div>
        <h4>Project-to-Beat Matching</h4>
        <input type="file" onChange={e => setProjectFile(e.target.files[0])} />
        <select value={projectType} onChange={e => setProjectType(e.target.value)}>
          <option value="video">Video</option>
          <option value="ad">Ad</option>
          <option value="game">Game</option>
          <option value="podcast">Podcast</option>
        </select>
        <button onClick={handleMatch}>Match Beats</button>
        <ul>
          {matchResults.map((id, idx) => (
            <li key={idx}>{id} <button onClick={() => handleLicense(id)}>License</button></li>
          ))}
        </ul>
        <div>{licenseStatus}</div>
      </div>
      <div>
        <h4>Available Beats for Sync</h4>
        <ul>
          {listings.map((beat, idx) => (
            <li key={idx}>{beat.beat_id} ({beat.tags?.join(', ')})</li>
          ))}
        </ul>
      </div>
      <div>
        <h4>Marketplace Analytics</h4>
        <div>Sales: {analytics.sales}</div>
        <div>Revenue: ${analytics.revenue}</div>
        <div>Top Beats: {analytics.top_beats?.join(', ')}</div>
      </div>
    </div>
  );
}
