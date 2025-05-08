import React, { useState } from 'react';

const defaultPlugins = [
  { name: 'RemixEnhancer', enabled: true, description: 'Advanced remix and variation generator.' },
  { name: 'AutoUploader', enabled: true, description: 'Automated browser-based upload to YouTube and more.' },
  { name: 'AnalyticsPro', enabled: false, description: 'Enhanced local analytics and reporting.' }
];

export default function PluginManager() {
  const [plugins, setPlugins] = useState(defaultPlugins);

  const togglePlugin = idx => {
    setPlugins(p => p.map((pl, i) => i === idx ? { ...pl, enabled: !pl.enabled } : pl));
  };

  return (
    <div className="plugin-manager">
      <h4>Plugin Manager</h4>
      <ul>
        {plugins.map((plugin, idx) => (
          <li key={plugin.name}>
            <label>
              <input type="checkbox" checked={plugin.enabled} onChange={() => togglePlugin(idx)} />
              <b>{plugin.name}</b>: {plugin.description}
            </label>
          </li>
        ))}
      </ul>
      <div className="plugin-info">Add, enable, or disable automation and AI plugins here. (Future: plugin marketplace, upload, and config)</div>
    </div>
  );
}