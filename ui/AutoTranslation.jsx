import React, { useState } from 'react';
export default function AutoTranslation() {
  const [text, setText] = useState('');
  const [lang, setLang] = useState('');
  const [output, setOutput] = useState('');
  const translate = async () => {
    const res = await fetch('/auto-translation/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, target_lang: lang }),
    });
    const data = await res.json();
    setOutput(data.output);
  };
  return (
    <div className="auto-translation">
      <h4>Auto-Translation</h4>
      <input placeholder="Text to translate" value={text} onChange={e => setText(e.target.value)} />
      <input placeholder="Target Language (e.g. es, fr, zh)" value={lang} onChange={e => setLang(e.target.value)} />
      <button onClick={translate}>Translate</button>
      <div>{output}</div>
    </div>
  );
}
