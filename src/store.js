import { CLIP_SIZE } from './dsp.js';

let dbPromise;
function openDB() {
  if (!dbPromise) dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open('keytone-lab', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('workspace');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}
export async function loadState() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const req = db.transaction('workspace').objectStore('workspace').get('current');
    req.onsuccess = () => resolve(req.result || null); req.onerror = () => reject(req.error);
  });
}
export async function saveState(state) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('workspace', 'readwrite');
    tx.objectStore('workspace').put(state, 'current');
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error); tx.onabort = () => reject(tx.error || new Error('Storage write aborted.'));
  });
}
export function exportDataset(sessions) {
  return { kind: 'keytone-dataset', version: 1, sampleRate: 16000, clipSize: CLIP_SIZE,
    sessions: sessions.map(s => ({ ...s, clips: s.clips.map(c => ({ ...c, samples: Array.from(c.samples) })) })) };
}
export function validateDataset(raw) {
  if (!raw || raw.kind !== 'keytone-dataset' || raw.version !== 1 || raw.sampleRate !== 16000 || raw.clipSize !== CLIP_SIZE) throw new Error('Use a Keytone dataset export (16 kHz, version 1).');
  if (!Array.isArray(raw.sessions) || !raw.sessions.length || raw.sessions.length > 100) throw new Error('The dataset must have 1–100 sessions.');
  const ids = new Set(), clipIds = new Set(); let count = 0;
  return raw.sessions.map(s => {
    if (!s || typeof s.id !== 'string' || !s.id || s.id.length > 150 || ids.has(s.id)) throw new Error('Invalid or duplicate session ID.');
    ids.add(s.id);
    if (!['recorded', 'synthetic'].includes(s.source) || typeof s.name !== 'string' || s.name.length > 100 || !Array.isArray(s.clips)) throw new Error('Invalid recording session.');
    return { id: s.id, name: s.name, source: s.source, createdAt: typeof s.createdAt === 'string' ? s.createdAt.slice(0, 50) : '', clips: s.clips.map(c => {
      if (++count > 5000) throw new Error('Import up to 5,000 clips at a time.');
      if (!c || typeof c.id !== 'string' || !c.id || c.id.length > 150 || clipIds.has(c.id) || !/^[A-Z0-9]$/.test(c.label)) throw new Error('Invalid clip label or duplicate clip ID.');
      clipIds.add(c.id);
      if (!Array.isArray(c.samples) || c.samples.length !== CLIP_SIZE || c.samples.some(x => typeof x !== 'number' || !Number.isFinite(x) || Math.abs(x) > 1)) throw new Error('A clip has invalid audio samples.');
      return { id: c.id, label: c.label, samples: Float32Array.from(c.samples), createdAt: typeof c.createdAt === 'string' ? c.createdAt.slice(0, 50) : '' };
    }) };
  });
}
