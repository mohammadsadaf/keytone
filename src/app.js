import { CLIP_SIZE, SAMPLE_RATE, audioQuality, segmentOnsets, encodeWav } from './dsp.js';
import { createDemo, syntheticPress } from './demo.js';
import { Microphone, decodeFile, playSamples } from './audio.js';
import { prepareSplit, predict, evaluate, serializeModel, validateModel, audioSignature } from './model.js';
import { loadState, saveState, exportDataset, validateDataset } from './store.js';
import { icon, escapeHTML as esc, percent, drawWave, drawSpectrum, drawLearning } from './ui.js';

const $ = selector => document.querySelector(selector);
const uid = () => crypto.randomUUID();
const state = { sessions: [], source: 'recorded', tab: 'capture', selectedKeys: ['A', 'S', 'D', 'F', 'J', 'K'], target: 25, models: {}, reports: {}, heldOut: {}, epochs: 45, augmentation: true, filterKey: 'all', filterSession: 'all', page: 0, selectedClip: null };
let microphone = null, recordingSession = null, recording = false, micStarting = false, pendingClip = false, lastPress = 0, captureGeneration = 0, activeKey = 'A';
let worker = null, training = false, history = [], trainStatus = '', trainProgress = 0;
let toastTimer, saveTimer, waveFrame, prediction = null, predicting = false, uploadData = null, importMode = 'dataset';
const sessions = () => state.sessions.filter(s => s.source === state.source);
const allClips = () => sessions().flatMap(s => s.clips.map(c => ({ ...c, sessionId: s.id, sessionName: s.name })));
const model = () => state.models[state.source];
const report = () => state.reports[state.source];
const isBusy = () => recording || micStarting || training || predicting;

function toast(message, error = false) {
  const element = $('#toast'); element.textContent = message; element.className = `visible ${error ? 'error' : ''}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => element.className = '', 5500);
}
function persist() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveState({ ...state, tab: state.tab }).catch(() => toast('Browser storage is full or unavailable. Export your dataset to keep it.', true)), 300);
}
function invalidate() { delete state.models[state.source]; delete state.reports[state.source]; prediction = null; }
function clipById(id) { return allClips().find(c => c.id === id); }
function btn(label, action, style = '', svg = '', extra = '') {
  return `<button class="button ${style}" data-action="${action}" ${extra}>${svg ? icon(svg, 16) : ''}${label}</button>`;
}
function sectionHead(eyebrow, title, description, actions = '') {
  return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${description}</p></div><div class="heading-actions">${actions}</div></div>`;
}
function empty(title, text, action = '') {
  return `<div class="empty-state"><span class="empty-icon">${icon('wave', 30)}</span><h3>${title}</h3><p>${text}</p>${action}</div>`;
}
function sidebar() {
  return `<aside class="sidebar"><a href="#capture" class="brand" aria-label="Keytone home"><span class="brand-mark">${icon('wave', 24)}</span>keytone<span class="brand-period">.</span></a>
    <div class="workspace-switch"><span class="workspace-icon">${icon('folder', 18)}</span><div>Personal workspace<small>Acoustic learning lab</small></div><span class="version">01</span></div>
    <div class="nav-label">EXPERIMENT</div><nav aria-label="Main navigation">${[
      ['capture', 'mic', 'Capture', '01'], ['dataset', 'layers', 'Dataset', '02'], ['train', 'brain', 'Train model', '03'], ['results', 'chart', 'Results', '04']
    ].map(([tab, svg, label, n]) => `<button data-tab="${tab}" aria-label="${label}" class="nav-item ${state.tab === tab ? 'active' : ''}" ${state.tab === tab ? 'aria-current="page"' : ''}>${icon(svg, 19)}<span>${label}</span><small>${n}</small></button>`).join('')}</nav>
    <div class="nav-label learn-label">LEARN</div><button class="nav-item ${state.tab === 'guide' ? 'active' : ''}" data-tab="guide">${icon('book', 19)}<span>How it works</span>${icon('chevron', 14)}</button>
    <a class="nav-item" href="https://arxiv.org/abs/2308.01074" target="_blank" rel="noreferrer">${icon('up', 18)}<span>Read the research</span></a>
    <div class="sidebar-bottom"><div class="privacy-card"><span>${icon('lock', 18)} YOUR LAB, YOUR DATA</span><p>Recorded here.<br>Trained here.<br>Kept right here.</p><small><i class="status-dot"></i> 100% local processing</small></div><div class="sidebar-footer"><span class="avatar">K</span><div>Acoustic lab<small>v1.0 · Open source</small></div></div></div>
  </aside>`;
}
function render() {
  cancelAnimationFrame(waveFrame);
  $('#app').innerHTML = `${sidebar()}<div class="main-shell"><header class="topbar"><div class="breadcrumb">Workspace ${icon('chevron', 12)} <span>Acoustic lab</span></div><div class="topbar-right"><span class="local-status"><i class="status-dot"></i> Running locally</span><button class="icon-button" data-tab="guide" aria-label="Help">${icon('info', 19)}</button></div></header>
    <main id="main-content"><div class="workspace-toolbar"><span class="workspace-title">${icon('grid', 15)} THE ACOUSTIC LEARNING LAB <span class="badge quiet">BETA</span></span><div class="source-switch" role="group" aria-label="Audio source"><button data-action="source-recorded" class="${state.source === 'recorded' ? 'selected' : ''}" ${isBusy() ? 'disabled' : ''}>My recordings</button><button data-action="source-synthetic" class="${state.source === 'synthetic' ? 'selected' : ''}" ${isBusy() ? 'disabled' : ''}>${icon('bolt', 13)} Demo</button></div></div>
    ${state.source === 'synthetic' ? `<div class="demo-banner">${icon('bolt', 16)}<span><strong>Synthetic demo.</strong> Generated sounds let you explore the full workflow. These scores do not measure a real keyboard.</span>${btn('Use my keyboard', 'source-recorded', 'text-button')}</div>` : ''}
    ${{ capture: renderCapture, dataset: renderDataset, train: renderTrain, results: renderResults, guide: renderGuide }[state.tab]?.() || renderCapture()}
    <footer class="main-footer"><span><i class="status-dot"></i> Audio never leaves this browser.</span><span>Made for curious ears & open experiments.</span></footer></main></div>`;
  requestAnimationFrame(paint);
}
function renderStats() {
  const clips = allClips(), keys = new Set(clips.map(c => c.label));
  return `<div class="stats-grid">${[
    ['wave', clips.length.toLocaleString(), 'Captured samples', `${(clips.length * CLIP_SIZE / SAMPLE_RATE).toFixed(1)}s of isolated audio`, 'accent'],
    ['keyboard', String(keys.size).padStart(2, '0'), 'Unique keys', keys.size ? [...keys].sort().join(' · ') : 'Your keyboard, one key at a time', 'violet'],
    ['folder', String(sessions().filter(s => s.clips.length).length).padStart(2, '0'), 'Recording sessions', 'Separate sessions. Better science.', 'orange'],
    ['brain', model() ? 'Trained' : 'Untrained', 'Model status', model() ? `${model().labels.length} keys · browser neural network` : 'Your first model starts here', 'blue']
  ].map(([svg, value, label, sub, color]) => `<div class="stat"><div class="stat-top"><span>${label}</span><span class="stat-icon ${color}">${icon(svg, 17)}</span></div><div class="stat-value ${label === 'Model status' ? 'word' : ''}">${value}</div><small>${sub}</small></div>`).join('')}</div>`;
}
function keyboard() {
  const counts = recordingSession ? Object.fromEntries(state.selectedKeys.map(k => [k, recordingSession.clips.filter(c => c.label === k).length])) : Object.fromEntries(state.selectedKeys.map(k => [k, allClips().filter(c => c.label === k).length]));
  return `<div class="keyboard" aria-label="Keyboard key selection">${['QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM'].map((row, i) => `<div class="key-row row-${i}">${[...row].map(key => `<button class="key ${state.selectedKeys.includes(key) ? 'included' : ''} ${recording && activeKey === key ? 'listening' : ''} ${recording && counts[key] >= state.target ? 'complete' : ''}" data-action="key" data-key="${key}" aria-label="${key}, ${state.selectedKeys.includes(key) ? 'selected' : 'not selected'}" aria-pressed="${state.selectedKeys.includes(key)}" ${isBusy() ? 'disabled' : ''}>${key}<small>${state.selectedKeys.includes(key) && counts[key] ? counts[key] : ''}</small></button>`).join('')}</div>`).join('')}<div class="key-row space-row"><span>ctrl</span><span>alt</span><span class="space-key"></span><span>alt</span><span>ctrl</span></div></div>`;
}
function renderCapture() {
  const clips = allClips(), currentCount = recordingSession?.clips.filter(c => c.label === activeKey).length || 0;
  return `${sectionHead('SOUND BECOMES A SIGNAL', 'Every key has a voice.', 'Teach your computer to hear the difference. Start with your own keyboard.', btn('Import audio', 'upload', 'secondary', 'up', state.source === 'synthetic' || isBusy() ? 'disabled' : ''))}
    ${renderStats()}
    <div class="capture-grid"><section class="panel recorder-panel"><div class="panel-heading"><div class="panel-title">${icon('mic', 18)}<h2>Capture studio</h2></div><span class="badge ${recording ? 'recording-badge' : ''}"><i class="status-dot"></i>${recording ? 'RECORDING' : micStarting ? 'CONNECTING' : 'STANDBY'}</span></div>
      <div class="wave-container"><div class="wave-overlay"><span>${recording ? 'LIVE INPUT' : 'INPUT WAVEFORM'}</span><span>${recording ? esc(microphone?.label || 'Microphone') : '16,000 Hz / MONO'}</span></div><canvas id="live-wave" aria-label="Audio waveform"></canvas><div class="wave-labels"><span>−1.0</span><span>${recording ? 'Listening to your microphone' : 'Connect your microphone to see a live signal'}</span><span>+1.0</span></div></div>
      <div class="keyboard-caption"><span>${recording ? `Press <strong>${activeKey}</strong> slowly, then release` : 'Choose the keys you want to teach'}</span><span class="mono">${state.selectedKeys.length} KEYS SELECTED</span></div>${keyboard()}
      <div class="capture-bottom"><div class="capture-instruction"><span class="instruction-key">${recording ? activeKey : icon('headphones', 23)}</span><div><strong>${recording ? `${currentCount} / ${state.target} presses for ${activeKey}` : state.source === 'synthetic' ? 'The demo is ready to explore' : 'Ready when you are'}</strong><small>${recording ? 'Wait at least 0.4 seconds between presses.' : state.source === 'synthetic' ? 'Inspect the samples, then train your first model.' : 'A quiet room. A steady mic. One key at a time.'}</small></div></div>${recording ? btn('Stop & save', 'stop', 'danger', 'stop') : state.source === 'synthetic' ? btn('Explore dataset', 'go-dataset', 'primary', 'arrow') : btn(micStarting ? 'Connecting…' : 'Start recording', 'record', 'primary', 'mic', isBusy() ? 'disabled' : '')}</div>
    </section>
    <aside class="setup-column"><section class="panel setup-panel"><div class="panel-heading"><div class="panel-title">${icon('dots', 18)}<h2>Session setup</h2></div><span class="tiny-label">01 / CAPTURE</span></div><div class="setup-fields"><label>Session name<input id="session-name" maxlength="80" value="Session ${String(sessions().length + 1).padStart(2, '0')}" ${isBusy() ? 'disabled' : ''}></label><label>Key set<select id="key-preset" ${isBusy() ? 'disabled' : ''}><option value="custom">Custom · ${state.selectedKeys.length} keys</option><option value="starter">Starter · A S D F J K</option><option value="home">Home row · A S D F G H J K L</option><option value="all">All letters · A–Z</option></select></label><label>Samples per key<select id="target-count" ${isBusy() ? 'disabled' : ''}>${[10, 25, 50].map(n => `<option value="${n}" ${n === state.target ? 'selected' : ''}>${n} samples ${n === 25 ? '· Recommended' : ''}</option>`).join('')}</select></label><div class="setup-total"><span>Session target</span><strong>${state.target * state.selectedKeys.length}<small> samples</small></strong></div></div></section>
    <div class="tip-card"><span class="tip-icon">${icon('bolt', 19)}</span><div><strong>Same setup. Fresh session.</strong><p>Keep the keyboard, mic, and desk in place. Record at least two separate sessions so your test stays honest.</p></div></div>
    <button class="demo-card" data-action="${state.source === 'synthetic' ? 'go-train' : 'source-synthetic'}" ${isBusy() ? 'disabled' : ''}><span class="demo-icon">${icon('play', 18)}</span><span><strong>${state.source === 'synthetic' ? 'Ready to train?' : 'Just looking around?'}</strong><small>${state.source === 'synthetic' ? 'Turn these sounds into a model' : 'Try the interactive demo'}</small></span>${icon('arrow', 18)}</button></aside></div>
    ${renderSessionLibrary()}
    <div class="flow-strip"><div><span>01</span>${icon('mic', 20)}<strong>Record</strong><small>Capture labeled key presses</small></div>${icon('chevron', 16)}<div><span>02</span>${icon('layers', 20)}<strong>Transform</strong><small>Turn sound into spectrograms</small></div>${icon('chevron', 16)}<div><span>03</span>${icon('brain', 20)}<strong>Learn</strong><small>Train on acoustic patterns</small></div>${icon('chevron', 16)}<div><span>04</span>${icon('chart', 20)}<strong>Evaluate</strong><small>Test on a separate session</small></div></div>`;
}
function renderSessionLibrary() {
  const list = sessions().filter(s => s.clips.length);
  return `<section class="panel library-panel"><div class="panel-heading"><div class="panel-title">${icon('folder', 18)}<h2>Your sessions</h2><span class="count-badge">${list.length}</span></div>${list.length ? btn('View dataset', 'go-dataset', 'text-button', 'arrow') : '<span class="tiny-label">SAVED AUTOMATICALLY</span>'}</div>${list.length ? `<div class="session-list">${list.map((s, i) => `<div class="session-row"><span class="session-symbol">${icon('wave', 18)}</span><div class="session-name"><strong>${esc(s.name)}</strong><small>${s.source === 'synthetic' ? 'Generated audio' : esc(new Date(s.createdAt).toLocaleDateString())} · ${new Set(s.clips.map(c => c.label)).size} keys</small></div><div class="session-distribution">${[...new Set(s.clips.map(c => c.label))].slice(0, 10).map(k => `<span title="${k}: ${s.clips.filter(c => c.label === k).length} samples">${k}<i style="height:${Math.min(24, 4 + s.clips.filter(c => c.label === k).length)}px"></i></span>`).join('')}</div><span class="session-count">${s.clips.length}<small>samples</small></span><button class="icon-button" data-action="view-session" data-id="${esc(s.id)}" aria-label="Inspect ${esc(s.name)}">${icon('arrow', 18)}</button></div>`).join('')}</div>` : `<div class="library-empty"><span class="empty-folder">${icon('layers', 25)}</span><div><strong>Your first session is a few keystrokes away.</strong><p>Record a session, or explore the synthetic demo to see what comes next.</p></div>${btn('Try demo', 'source-synthetic', 'secondary', 'play', isBusy() ? 'disabled' : '')}</div>`}</section>`;
}
function renderDataset() {
  const clips = allClips(), selected = clipById(state.selectedClip) || clips[0];
  const filtered = clips.filter(c => (state.filterKey === 'all' || c.label === state.filterKey) && (state.filterSession === 'all' || c.sessionId === state.filterSession));
  const pages = Math.max(1, Math.ceil(filtered.length / 12)); state.page = Math.min(state.page, pages - 1);
  return `${sectionHead('FROM A CLICK TO A PICTURE', 'See what a key sounds like.', 'Inspect the signal, listen to each clip, and keep your dataset clean.', btn('Export dataset', 'export-dataset', 'secondary', 'down', !clips.length || isBusy() ? 'disabled' : '') + btn('Import dataset', 'import-dataset', 'primary', 'up', isBusy() ? 'disabled' : ''))}
  ${selected ? `<section class="panel inspector"><div class="panel-heading"><div class="panel-title"><span class="sample-key">${selected.label}</span><h2>Acoustic fingerprint</h2><span class="badge quiet">${esc(selected.sessionName)}</span></div><div class="inline-actions">${btn('Listen', 'play', 'secondary small', 'play', `data-id="${esc(selected.id)}"`)}${btn('WAV', 'download-clip', 'secondary small', 'down', `data-id="${esc(selected.id)}"`)}</div></div><div class="inspector-visuals"><div><div class="chart-label">WAVEFORM <span>256 ms</span></div><canvas id="inspect-wave" aria-label="Selected sample waveform"></canvas></div><div><div class="chart-label">MEL SPECTROGRAM <span>80 Hz – 7.6 kHz</span></div><canvas id="inspect-spectrum" aria-label="Selected sample mel spectrogram"></canvas><div class="spectrum-legend"><span>0 ms</span><span>Time →</span><span>256 ms</span></div></div></div><div class="inspector-meta"><span>${icon('check', 13)} ${CLIP_SIZE.toLocaleString()} samples · 16 kHz mono</span><span>Peak ${audioQuality(selected.samples).peak.toFixed(3)} · ${audioQuality(selected.samples).clipped > .01 ? 'Clipping detected' : 'No digital clipping'}</span></div></section>` : ''}
  <section class="panel dataset-panel"><div class="panel-heading"><div class="panel-title"><h2>Sample library</h2><span class="count-badge">${filtered.length}</span></div><div class="inline-actions"><select id="filter-session" aria-label="Filter by session"><option value="all">All sessions</option>${sessions().map(s => `<option value="${esc(s.id)}" ${s.id === state.filterSession ? 'selected' : ''}>${esc(s.name)}</option>`).join('')}</select><select id="filter-key" aria-label="Filter by key"><option value="all">All keys</option>${[...new Set(clips.map(c => c.label))].sort().map(k => `<option ${k === state.filterKey ? 'selected' : ''}>${k}</option>`).join('')}</select></div></div>
  ${filtered.length ? `<div class="sample-grid">${filtered.slice(state.page * 12, state.page * 12 + 12).map(c => `<article class="sample-card ${selected?.id === c.id ? 'selected' : ''}"><button class="sample-main" data-action="inspect" data-id="${esc(c.id)}" aria-label="Inspect key ${c.label}"><span class="sample-card-heading"><span class="sample-key">${c.label}</span><span>${esc(c.sessionName)}</span></span><canvas data-spectrum="${esc(c.id)}" aria-hidden="true"></canvas><span class="sample-card-footer">256 ms <span>16 kHz</span></span></button><div class="sample-actions"><button data-action="play" data-id="${esc(c.id)}" aria-label="Play sample">${icon('play', 14)} Listen</button><button data-action="delete-clip" data-id="${esc(c.id)}" aria-label="Delete sample" ${isBusy() ? 'disabled' : ''}>${icon('trash', 14)}</button></div></article>`).join('')}</div><div class="pagination"><span>${state.page * 12 + 1}–${Math.min(filtered.length, (state.page + 1) * 12)} of ${filtered.length} samples</span><div>${btn('Previous', 'page-prev', 'secondary small', '', !state.page ? 'disabled' : '')}<span>${state.page + 1} / ${pages}</span>${btn('Next', 'page-next', 'secondary small', '', state.page >= pages - 1 ? 'disabled' : '')}</div></div>` : empty('No samples here yet.', 'Start with a recording session or load the demo.', btn('Go to capture', 'go-capture', 'primary', 'mic'))}</section>`;
}
function splitError() { try { prepareSplit(sessions(), selectedHeldOut()); return ''; } catch (error) { return error.message; } }
function selectedHeldOut() { return sessions().find(s => s.id === state.heldOut[state.source])?.id || sessions().filter(s => s.clips.length).at(-1)?.id; }
function renderTrain() {
  const testId = selectedHeldOut(), test = sessions().find(s => s.id === testId), trainCount = sessions().filter(s => s.id !== testId).reduce((a, s) => a + s.clips.length, 0), issue = splitError();
  const curve = training ? history : model()?.history || [];
  return `${sectionHead('LET THE PATTERNS SPEAK', 'Teach it to listen.', 'A small neural network learns from the spectrograms. One whole session stays unseen.', btn('Import model', 'import-model', 'secondary', 'up', isBusy() ? 'disabled' : ''))}
  <div class="train-grid"><section class="panel"><div class="panel-heading"><div class="panel-title">${icon('brain', 18)}<h2>Training setup</h2></div><span class="badge quiet">LOCAL CPU</span></div><div class="setup-fields"><label>Held-out test session<select id="held-out" ${isBusy() ? 'disabled' : ''}>${!sessions().length ? '<option>No sessions available</option>' : sessions().map(s => `<option value="${esc(s.id)}" ${s.id === testId ? 'selected' : ''}>${esc(s.name)} · ${s.clips.length} clips</option>`).join('')}</select><small>This entire session is excluded from training.</small></label><div class="split-bar"><span style="flex:${Math.max(1, trainCount)}"></span><span style="flex:${Math.max(1, test?.clips.length || 0)}"></span></div><div class="split-legend"><span><i></i> ${trainCount} train</span><span><i></i> ${test?.clips.length || 0} test</span></div><label>Training epochs<select id="epochs" ${isBusy() ? 'disabled' : ''}>${[15, 30, 45, 75, 100].map(n => `<option ${n === state.epochs ? 'selected' : ''}>${n}</option>`).join('')}</select></label><label class="toggle-label"><span>Data augmentation<small>Small time shifts & frequency masks</small></span><input id="augmentation" type="checkbox" ${state.augmentation ? 'checked' : ''} ${isBusy() ? 'disabled' : ''}><span class="toggle"></span></label><div class="model-spec"><span>Architecture<strong>512 → 32 ReLU → ${Math.max(2, new Set(allClips().map(c => c.label)).size)} softmax</strong></span><span>Input<strong>32 mel bands × 16 time bins</strong></span><span>Optimizer<strong>Adam · 0.0015 learning rate</strong></span></div>
  ${training ? btn('Cancel training', 'cancel-training', 'danger full', 'stop') : btn('Train model', 'train', 'primary full', 'bolt', issue || isBusy() ? 'disabled' : '')}${issue ? `<div class="inline-note">${icon('info', 16)}<span>${esc(issue)}</span></div>` : `<p class="field-help">Fixed epochs. Training-only normalization and augmentation. Test session scored once after training.</p>`}</div></section>
  <div class="training-right"><section class="panel learning-panel"><div class="panel-heading"><div class="panel-title"><h2>Learning curve</h2></div><span class="chart-legend"><i></i> Training accuracy</span></div><div class="training-figures"><div><small>EPOCH</small><strong id="epoch-value">${curve.length}<span> / ${training ? state.epochs : model()?.epochs || state.epochs}</span></strong></div><div><small>TRAIN ACCURACY</small><strong id="accuracy-value">${curve.length ? percent(curve.at(-1).accuracy) : '—'}</strong></div><div><small>TRAIN LOSS</small><strong id="loss-value">${curve.length ? curve.at(-1).loss.toFixed(3) : '—'}</strong></div></div><canvas id="learning-chart" aria-label="Training accuracy by epoch"></canvas><div class="training-progress"><div><span id="train-status">${training ? esc(trainStatus) : model() ? 'Training complete. Your held-out results are ready.' : 'Ready to learn when you are.'}</span><span id="train-percent">${training ? Math.round(trainProgress * 100) + '%' : ''}</span></div><div class="progress-track"><span id="train-progress-fill" style="width:${training ? trainProgress * 100 : model() ? 100 : 0}%"></span></div></div></section>
  <div class="method-note">${icon('lock', 22)}<div><strong>A clean split is the whole point.</strong><p>Clips from your test session never enter training, augmentation, or normalization. The score tells you how well the model handles a fresh session.</p></div></div>${model() ? `<div class="result-ready"><span>${icon('check', 20)} Your ${state.source === 'synthetic' ? 'demo ' : ''}model is ready.</span>${btn('See results', 'go-results', 'primary', 'arrow')}</div>` : ''}</div></div>`;
}
function renderResults() {
  const r = report(), m = model();
  if (!m) return `${sectionHead('THE MOMENT OF TRUTH', 'How well did it listen?', 'Measured on presses from a separate recording session.')}<section class="panel">${empty('Your results start with a model.', 'Collect at least two sessions, then train your first acoustic classifier.', btn('Set up training', 'go-train', 'primary', 'brain'))}</section>`;
  return `${sectionHead('THE MOMENT OF TRUTH', 'Put its ears to the test.', r ? `Held-out evaluation · ${esc(r.testSessionName)} · ${r.total} unseen presses` : 'Model imported. Try a single press, or evaluate on a fresh recording session.', btn('Export model', 'export-model', 'secondary', 'down') + (r ? btn('Export report', 'export-report', 'secondary', 'down') : ''))}
  ${r ? `<div class="results-stats"><div class="result-score"><span>HELD-OUT ACCURACY</span><strong>${percent(r.accuracy)}</strong><small>${r.correct} of ${r.total} keys identified correctly</small></div><div><span>95% ACCURACY INTERVAL</span><strong>${percent(r.interval[0])}–${percent(r.interval[1])}</strong><small>Wilson interval · small samples mean uncertainty</small></div><div><span>UNIFORM GUESS BASELINE</span><strong>${percent(1 / m.labels.length)}</strong><small>Across ${m.labels.length} known keys</small></div></div>` : `<div class="inline-note spaced">${icon('info', 18)} Imported models do not include a trusted evaluation report. Evaluate a new session to measure performance here.</div>`}
  <div class="results-grid"><section class="panel confusion-panel"><div class="panel-heading"><div class="panel-title">${icon('grid', 18)}<h2>Confusion matrix</h2></div><span class="tiny-label">ACTUAL ↓ / PREDICTED →</span></div>${r ? `<div class="matrix-scroll"><table class="confusion-table"><thead><tr><th scope="col"></th>${m.labels.map(k => `<th scope="col">${k}</th>`).join('')}</tr></thead><tbody>${r.matrix.map((row, i) => `<tr><th scope="row">${m.labels[i]}</th>${row.map((n, j) => `<td title="Actual ${m.labels[i]}, predicted ${m.labels[j]}: ${n}" style="--intensity:${n ? .12 + .65 * n / Math.max(1, r.perKey[i].total) : .025}" class="${i === j ? 'diagonal' : ''}">${n || '·'}</td>`).join('')}</tr>`).join('')}</tbody></table></div><div class="matrix-footer"><span><i class="matrix-swatch"></i> Correct prediction</span><span>Darker cells = fewer presses</span></div>` : empty('No evaluation yet.', 'Choose a recorded session below to compute a fresh score.')}</section>
  <div><section class="panel prediction-panel"><div class="panel-heading"><div class="panel-title">${icon('headphones', 18)}<h2>Try a single press</h2></div></div><div class="prediction-body"><p>${state.source === 'synthetic' ? 'Run a generated held-out clip through the demo model.' : 'Record one isolated key press. The model gets only the sound.'}</p><div class="prediction-key">${predicting ? '<span class="listening-dots">•••</span>' : prediction ? prediction[0].label : '?'}</div>${prediction ? `<div class="prediction-bars">${prediction.slice(0, 3).map(p => `<div><strong>${p.label}</strong><span><i style="width:${p.probability * 100}%"></i></span><small>${percent(p.probability)}</small></div>`).join('')}</div><small class="field-help">Model probabilities are uncalibrated; this is not a guarantee.</small>` : `<span class="field-help">${predicting ? 'Listening for 3 seconds… press a key now.' : `${m.labels.join(' · ')} · known keys`}</span>`}<div class="prediction-buttons">${btn(predicting ? 'Listening…' : state.source === 'synthetic' ? 'Predict a demo sample' : 'Listen for 3 seconds', 'predict', 'primary full', 'mic', isBusy() ? 'disabled' : '')}${state.source === 'recorded' ? btn('Predict from audio file', 'predict-file', 'secondary full', 'up', isBusy() ? 'disabled' : '') : ''}</div></div></section></div></div>
  <section class="panel fresh-evaluation"><div><h2>Evaluate a fresh session</h2><p>The app rejects training session IDs and audio copied from training.</p></div><div class="inline-actions"><select id="evaluate-session" aria-label="Evaluation session">${sessions().filter(s => !m.trainSessionIds.includes(s.id)).map(s => `<option value="${esc(s.id)}">${esc(s.name)} · ${s.clips.length} clips</option>`).join('') || '<option value="">No unused sessions</option>'}</select>${btn('Evaluate', 'evaluate', 'secondary', 'chart', isBusy() ? 'disabled' : '')}</div></section>
  <div class="method-note">${icon('info', 22)}<div><strong>${state.source === 'synthetic' ? 'Synthetic sounds are deliberately easy to separate.' : 'This score belongs to this experiment.'}</strong><p>${state.source === 'synthetic' ? 'A high demo score verifies the workflow. It says nothing about how well your real keyboard can be recognized.' : 'Keyboard, typist, desk, and microphone all affect the sound. Test new sessions under different conditions to explore the limits. Repeatedly tuning against this test set makes its score less reliable.'}</p></div></div>`;
}
function renderGuide() {
  return `${sectionHead('A LITTLE SCIENCE. A LOT OF CURIOSITY.', 'A keyboard is a tiny instrument.', 'Different keys excite different vibrations. We turn those vibrations into patterns a model can learn.')}
  <div class="guide-hero"><div><span class="eyebrow">THE IDEA IN 30 SECONDS</span><h2>Sound → picture → prediction.</h2><p>Record a labeled press, turn the sound into a mel spectrogram, and let a neural network learn the difference between keys. Then test it on a session it has never heard.</p>${btn('Start an experiment', 'go-capture', 'primary', 'arrow')}</div><div class="guide-graphic"><span class="large-key">A</span>${icon('arrow', 24)}<canvas id="guide-spectrum" aria-label="Illustrative synthetic key spectrogram"></canvas>${icon('arrow', 24)}<span class="large-prediction">A<small>PREDICTION</small></span></div></div>
  <div class="guide-cards">${[
    ['01', 'mic', 'Capture clean examples', 'Start with 4–6 keys. Press each 25 times, slowly and separately. Keep the microphone fixed. The studio uses key events only to label the recording; the model sees audio features only.'],
    ['02', 'layers', 'Inspect the fingerprint', 'Each 256 ms clip becomes a spectrogram: time runs left to right, mel frequency runs bottom to top, and brighter colors mean more energy. Listen for bad cuts or extra sounds and remove those samples.'],
    ['03', 'brain', 'Learn from the pictures', 'This lab trains a compact 512 → 32 → key-count neural network from scratch with Adam. Small time shifts and frequency masks augment training clips. Feature statistics come only from the training sessions.'],
    ['04', 'chart', 'Keep the test honest', 'Make a second recording session with the same keys. Hold it out in full. Train with fixed epochs, then measure accuracy and the confusion matrix. Use more sessions to check whether the model generalizes.']
  ].map(([n, svg, title, text]) => `<article class="panel guide-card"><div><span class="guide-number">${n}</span>${icon(svg, 23)}</div><h3>${title}</h3><p>${text}</p></article>`).join('')}</div>
  <div class="guide-bottom"><section class="panel prose"><span class="eyebrow">INSPIRED BY THE RESEARCH</span><h2>A small lab, with honest limits.</h2><p>Harrison, Toreini, and Mehrnezhad reported 95% single-key accuracy for smartphone-recorded audio and 93% for Zoom-recorded audio in their 2023 study. Their system used CoAtNet and a different experimental protocol.</p><p>This project implements the core learning workflow with a much smaller browser model. It is not a reproduction of their architecture or benchmark, and it makes no accuracy promise. The synthetic demo uses generated resonant clicks, not real keyboard recordings.</p><a href="https://arxiv.org/abs/2308.01074" target="_blank" rel="noreferrer">Read the original paper ${icon('arrow', 16)}</a></section><section class="panel prose"><span class="eyebrow">A BETTER EXPERIMENT</span><h2>Change one thing at a time.</h2><p>Try a new mic distance, a different desk, or another day’s recording. Watch which keys become confused. A low score is useful evidence, too.</p><p>Browser audio processing may still vary by hardware even though echo cancellation, noise suppression, and automatic gain are requested off. Training and recordings live in this browser’s storage; export a backup before clearing site data.</p><p>Use your own keyboard and intentional recordings. Password-manager autofill avoids producing password keystrokes.</p></section></div>`;
}

function paint() {
  if (state.tab === 'capture') {
    const tick = () => {
      if (recording && microphone?.analyser) {
        const data = new Float32Array(microphone.analyser.fftSize); microphone.analyser.getFloatTimeDomainData(data); drawWave($('#live-wave'), data);
        waveFrame = requestAnimationFrame(tick);
      } else drawWave($('#live-wave'), null);
    }; tick();
  }
  if (state.tab === 'dataset') {
    const clip = clipById(state.selectedClip) || allClips()[0];
    if (clip) { drawWave($('#inspect-wave'), clip.samples); drawSpectrum($('#inspect-spectrum'), clip.samples); }
    document.querySelectorAll('[data-spectrum]').forEach(canvas => drawSpectrum(canvas, clipById(canvas.dataset.spectrum)?.samples));
  }
  if (state.tab === 'train') drawLearning($('#learning-chart'), training ? history : model()?.history || []);
  if (state.tab === 'guide') drawSpectrum($('#guide-spectrum'), syntheticPress(0));
}

async function navigate(tab) {
  if (recording) await stopCapture();
  if (micStarting || predicting) return;
  state.tab = ['capture', 'dataset', 'train', 'results', 'guide'].includes(tab) ? tab : 'capture';
  if (location.hash !== `#${state.tab}`) historyReplaceHash(state.tab);
  render(); persist(); window.scrollTo({ top: 0, behavior: 'instant' });
}
function historyReplaceHash(tab) { window.history.replaceState(null, '', `#${tab}`); }
async function setSource(source) {
  if (isBusy()) return;
  state.source = source; state.filterKey = 'all'; state.filterSession = 'all'; state.selectedClip = null; state.page = 0; prediction = null;
  if (source === 'synthetic' && !sessions().length) { state.sessions.push(...createDemo()); toast('Demo loaded: 324 generated clicks across 3 sessions.'); }
  render(); persist();
}
async function startCapture() {
  if (isBusy() || state.source !== 'recorded') return;
  if (state.selectedKeys.length < 2) return toast('Choose at least two keys on the keyboard.', true);
  const name = $('#session-name')?.value.trim() || `Session ${sessions().length + 1}`;
  micStarting = true; render();
  const mic = new Microphone();
  try {
    await mic.start();
    microphone = mic; recording = true; micStarting = false; captureGeneration++; pendingClip = false; lastPress = 0;
    recordingSession = { id: uid(), name, source: 'recorded', createdAt: new Date().toISOString(), clips: [] };
    state.sessions.push(recordingSession); activeKey = state.selectedKeys[0];
    render(); toast(`Recording started. Press ${activeKey} ${state.target} times, slowly.`);
  } catch (error) { micStarting = false; toast(error.message, true); render(); }
}
async function stopCapture() {
  if (!recording) return;
  recording = false; captureGeneration++;
  await microphone?.stop();
  microphone = null; pendingClip = false;
  if (recordingSession && !recordingSession.clips.length) state.sessions = state.sessions.filter(s => s.id !== recordingSession.id);
  const total = recordingSession?.clips.length || 0;
  recordingSession = null; persist(); render(); toast(`${total} samples saved. Microphone disconnected.`);
}
document.addEventListener('keydown', event => {
  if (!recording || !microphone?.active || event.repeat || event.metaKey || event.ctrlKey || event.altKey || event.target.closest('input, select, textarea, button') || $('#dialog').open) return;
  const key = event.key.toUpperCase();
  if (key !== activeKey || pendingClip || performance.now() - lastPress < 400) return;
  event.preventDefault(); pendingClip = true; lastPress = performance.now();
  const generation = captureGeneration, time = microphone.context.currentTime;
  setTimeout(async () => {
    if (!recording || generation !== captureGeneration) return;
    try {
      const samples = microphone.clipAt(time), quality = audioQuality(samples);
      if (quality.peak < .004) return toast('That press was too quiet. Move the mic closer and try again.', true);
      if (quality.clipped > .01) return toast('That sample clipped. Reduce microphone input gain and try again.', true);
      invalidate();
      recordingSession.clips.push({ id: uid(), label: key, samples, createdAt: new Date().toISOString() });
      const count = recordingSession.clips.filter(c => c.label === key).length;
      if (count >= state.target) {
        const next = state.selectedKeys.find(k => recordingSession.clips.filter(c => c.label === k).length < state.target);
        if (!next) return await stopCapture();
        activeKey = next; toast(`Great. Now press ${next} ${state.target} times.`);
      }
      persist(); render();
    } catch (error) { toast(error.message, true); }
    finally { pendingClip = false; }
  }, 350);
});

function startTraining() {
  if (isBusy()) return;
  const issue = splitError(); if (issue) return toast(issue, true);
  training = true; history = []; trainProgress = 0; trainStatus = 'Preparing mel spectrograms…';
  worker = new Worker(new URL('./train-worker.js', import.meta.url), { type: 'module' });
  const source = state.source;
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') {
      if (data.phase === 'training') history.push({ epoch: data.epoch, accuracy: data.accuracy, loss: data.loss });
      trainProgress = data.phase === 'features' ? data.progress * .1 : .1 + data.progress * .9;
      trainStatus = data.phase === 'features' ? 'Preparing mel spectrograms…' : data.phase === 'evaluating' ? 'Scoring the held-out session…' : `Learning acoustic patterns · epoch ${data.epoch} of ${state.epochs}`;
      if (state.tab === 'train') {
        $('#train-status').textContent = trainStatus; $('#train-percent').textContent = `${Math.round(trainProgress * 100)}%`; $('#train-progress-fill').style.width = `${trainProgress * 100}%`;
        if (data.phase === 'training') { $('#epoch-value').innerHTML = `${data.epoch}<span> / ${state.epochs}</span>`; $('#accuracy-value').textContent = percent(data.accuracy); $('#loss-value').textContent = data.loss.toFixed(3); drawLearning($('#learning-chart'), history); }
      }
    } else if (data.type === 'done') {
      state.models[source] = data.model; state.reports[source] = data.report; training = false; worker.terminate(); worker = null;
      prediction = null; persist(); render(); toast('Training complete. Your held-out results are ready.');
    } else if (data.type === 'error') finishError(data.message);
  };
  worker.onerror = event => finishError(event.message || 'Training worker failed. Try reloading the page.');
  function finishError(message) { training = false; worker?.terminate(); worker = null; render(); toast(message, true); }
  worker.postMessage({ sessions: sessions(), options: { testSessionId: selectedHeldOut(), epochs: state.epochs, augmentation: state.augmentation } });
  render();
}
async function predictSingle() {
  if (isBusy() || !model()) return;
  if (state.source === 'synthetic') {
    const test = sessions().find(s => s.id === report()?.testSessionId) || sessions().at(-1);
    const clip = test?.clips[Math.floor(Math.random() * test.clips.length)];
    if (!clip) return toast('Load a demo dataset first.', true);
    prediction = predict(model(), clip.samples); render(); toast(`Demo prediction complete. The clip’s actual label is ${clip.label}.`); return;
  }
  predicting = true; prediction = null; render();
  const mic = new Microphone();
  try {
    await mic.start(); microphone = mic; render();
    const start = mic.context.currentTime;
    await new Promise(resolve => setTimeout(resolve, 3150));
    if (!mic.active) throw new Error('Recording stopped because the page lost visibility. Try again.');
    const samples = mic.range(start, start + 3);
    const clips = segmentOnsets(samples);
    if (clips.length !== 1) throw new Error(`Heard ${clips.length} isolated sounds. Try again with exactly one key press in a quiet room.`);
    prediction = predict(model(), clips[0].samples);
  } catch (error) { toast(error.message, true); }
  finally { await mic.stop(); microphone = null; predicting = false; render(); }
}
function download(name, contents, type = 'application/json') {
  const blob = new Blob([typeof contents === 'string' || contents instanceof ArrayBuffer ? contents : JSON.stringify(contents)], { type });
  const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function evaluateSession() {
  const m = model(), session = sessions().find(s => s.id === $('#evaluate-session')?.value);
  if (!session?.clips.length) return toast('Choose a nonempty unused session.', true);
  if (m.trainSessionIds.includes(session.id)) return toast('That session was used for training.', true);
  const signatures = new Set(m.trainSampleSignatures);
  if (session.clips.some(c => signatures.has(audioSignature(c.samples)))) return toast('This session contains training audio. Record new presses.', true);
  for (const label of m.labels) if (!session.clips.some(c => c.label === label)) return toast(`This session is missing key ${label}.`, true);
  state.reports[state.source] = { ...evaluate(m, session.clips), testSessionId: session.id, testSessionName: session.name, source: state.source, labels: m.labels };
  persist(); render(); toast('Fresh session evaluated.');
}
function openUpload() {
  if (isBusy() || state.source !== 'recorded') return;
  uploadData = null;
  $('#dialog').innerHTML = `<div class="dialog-heading"><div><span class="eyebrow">BRING YOUR OWN RECORDING</span><h2>Import key audio</h2></div><button class="icon-button" data-action="close-dialog" aria-label="Close">${icon('close')}</button></div><p class="dialog-intro">One file, one key. Record slow, separate presses of the same key. Review the detected clips before adding them.</p><div class="upload-fields"><label>Recording session<select id="upload-session"><option value="new">New recording session</option>${sessions().map(s => `<option value="${esc(s.id)}">${esc(s.name)}</option>`).join('')}</select></label><label>New session name<input id="upload-name" value="Imported session ${sessions().length + 1}" maxlength="80"></label><label>Key in this recording<select id="upload-key">${[...'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'].map(k => `<option>${k}</option>`).join('')}</select></label><label>Detection threshold<input id="upload-threshold" type="number" value="0.012" min="0.002" max="0.3" step="0.002"><small>Raise this if room noise is detected as a press.</small></label></div><label class="file-drop">${icon('up', 24)}<strong>Choose an audio recording</strong><small>WAV recommended · up to 30 MB / 5 minutes</small><input id="audio-upload" type="file" accept="audio/*"></label><div id="upload-preview"></div><div class="dialog-footer"><small>Keep recordings from one sitting in the same session.</small>${btn('Add selected clips', 'accept-upload', 'primary', 'plus', 'disabled id="accept-upload"')}</div>`;
  $('#dialog').showModal();
}
function previewUpload() {
  if (!uploadData) return;
  const threshold = Number($('#upload-threshold').value);
  if (!Number.isFinite(threshold) || threshold < .002 || threshold > .3) return toast('Choose a threshold from 0.002 to 0.3.', true);
  uploadData.clips = segmentOnsets(uploadData.samples, threshold).slice(0, 500);
  $('#upload-preview').innerHTML = `<div class="upload-summary"><strong>${uploadData.clips.length} clips detected</strong><small>Uncheck any extra sounds. Listen before keeping.</small></div><div class="detected-clips">${uploadData.clips.map((c, i) => `<div><label><input type="checkbox" data-candidate="${i}" checked> Clip ${i + 1}<small>${c.at.toFixed(2)}s</small></label><button class="icon-button" data-action="play-candidate" data-index="${i}" aria-label="Listen to clip ${i + 1}">${icon('play', 15)}</button></div>`).join('') || '<p>No clicks found. Lower the threshold or use a louder recording.</p>'}</div>`;
  $('#accept-upload').disabled = !uploadData.clips.length;
}
function acceptUpload() {
  if (!uploadData || isBusy()) return;
  const label = $('#upload-key').value, sessionId = $('#upload-session').value;
  const kept = [...document.querySelectorAll('[data-candidate]:checked')].map(el => uploadData.clips[Number(el.dataset.candidate)]).filter(c => audioQuality(c.samples).peak >= .004 && audioQuality(c.samples).clipped <= .01);
  if (!kept.length) return toast('Select at least one audible, unclipped sample.', true);
  let session = sessions().find(s => s.id === sessionId);
  if (!session) { session = { id: uid(), name: $('#upload-name').value.trim() || 'Imported session', source: 'recorded', createdAt: new Date().toISOString(), clips: [] }; state.sessions.push(session); }
  for (const clip of kept) session.clips.push({ id: uid(), label, samples: clip.samples, createdAt: new Date().toISOString() });
  invalidate(); $('#dialog').close(); uploadData = null; persist(); render(); toast(`${kept.length} ${label} clips added to ${session.name}.`);
}

document.addEventListener('click', async event => {
  const tab = event.target.closest('[data-tab]'); if (tab) return navigate(tab.dataset.tab);
  const button = event.target.closest('[data-action]'); if (!button || button.disabled) return;
  const action = button.dataset.action;
  try {
    if (action.startsWith('go-')) return navigate(action.slice(3));
    if (action.startsWith('source-')) return setSource(action.slice(7));
    if (action === 'key' && !isBusy()) { const key = button.dataset.key; state.selectedKeys = state.selectedKeys.includes(key) ? state.selectedKeys.filter(k => k !== key) : [...state.selectedKeys, key]; persist(); render(); }
    else if (action === 'record') { button.blur(); await startCapture(); }
    else if (action === 'stop') await stopCapture();
    else if (action === 'train') startTraining();
    else if (action === 'cancel-training') { worker?.terminate(); worker = null; training = false; render(); toast('Training canceled. Any previous model was kept.'); }
    else if (action === 'view-session') { state.filterSession = button.dataset.id; state.filterKey = 'all'; state.page = 0; await navigate('dataset'); }
    else if (action === 'inspect') { state.selectedClip = button.dataset.id; render(); }
    else if (action === 'play') await playSamples(clipById(button.dataset.id).samples);
    else if (action === 'download-clip') { const clip = clipById(button.dataset.id); download(`keytone-${clip.label}.wav`, encodeWav(clip.samples), 'audio/wav'); }
    else if (action === 'delete-clip' && !isBusy()) {
      const clip = clipById(button.dataset.id); if (!clip) return;
      const session = sessions().find(s => s.id === clip.sessionId); session.clips = session.clips.filter(c => c.id !== clip.id);
      invalidate(); persist(); render();
      // Deletion is reversible without another prompt.
      $('#toast').innerHTML = `Sample removed. <button id="undo-delete">Undo</button>`; $('#toast').className = 'visible';
      $('#undo-delete').onclick = () => { if (isBusy()) return toast('Finish the current operation before restoring this clip.', true); const { sessionId, sessionName, ...raw } = clip; session.clips.push(raw); delete state.models[session.source]; delete state.reports[session.source]; prediction = null; persist(); render(); toast('Sample restored.'); };
      clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').className = '', 10000);
    }
    else if (action === 'page-prev') { state.page--; render(); }
    else if (action === 'page-next') { state.page++; render(); }
    else if (action === 'export-dataset') download(`keytone-${state.source}-dataset.json`, exportDataset(sessions()));
    else if (action === 'export-model') download(`keytone-${state.source}-model.json`, serializeModel(model()));
    else if (action === 'export-report') download(`keytone-${state.source}-report.json`, { kind: 'keytone-report', version: 1, ...report(), model: { architecture: '512-32-ReLU-softmax', epochs: model().epochs, trainSessionIds: model().trainSessionIds } });
    else if (action === 'import-dataset' || action === 'import-model') { importMode = action === 'import-model' ? 'model' : 'dataset'; $('#json-import').value = ''; $('#json-import').click(); }
    else if (action === 'predict') await predictSingle();
    else if (action === 'evaluate' && !isBusy()) evaluateSession();
    else if (action === 'upload') openUpload();
    else if (action === 'close-dialog') { $('#dialog').close(); uploadData = null; }
    else if (action === 'play-candidate') await playSamples(uploadData.clips[Number(button.dataset.index)].samples);
    else if (action === 'accept-upload') acceptUpload();
    else if (action === 'predict-file' && !isBusy()) {
      const input = document.createElement('input'); input.type = 'file'; input.accept = 'audio/*';
      input.onchange = async () => {
        if (!input.files[0]) return;
        try { const samples = await decodeFile(input.files[0]), clips = segmentOnsets(samples); if (clips.length !== 1) throw new Error(`Detected ${clips.length} sounds. Use a file with exactly one press.`); prediction = predict(model(), clips[0].samples); render(); } catch (error) { toast(error.message, true); }
      }; input.click();
    }
  } catch (error) { toast(error.message, true); }
});
document.addEventListener('change', async event => {
  const el = event.target;
  try {
    if (el.id === 'json-import' && el.files[0]) {
      if (isBusy()) return;
      if (el.files[0].size > 100 * 1024 * 1024) throw new Error('Import a JSON file smaller than 100 MB.');
      const raw = JSON.parse(await el.files[0].text());
      if (importMode === 'model') { const imported = validateModel(raw); state.source = imported.source; state.models[state.source] = imported; delete state.reports[state.source]; prediction = null; state.tab = 'results'; }
      else {
        const imported = validateDataset(raw), existing = new Set(state.sessions.map(s => s.id)), clipIds = new Set(state.sessions.flatMap(s => s.clips.map(c => c.id)));
        const additions = imported.filter(s => !existing.has(s.id));
        if (!additions.length) throw new Error('Those session IDs already exist. Your saved data was kept.');
        if (additions.some(s => s.clips.some(c => clipIds.has(c.id)))) throw new Error('Imported clips duplicate existing IDs. Your saved data was kept.');
        state.sessions.push(...additions); state.source = additions[0].source;
        for (const source of new Set(additions.map(s => s.source))) { delete state.models[source]; delete state.reports[source]; }
        state.tab = 'dataset'; state.filterKey = 'all'; state.filterSession = 'all'; state.selectedClip = null;
      }
      persist(); render(); toast(`${importMode === 'model' ? 'Model' : 'Dataset'} imported successfully.`);
    } else if (el.id === 'key-preset') { state.selectedKeys = { starter: [...'ASDFJK'], home: [...'ASDFGHJKL'], all: [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'] }[el.value] || state.selectedKeys; persist(); render(); }
    else if (el.id === 'target-count') { state.target = Number(el.value); persist(); render(); }
    else if (el.id === 'filter-key') { state.filterKey = el.value; state.page = 0; render(); }
    else if (el.id === 'filter-session') { state.filterSession = el.value; state.page = 0; render(); }
    else if (el.id === 'held-out') { state.heldOut[state.source] = el.value; persist(); render(); }
    else if (el.id === 'epochs') { state.epochs = Number(el.value); persist(); }
    else if (el.id === 'augmentation') { state.augmentation = el.checked; persist(); }
    else if (el.id === 'audio-upload' && el.files[0]) { $('#upload-preview').innerHTML = '<p class="field-help">Decoding audio locally…</p>'; uploadData = { samples: await decodeFile(el.files[0]), clips: [] }; previewUpload(); }
    else if (el.id === 'upload-threshold') previewUpload();
    else if (el.id === 'upload-session') $('#upload-name').disabled = el.value !== 'new';
  } catch (error) { toast(error instanceof SyntaxError ? 'That file is not valid JSON.' : error.message, true); if (el.id === 'audio-upload') { uploadData = null; $('#upload-preview').textContent = error.message; } }
});
window.addEventListener('hashchange', () => navigate(location.hash.slice(1)));
window.addEventListener('resize', () => { if (!recording) paint(); });
document.addEventListener('visibilitychange', () => { if (document.hidden && recording) stopCapture(); if (document.hidden && predicting) microphone?.stop(); });
window.addEventListener('beforeunload', event => { if (isBusy()) { event.preventDefault(); event.returnValue = ''; } });

try {
  const saved = await loadState();
  if (saved) {
    for (const key of ['sessions', 'source', 'selectedKeys', 'target', 'models', 'reports', 'heldOut', 'epochs', 'augmentation']) if (saved[key] !== undefined) state[key] = saved[key];
  }
} catch { toast('Persistent storage is unavailable. Export your work before leaving this page.', true); }
if (!['recorded', 'synthetic'].includes(state.source)) state.source = 'recorded';
const initialTab = location.hash.slice(1); if (['capture', 'dataset', 'train', 'results', 'guide'].includes(initialTab)) state.tab = initialTab;
render();
