import { FEATURE_SIZE, TIME_BINS, MEL_BANDS, extractFeatures } from './dsp.js';

export function seededRandom(seed = 42) {
  return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
}
const argmax = values => values.indexOf(Math.max(...values));

export function prepareSplit(sessions, testSessionId) {
  const testSession = sessions.find(s => s.id === testSessionId);
  if (!testSession?.clips.length) throw new Error('Choose a nonempty held-out session.');
  const trainSessions = sessions.filter(s => s.id !== testSessionId && s.clips.length);
  if (!trainSessions.length) throw new Error('Record at least two separate sessions before training.');
  if (new Set(sessions.map(s => s.source)).size !== 1) throw new Error('Synthetic and recorded audio must stay in separate experiments.');
  const trainClips = trainSessions.flatMap(s => s.clips), testClips = testSession.clips;
  const labels = [...new Set(trainClips.map(c => c.label))].sort();
  if (labels.length < 2) throw new Error('Train on at least two different keys.');
  if (testClips.some(c => !labels.includes(c.label))) throw new Error('A held-out key is missing from the training sessions.');
  for (const label of labels) {
    if (trainClips.filter(c => c.label === label).length < 5) throw new Error(`Add at least 5 training clips for ${label}.`);
    if (testClips.filter(c => c.label === label).length < 2) throw new Error(`Add at least 2 held-out clips for ${label}.`);
  }
  const trainIds = new Set(trainClips.map(c => c.id));
  if (testClips.some(c => trainIds.has(c.id))) throw new Error('A clip appears in both training and test sessions.');
  // Also reject copied audio with new IDs. Session IDs alone cannot detect this leakage.
  const signatures = new Set(trainClips.map(c => audioSignature(c.samples)));
  if (testClips.some(c => signatures.has(audioSignature(c.samples)))) throw new Error('Identical audio appears in both training and test sessions. Record new presses.');
  return { labels, trainClips, testClips, trainSessions, testSession };
}

export function audioSignature(samples) {
  let a = 2166136261, b = 5381;
  const view = new DataView(new ArrayBuffer(4));
  for (const x of samples) {
    view.setFloat32(0, x, true);
    const n = view.getUint32(0, true);
    a = Math.imul(a ^ n, 16777619); b = Math.imul(b, 33) ^ n;
  }
  return `${samples.length}:${a >>> 0}:${b >>> 0}`;
}

function forward(model, x) {
  const { hidden, labels, w1, b1, w2, b2 } = model, k = labels.length;
  const h = new Float32Array(hidden), p = new Float32Array(k);
  for (let j = 0; j < hidden; j++) {
    let s = b1[j];
    const offset = j * FEATURE_SIZE;
    for (let i = 0; i < FEATURE_SIZE; i++) s += w1[offset + i] * x[i];
    h[j] = Math.max(0, s);
  }
  let max = -Infinity;
  for (let c = 0; c < k; c++) {
    let s = b2[c];
    for (let j = 0; j < hidden; j++) s += w2[c * hidden + j] * h[j];
    p[c] = s; max = Math.max(max, s);
  }
  let sum = 0;
  for (let c = 0; c < k; c++) { p[c] = Math.exp(p[c] - max); sum += p[c]; }
  for (let c = 0; c < k; c++) p[c] /= sum;
  return { h, p };
}

function normalize(feature, mean, scale) {
  return Float32Array.from(feature, (x, i) => Math.max(-5, Math.min(5, (x - mean[i]) / scale[i])));
}

function augment(feature, rng) {
  const out = new Float32Array(FEATURE_SIZE);
  const shift = Math.floor(rng() * 3) - 1;
  const mask = rng() < .45 ? Math.floor(rng() * (MEL_BANDS - 2)) : -5;
  for (let m = 0; m < MEL_BANDS; m++) for (let t = 0; t < TIME_BINS; t++) {
    const source = t - shift;
    out[m * TIME_BINS + t] = m >= mask && m < mask + 2 ? 0 : source < 0 || source >= TIME_BINS ? 0 : feature[m * TIME_BINS + source];
  }
  return out;
}

export function predictFeatures(model, features) {
  return forward(model, normalize(features, model.mean, model.scale)).p;
}
export function predict(model, samples) {
  const p = predictFeatures(model, extractFeatures(samples));
  return model.labels.map((label, i) => ({ label, probability: p[i] })).sort((a, b) => b.probability - a.probability);
}

export function evaluate(model, clips) {
  const matrix = model.labels.map(() => model.labels.map(() => 0));
  let correct = 0, loss = 0;
  for (const clip of clips) {
    const target = model.labels.indexOf(clip.label);
    if (target < 0) throw new Error(`The model has never learned key ${clip.label}.`);
    const p = predictFeatures(model, clip.features || extractFeatures(clip.samples));
    const predicted = argmax(p);
    matrix[target][predicted]++;
    correct += predicted === target ? 1 : 0;
    loss -= Math.log(Math.max(1e-8, p[target]));
  }
  const perKey = model.labels.map((label, i) => {
    const total = matrix[i].reduce((a, b) => a + b, 0);
    return { label, total, correct: matrix[i][i], recall: total ? matrix[i][i] / total : null };
  });
  const n = clips.length, accuracy = n ? correct / n : 0;
  // Wilson 95% interval makes the uncertainty of a small test set visible.
  const z = 1.96, denom = 1 + z * z / Math.max(1, n);
  const center = (accuracy + z * z / (2 * Math.max(1, n))) / denom;
  const half = z * Math.sqrt(accuracy * (1 - accuracy) / Math.max(1, n) + z * z / (4 * Math.max(1, n) ** 2)) / denom;
  return { accuracy, correct, total: n, loss: n ? loss / n : 0, matrix, perKey, interval: [Math.max(0, Math.min(accuracy, center - half)), Math.min(1, Math.max(accuracy, center + half))] };
}

export async function trainModel(sessions, options = {}, onProgress = () => {}) {
  const { labels, trainClips, testClips, trainSessions, testSession } = prepareSplit(sessions, options.testSessionId);
  const epochs = Math.max(5, Math.min(150, Number(options.epochs) || 45));
  const learningRate = .0015, hidden = 32, batchSize = 16, rng = seededRandom(42);
  onProgress({ phase: 'features', progress: 0 });
  const train = [];
  for (let i = 0; i < trainClips.length; i++) {
    train.push({ ...trainClips[i], features: extractFeatures(trainClips[i].samples) });
    if (i % 32 === 0) { onProgress({ phase: 'features', progress: i / trainClips.length }); await new Promise(resolve => setTimeout(resolve, 0)); }
  }
  const mean = new Float32Array(FEATURE_SIZE), scale = new Float32Array(FEATURE_SIZE);
  for (const clip of train) for (let i = 0; i < FEATURE_SIZE; i++) mean[i] += clip.features[i] / train.length;
  for (const clip of train) for (let i = 0; i < FEATURE_SIZE; i++) scale[i] += (clip.features[i] - mean[i]) ** 2 / train.length;
  for (let i = 0; i < FEATURE_SIZE; i++) scale[i] = Math.max(.08, Math.sqrt(scale[i]));
  const model = {
    kind: 'keytone-model', version: 1, featureVersion: 1, hidden, labels, mean, scale,
    w1: Float32Array.from({ length: hidden * FEATURE_SIZE }, () => (rng() * 2 - 1) * Math.sqrt(6 / FEATURE_SIZE)),
    b1: new Float32Array(hidden), w2: Float32Array.from({ length: labels.length * hidden }, () => (rng() * 2 - 1) * Math.sqrt(6 / hidden)),
    b2: new Float32Array(labels.length), source: testSession.source,
    trainSessionIds: trainSessions.map(s => s.id), testSessionId: testSession.id,
    trainSampleSignatures: trainClips.map(c => audioSignature(c.samples)),
    trainedAt: new Date().toISOString(), epochs, history: []
  };
  const params = ['w1', 'b1', 'w2', 'b2'];
  const moments = Object.fromEntries(params.map(key => [key, [new Float32Array(model[key].length), new Float32Array(model[key].length)]]));
  const indices = Array.from({ length: train.length }, (_, i) => i);
  let step = 0;
  for (let epoch = 1; epoch <= epochs; epoch++) {
    for (let i = indices.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [indices[i], indices[j]] = [indices[j], indices[i]]; }
    let epochLoss = 0, epochCorrect = 0;
    for (let start = 0; start < indices.length; start += batchSize) {
      const grads = Object.fromEntries(params.map(key => [key, new Float32Array(model[key].length)]));
      const size = Math.min(batchSize, indices.length - start);
      for (let n = start; n < start + size; n++) {
        const clip = train[indices[n]], target = labels.indexOf(clip.label);
        const features = options.augmentation === false ? clip.features : augment(clip.features, rng);
        const x = normalize(features, mean, scale), { h, p } = forward(model, x);
        epochLoss -= Math.log(Math.max(1e-8, p[target])); epochCorrect += argmax(p) === target ? 1 : 0;
        p[target] -= 1;
        const dh = new Float32Array(hidden);
        for (let c = 0; c < labels.length; c++) {
          grads.b2[c] += p[c];
          for (let j = 0; j < hidden; j++) { grads.w2[c * hidden + j] += p[c] * h[j]; dh[j] += model.w2[c * hidden + j] * p[c]; }
        }
        for (let j = 0; j < hidden; j++) {
          if (h[j] <= 0) continue;
          grads.b1[j] += dh[j];
          const offset = j * FEATURE_SIZE;
          for (let i = 0; i < FEATURE_SIZE; i++) grads.w1[offset + i] += dh[j] * x[i];
        }
      }
      step++;
      for (const key of params) {
        const [m, v] = moments[key], param = model[key], grad = grads[key];
        const correction1 = 1 - .9 ** step, correction2 = 1 - .999 ** step;
        for (let i = 0; i < param.length; i++) {
          const g = Math.max(-5, Math.min(5, grad[i] / size + (key.startsWith('w') ? 1e-4 * param[i] : 0)));
          m[i] = .9 * m[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g;
          param[i] -= learningRate * (m[i] / correction1) / (Math.sqrt(v[i] / correction2) + 1e-8);
        }
      }
    }
    const entry = { epoch, loss: epochLoss / train.length, accuracy: epochCorrect / train.length };
    model.history.push(entry);
    onProgress({ phase: 'training', progress: epoch / epochs, ...entry });
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  onProgress({ phase: 'evaluating', progress: 1 });
  // Test data is first transformed and scored here, after all fixed training epochs.
  const report = { ...evaluate(model, testClips), testSessionName: testSession.name, testSessionId: testSession.id, trainCount: train.length, source: model.source, labels };
  return { model, report };
}

export function serializeModel(model) {
  return Object.fromEntries(Object.entries(model).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v]));
}

export function validateModel(raw) {
  if (!raw || raw.kind !== 'keytone-model' || raw.version !== 1 || raw.featureVersion !== 1 || raw.hidden !== 32) throw new Error('Unsupported Keytone model.');
  if (!Array.isArray(raw.labels) || raw.labels.length < 2 || raw.labels.length > 36 || raw.labels.some(l => !/^[A-Z0-9]$/.test(l)) || new Set(raw.labels).size !== raw.labels.length) throw new Error('Invalid model labels.');
  if (!['recorded', 'synthetic'].includes(raw.source)) throw new Error('Missing model audio source.');
  const dims = { w1: FEATURE_SIZE * raw.hidden, b1: raw.hidden, w2: raw.labels.length * raw.hidden, b2: raw.labels.length, mean: FEATURE_SIZE, scale: FEATURE_SIZE };
  // Whitelist fields; a model file cannot inject arbitrary state into the application.
  const model = { kind: raw.kind, version: 1, featureVersion: 1, hidden: 32, labels: [...raw.labels], source: raw.source, history: [], trainSessionIds: [], trainSampleSignatures: [] };
  for (const [key, size] of Object.entries(dims)) {
    if (!raw[key] || raw[key].length !== size || Array.from(raw[key]).some(v => typeof v !== 'number' || !Number.isFinite(v) || Math.abs(v) > 1e4 || (key === 'scale' && v < .0001))) throw new Error(`Invalid model ${key}.`);
    model[key] = Float32Array.from(raw[key]);
  }
  if (!Array.isArray(raw.trainSessionIds) || !Array.isArray(raw.trainSampleSignatures) || raw.trainSessionIds.some(x => typeof x !== 'string') || raw.trainSampleSignatures.some(x => typeof x !== 'string')) throw new Error('Missing training provenance.');
  model.trainSessionIds = raw.trainSessionIds.slice(0, 100);
  model.trainSampleSignatures = raw.trainSampleSignatures.slice(0, 5000);
  model.epochs = Number(raw.epochs) || 0;
  model.trainedAt = typeof raw.trainedAt === 'string' ? raw.trainedAt.slice(0, 50) : '';
  model.testSessionId = typeof raw.testSessionId === 'string' ? raw.testSessionId : null;
  return model;
}
