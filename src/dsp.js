// The same deterministic feature pipeline is used for training and inference.
export const SAMPLE_RATE = 16000;
export const CLIP_SIZE = 4096;
export const FFT_SIZE = 512;
export const HOP = 128;
export const MEL_BANDS = 32;
export const TIME_BINS = 16;
export const FEATURE_SIZE = MEL_BANDS * TIME_BINS;
export const FRAMES = Math.floor((CLIP_SIZE - FFT_SIZE) / HOP) + 1;

export function resample(input, fromRate, toRate = SAMPLE_RATE) {
  if (!(fromRate > 0) || !(toRate > 0)) throw new Error('Invalid sample rate.');
  if (fromRate === toRate) return Float32Array.from(input);
  const length = Math.floor(input.length * toRate / fromRate);
  const result = new Float32Array(length);
  // Area averaging prevents high-frequency content from simply aliasing when downsampling.
  for (let i = 0; i < length; i++) {
    const start = i * fromRate / toRate, end = Math.min(input.length, (i + 1) * fromRate / toRate);
    let total = 0;
    for (let j = Math.floor(start); j < Math.ceil(end); j++) {
      total += (input[Math.min(j, input.length - 1)] || 0) * Math.max(0, Math.min(j + 1, end) - Math.max(j, start));
    }
    result[i] = total / (end - start);
  }
  return result;
}

export function fftPower(input) {
  const n = input.length;
  if (n < 2 || (n & (n - 1))) throw new Error('FFT size must be a power of two.');
  const re = Float64Array.from(input), im = new Float64Array(n);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) [re[i], re[j]] = [re[j], re[i]];
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -2 * Math.PI / len, cr = Math.cos(angle), ci = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wr = 1, wi = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j, b = a + len / 2;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const next = wr * cr - wi * ci;
        wi = wr * ci + wi * cr; wr = next;
      }
    }
  }
  return Float64Array.from({ length: n / 2 + 1 }, (_, i) => re[i] ** 2 + im[i] ** 2);
}

const hzToMel = hz => 2595 * Math.log10(1 + hz / 700);
const melToHz = mel => 700 * (10 ** (mel / 2595) - 1);
const low = hzToMel(80), high = hzToMel(7600);
const centers = Array.from({ length: MEL_BANDS + 2 }, (_, i) => melToHz(low + (high - low) * i / (MEL_BANDS + 1)));
const filters = Array.from({ length: MEL_BANDS }, (_, m) => Float64Array.from({ length: FFT_SIZE / 2 + 1 }, (_, k) => {
  const hz = k * SAMPLE_RATE / FFT_SIZE;
  return Math.max(0, Math.min((hz - centers[m]) / (centers[m + 1] - centers[m]), (centers[m + 2] - hz) / (centers[m + 2] - centers[m + 1])));
}));
const window = Float64Array.from({ length: FFT_SIZE }, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / (FFT_SIZE - 1)));

export function spectrogram(samples) {
  const matrix = new Float32Array(MEL_BANDS * FRAMES);
  const frame = new Float64Array(FFT_SIZE);
  let maxDb = -120;
  for (let t = 0; t < FRAMES; t++) {
    for (let i = 0; i < FFT_SIZE; i++) frame[i] = (samples[t * HOP + i] || 0) * window[i];
    const power = fftPower(frame);
    for (let m = 0; m < MEL_BANDS; m++) {
      let energy = 0;
      for (let k = 0; k < power.length; k++) energy += power[k] * filters[m][k];
      const db = 10 * Math.log10(Math.max(1e-12, energy));
      matrix[m * FRAMES + t] = db;
      maxDb = Math.max(maxDb, db);
    }
  }
  for (let i = 0; i < matrix.length; i++) matrix[i] = Math.max(0, (matrix[i] - maxDb + 60) / 60);
  // Pure silence has no spectral information, even though all bins are equally loud.
  if (maxDb <= -119) matrix.fill(0);
  return matrix;
}

export function extractFeatures(samples) {
  const matrix = spectrogram(samples), feature = new Float32Array(FEATURE_SIZE);
  for (let m = 0; m < MEL_BANDS; m++) {
    for (let t = 0; t < TIME_BINS; t++) {
      const start = Math.floor(t * FRAMES / TIME_BINS), end = Math.floor((t + 1) * FRAMES / TIME_BINS);
      let sum = 0;
      for (let j = start; j < end; j++) sum += matrix[m * FRAMES + j];
      feature[m * TIME_BINS + t] = sum / (end - start);
    }
  }
  return feature;
}

export function audioQuality(samples) {
  let peak = 0, energy = 0, clipped = 0;
  for (const x of samples) { peak = Math.max(peak, Math.abs(x)); energy += x * x; if (Math.abs(x) >= .995) clipped++; }
  return { peak, rms: Math.sqrt(energy / Math.max(1, samples.length)), clipped: clipped / Math.max(1, samples.length) };
}

export function alignClip(samples, around = null) {
  // Align to the strongest 2 ms transient, with 24 ms of context before it.
  let best = -1, at = 0;
  const start = around === null ? 0 : Math.max(0, around - 960);
  const end = around === null ? samples.length : Math.min(samples.length, around + 1280);
  for (let i = start; i < end - 32; i += 16) {
    let energy = 0;
    for (let j = 0; j < 32; j++) energy += samples[i + j] ** 2;
    if (energy > best) { best = energy; at = i; }
  }
  const result = new Float32Array(CLIP_SIZE);
  for (let i = 0; i < CLIP_SIZE; i++) result[i] = samples[at - 384 + i] || 0;
  return result;
}

export function segmentOnsets(samples, threshold = .012) {
  const step = 80, energies = [];
  for (let i = 0; i < samples.length; i += step) {
    let sum = 0;
    for (let j = i; j < Math.min(i + step, samples.length); j++) sum += samples[j] ** 2;
    energies.push(Math.sqrt(sum / step));
  }
  const sorted = [...energies].sort((a, b) => a - b);
  const noise = sorted[Math.floor(sorted.length * .3)] || 0;
  const gate = Math.max(threshold, noise * 5);
  const peaks = [];
  let last = -1000;
  for (let i = 1; i < energies.length - 1; i++) {
    if (energies[i] < gate || energies[i] < energies[i - 1] || energies[i] < energies[i + 1]) continue;
    if (i - last < 44) continue; // 220 ms refractory period groups down/up clicks.
    let best = i;
    for (let j = i; j < Math.min(i + 16, energies.length); j++) if (energies[j] > energies[best]) best = j;
    peaks.push(best * step); last = best;
  }
  return peaks.map(at => {
    const start = Math.max(0, at - 1280), end = Math.min(samples.length, at + CLIP_SIZE);
    return { at: at / SAMPLE_RATE, samples: alignClip(samples.slice(start, end), at - start) };
  });
}

export function encodeWav(samples, sampleRate = SAMPLE_RATE) {
  const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
  const str = (at, s) => { for (let i = 0; i < s.length; i++) view.setUint8(at + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((x, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, x)) * (x < 0 ? 32768 : 32767)), true));
  return buffer;
}
