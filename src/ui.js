import { MEL_BANDS, FRAMES, spectrogram } from './dsp.js';

const paths = {
  wave: '<path d="M3 10v4m4-9v14m5-18v22m5-18v14m4-9v4"/>',
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  mic: '<rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10v2a7 7 0 0014 0v-2M12 19v3m-4 0h8"/>',
  layers: '<path d="m12 3 10 5-10 5L2 8l10-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
  brain: '<path d="M9 4a3 3 0 0 0-5 3 4 4 0 0 0-1 7 4 4 0 0 0 6 5m0-15v16m6-16a3 3 0 0 1 5 3 4 4 0 0 1 1 7 4 4 0 0 1-6 5m0-15v16M9 8H6m9 5h4M9 16H6m9-8h2"/>',
  chart: '<path d="M4 3v17h17M8 15l4-5 4 2 5-7"/>',
  book: '<path d="M12 5v16M3 3c4-1 7 0 9 2 2-2 5-3 9-2v16c-4-1-7 0-9 2-2-2-5-3-9-2V3Z"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  up: '<path d="M12 17V3m-5 5 5-5 5 5M4 15v6h16v-6"/>',
  down: '<path d="M12 3v14m-5-5 5 5 5-5M4 17v4h16v-4"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  play: '<path d="m8 4 12 8-12 8V4Z"/>',
  stop: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V6a4 4 0 0 1 8 0v4m-4 5v2"/>',
  dots: '<circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  headphones: '<path d="M3 14v-3a9 9 0 0 1 18 0v3"/><rect x="2" y="12" width="5" height="9" rx="2"/><rect x="17" y="12" width="5" height="9" rx="2"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
  trash: '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  bolt: '<path d="m13 2-9 12h7l-1 8 10-13h-7l0-7Z"/>',
  folder: '<path d="M3 6h7l2 3h9v11H3V6Z"/>',
  keyboard: '<rect x="2" y="5" width="20" height="14" rx="3"/><path d="M6 9h1m3 0h1m3 0h1m3 0h1M6 12h1m3 0h1m3 0h1m3 0h1M7 16h10"/>'
};
export const icon = (name, size = 20) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.wave}</svg>`;
export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const percent = n => `${Math.round(n * 100)}%`;

function setup(canvas) {
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (!rect.width || !rect.height) return null;
  canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
}
export function drawWave(canvas, samples, color = '#79b8ff') {
  const s = setup(canvas); if (!s) return;
  const { ctx, w, h } = s;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#ffffff08'; ctx.lineWidth = 1;
  for (let i = 1; i < 5; i++) { ctx.beginPath(); ctx.moveTo(0, h * i / 5); ctx.lineTo(w, h * i / 5); ctx.stroke(); }
  for (let x = 0; x < w; x += 48) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  ctx.strokeStyle = color; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(0, h / 2);
  if (samples?.length) {
    const stride = Math.max(1, Math.floor(samples.length / w));
    for (let x = 0; x < w; x++) {
      let lo = 0, hi = 0;
      const start = Math.floor(x * samples.length / w);
      for (let j = start; j < Math.min(samples.length, start + stride); j++) { lo = Math.min(lo, samples[j]); hi = Math.max(hi, samples[j]); }
      ctx.moveTo(x, h / 2 - hi * h * .46); ctx.lineTo(x, h / 2 - lo * h * .46);
    }
  } else { ctx.lineTo(w, h / 2); }
  ctx.stroke();
}
const stops = [[0, [15, 18, 22]], [.18, [30, 28, 52]], [.4, [75, 48, 115]], [.62, [162, 79, 103]], [.82, [239, 146, 87]], [1, [250, 229, 151]]];
function colorAt(v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < stops.length; i++) if (v <= stops[i][0]) {
    const t = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
    return stops[i][1].map((x, j) => Math.round(stops[i - 1][1][j] + (x - stops[i - 1][1][j]) * t));
  }
  return stops.at(-1)[1];
}
export function drawSpectrum(canvas, samples) {
  const s = setup(canvas); if (!s) return;
  const { ctx, w, h } = s, matrix = spectrogram(samples || new Float32Array(4096));
  const temp = document.createElement('canvas'); temp.width = FRAMES; temp.height = MEL_BANDS;
  const tc = temp.getContext('2d'), data = tc.createImageData(FRAMES, MEL_BANDS);
  for (let m = 0; m < MEL_BANDS; m++) for (let t = 0; t < FRAMES; t++) {
    const i = ((MEL_BANDS - 1 - m) * FRAMES + t) * 4, color = colorAt(matrix[m * FRAMES + t]);
    data.data.set([...color, 255], i);
  }
  tc.putImageData(data, 0, 0); ctx.imageSmoothingEnabled = true; ctx.drawImage(temp, 0, 0, w, h);
}
export function drawLearning(canvas, history) {
  const s = setup(canvas); if (!s) return;
  const { ctx, w, h } = s, left = 34, bottom = h - 25, top = 15;
  ctx.font = '10px ui-monospace, monospace'; ctx.fillStyle = '#798ba4'; ctx.strokeStyle = '#ffffff0a';
  for (let i = 0; i <= 4; i++) {
    const y = top + (bottom - top) * i / 4;
    ctx.fillText(`${100 - i * 25}`, 0, y + 4); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w, y); ctx.stroke();
  }
  if (!history.length) { ctx.fillStyle = '#798ba4'; ctx.fillText('Your training curve will appear here.', left + 20, h / 2); return; }
  ctx.strokeStyle = '#79b8ff'; ctx.lineWidth = 2; ctx.beginPath();
  history.forEach((p, i) => { const x = left + i / Math.max(1, history.length - 1) * (w - left - 10), y = bottom - p.accuracy * (bottom - top); if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.stroke();
  ctx.fillStyle = '#798ba4'; ctx.fillText('Epoch 1', left, h - 4); ctx.fillText(`Epoch ${history.length}`, w - 65, h - 4);
}
