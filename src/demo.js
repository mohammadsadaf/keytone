import { CLIP_SIZE, SAMPLE_RATE, alignClip } from './dsp.js';
import { seededRandom } from './model.js';

export const DEMO_KEYS = ['A', 'S', 'D', 'F', 'J', 'K'];
export function syntheticPress(keyIndex, rng = seededRandom(42), session = 0) {
  const audio = new Float32Array(CLIP_SIZE);
  const modes = [520 + keyIndex * 225, 1420 + keyIndex * 410, 3400 + keyIndex * 430];
  const start = 300 + Math.floor(rng() * 100), gain = .38 + rng() * .18;
  const detune = 1 + (rng() - .5) * .025 + session * .003;
  const phases = modes.map(() => rng() * .2);
  for (let i = 0; i < audio.length; i++) {
    const t = (i - start) / SAMPLE_RATE;
    let value = (rng() - .5) * .002;
    if (t >= 0) {
      value += modes.reduce((sum, f, j) => sum + Math.sin(2 * Math.PI * f * detune * t + phases[j]) * Math.exp(-t * (42 + j * 32)) / (j + 1), 0) * gain;
      value += (rng() - .5) * .35 * Math.exp(-t * 380);
      if (t > .085) value += Math.sin(2 * Math.PI * modes[1] * (t - .085)) * .065 * Math.exp(-(t - .085) * 100);
    }
    audio[i] = Math.max(-1, Math.min(1, value));
  }
  return alignClip(audio);
}
export function createDemo() {
  const rng = seededRandom(2026);
  return Array.from({ length: 3 }, (_, session) => ({
    id: `demo-session-${session + 1}`, name: `Demo session ${String(session + 1).padStart(2, '0')}`,
    source: 'synthetic', createdAt: new Date(2026, 0, session + 1).toISOString(),
    clips: DEMO_KEYS.flatMap((label, keyIndex) => Array.from({ length: 18 }, (_, i) => ({
      id: `demo-${session}-${label}-${i}`, label, samples: syntheticPress(keyIndex, rng, session), createdAt: new Date(2026, 0, session + 1).toISOString()
    })))
  }));
}
