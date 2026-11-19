import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fftPower, spectrogram, extractFeatures, segmentOnsets, alignClip, resample, encodeWav, audioQuality, SAMPLE_RATE, CLIP_SIZE, FEATURE_SIZE, FRAMES, MEL_BANDS } from '../src/dsp.js';
import { syntheticPress } from '../src/demo.js';
import { seededRandom } from '../src/model.js';

test('FFT locates a known sinusoid and preserves the expected power', () => {
  const n = 512, frequencyBin = 32;
  const samples = Float32Array.from({ length: n }, (_, i) => Math.sin(2 * Math.PI * frequencyBin * i / n));
  const power = fftPower(samples);
  assert.equal(power.indexOf(Math.max(...power)), frequencyBin);
  assert.ok(Math.abs(power[frequencyBin] - (n / 2) ** 2) < 1);
  assert.throws(() => fftPower([1, 2, 3]), /power of two/);
});
test('mel transform produces finite, bounded features, and silence stays silent', () => {
  const press = syntheticPress(0);
  const spectrum = spectrogram(press), features = extractFeatures(press);
  assert.equal(spectrum.length, MEL_BANDS * FRAMES);
  assert.equal(features.length, FEATURE_SIZE);
  assert.ok([...features].every(x => Number.isFinite(x) && x >= 0 && x <= 1.00001));
  assert.ok(features.some(x => x > .8));
  assert.ok(extractFeatures(new Float32Array(CLIP_SIZE)).every(x => x === 0));
});
test('resampling preserves duration and DC level', () => {
  const samples = new Float32Array(48000).fill(.25);
  const result = resample(samples, 48000);
  assert.equal(result.length, SAMPLE_RATE);
  assert.ok(result.every(x => Math.abs(x - .25) < 1e-6));
});
test('segmentation isolates slow presses and groups press/release sounds', () => {
  const samples = new Float32Array(3 * SAMPLE_RATE), rng = seededRandom(7);
  for (const [i, at] of [8000, 20000, 34000].entries()) samples.set(syntheticPress(i, rng), at);
  const clips = segmentOnsets(samples);
  assert.equal(clips.length, 3);
  assert.ok(clips.every(c => c.samples.length === CLIP_SIZE));
  assert.equal(segmentOnsets(new Float32Array(SAMPLE_RATE)).length, 0);
});
test('alignment pads boundaries and quality flags digital clipping', () => {
  const pulse = new Float32Array(100); pulse[2] = .5;
  const aligned = alignClip(pulse);
  assert.equal(aligned.length, CLIP_SIZE);
  assert.equal(Math.max(...aligned), .5);
  assert.equal(audioQuality(new Float32Array(100).fill(1)).clipped, 1);
});
test('WAV export has a valid PCM header and saturates extreme samples', () => {
  const bytes = encodeWav(new Float32Array([0, 1, -1, 2]));
  const v = new DataView(bytes);
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'RIFF');
  assert.equal(v.getUint32(24, true), 16000);
  assert.equal(v.getUint32(40, true), 8);
  assert.equal(v.getInt16(46, true), 32767);
  assert.equal(v.getInt16(48, true), -32768);
  assert.equal(v.getInt16(50, true), 32767);
});
