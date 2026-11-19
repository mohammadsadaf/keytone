import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDemo } from '../src/demo.js';
import { trainModel, prepareSplit, predict, serializeModel, validateModel } from '../src/model.js';
import { exportDataset, validateDataset } from '../src/store.js';

test('split preserves whole sessions and requires class coverage', () => {
  const sessions = createDemo();
  const split = prepareSplit(sessions, sessions[2].id);
  assert.equal(split.trainClips.length, 216); assert.equal(split.testClips.length, 108);
  assert.ok(!split.trainSessions.some(s => s.id === split.testSession.id));
  assert.throws(() => prepareSplit([sessions[0]], sessions[0].id), /two separate/);
  sessions[2].clips = sessions[2].clips.filter(c => c.label !== 'K');
  assert.throws(() => prepareSplit(sessions, sessions[2].id), /held-out clips for K/);
});
test('split rejects repeated clips, relabeled copies, and mixed sources', () => {
  const sessions = createDemo();
  sessions[2].clips[0].samples = sessions[0].clips[0].samples.slice();
  assert.throws(() => prepareSplit(sessions, sessions[2].id), /Identical audio/);
  sessions[2].clips[0].id = sessions[0].clips[0].id;
  assert.throws(() => prepareSplit(sessions, sessions[2].id), /both training and test/);
  sessions[2].source = 'recorded';
  assert.throws(() => prepareSplit(sessions, sessions[2].id), /separate experiments/);
});
test('the neural network learns audio features, evaluates an unseen session, and round-trips weights', async () => {
  const sessions = createDemo(), progress = [];
  const { model, report } = await trainModel(sessions, { testSessionId: sessions[2].id, epochs: 15 }, p => progress.push(p));
  assert.ok(report.accuracy > .9, `Demo accuracy was ${report.accuracy}`);
  assert.equal(report.total, 108);
  assert.equal(report.matrix.flat().reduce((a, b) => a + b, 0), 108);
  assert.ok(model.history.at(-1).loss < model.history[0].loss);
  assert.ok(progress.some(p => p.phase === 'features'));
  assert.ok(progress.some(p => p.phase === 'training'));
  assert.ok(report.interval[0] <= report.accuracy && report.interval[1] >= report.accuracy);
  const restored = validateModel(JSON.parse(JSON.stringify(serializeModel(model))));
  const clip = sessions[2].clips[0], p = predict(model, clip.samples), roundTrip = predict(restored, clip.samples);
  assert.deepEqual(p, roundTrip);
  assert.equal(p[0].label, clip.label);
  assert.ok(Math.abs(p.reduce((a, b) => a + b.probability, 0) - 1) < 1e-6);
  const broken = serializeModel(model); broken.scale[0] = 0;
  assert.throws(() => validateModel(broken), /scale/);
});
test('held-out audio does not affect training weights or normalization', async () => {
  const first = createDemo(), second = createDemo();
  for (const clip of second[2].clips) clip.samples = Float32Array.from(clip.samples, x => x * .01);
  const options = { testSessionId: first[2].id, epochs: 5 };
  const a = (await trainModel(first, options)).model, b = (await trainModel(second, options)).model;
  for (const key of ['w1', 'w2', 'b1', 'b2', 'mean', 'scale']) assert.deepEqual(a[key], b[key]);
});
test('dataset exports round-trip and malformed audio and duplicate IDs fail validation', () => {
  const original = createDemo().slice(0, 1), raw = exportDataset(original);
  const restored = validateDataset(JSON.parse(JSON.stringify(raw)));
  assert.deepEqual(restored[0].clips[0].samples, original[0].clips[0].samples);
  raw.sessions[0].clips[0].samples[0] = '0.5';
  assert.throws(() => validateDataset(raw), /invalid audio/);
  const duplicate = exportDataset(original); duplicate.sessions.push(duplicate.sessions[0]);
  assert.throws(() => validateDataset(duplicate), /duplicate session/);
});
