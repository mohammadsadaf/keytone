# Keytone

A complete, local acoustic keyboard learning lab. Record your own labeled keystrokes, turn the audio into mel spectrograms, train a neural network, and measure its performance on a separate recording session.

[Open the browser app](https://mohammadsadaf.github.io/keytone/) · [Source on GitHub](https://github.com/mohammadsadaf/keytone)

**No install step, API key, account, Python environment, or cloud service is required.** The app has no third-party runtime dependencies. Its Node server only serves static assets; audio and machine learning run in your browser.

## Tech stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Frontend | HTML5, CSS3, vanilla JavaScript (ES modules) | Responsive interface with a blue theme |
| Audio | Web Audio API and AudioWorklet | Microphone capture, audio decoding, and PCM processing |
| Signal processing | Custom JavaScript FFT and mel filter bank | Onset detection, clip alignment, and mel spectrogram features |
| Machine learning | Custom JavaScript neural network and Web Workers | Train a 512 → 32 ReLU → softmax classifier with Adam in the background |
| Visualizations | Canvas 2D API | Waveforms, spectrograms, and training curves |
| Local storage | IndexedDB | Save each visitor's recordings, sessions, models, and reports in their browser |
| Development | Node.js 20+ and npm scripts | Serve the app locally without installing dependencies |
| Testing | Node.js built-in test runner and assert module | Verify audio processing, training, data splits, and import/export validation |
| Version control and hosting | Git, GitHub, and GitHub Pages | Manage source code and publish the static HTTPS app |

The published app runs entirely in the browser. It uses no frontend framework, external machine-learning library, Python runtime, or backend API.

## Publish on GitHub Pages

This app can run entirely as a static HTTPS site. Upload the repository with its `src/` and `tests/` folders intact, then choose **Settings → Pages → Deploy from a branch → main → / (root)**. The included `.nojekyll` file serves the application directly. Module, stylesheet, training-worker, and audio-worklet URLs work under a repository subdirectory as well as at a domain root.

Visitors can explore the synthetic demo immediately. Microphone experiments require a supported browser and explicit microphone permission. All recordings and training remain in each visitor's own browser; publishing the app does not publish or transfer your local recordings.

## Run

Requires **Node.js 20 or later** and a current desktop browser with Web Audio / AudioWorklet support. Chrome or Edge is a practical choice for microphone experiments.

```sh
cd /path/to/keytone
npm start
```

Open **http://localhost:3000**. Use `Ctrl+C` in the terminal to stop the server. If port 3000 is occupied:

```sh
PORT=3001 npm start
```

Then open `http://localhost:3001`. Browser data is tied to the exact origin, including its port: export your dataset before switching ports or browsers. Open the app through the server; double-clicking `index.html` will not enable module loading and microphone capture correctly.

## Try the full workflow in a minute

1. Click **Demo**. It generates 324 resonant clicks across six keys and three separate synthetic sessions.
2. Open **Dataset**. Select a clip, inspect its waveform and spectrogram, and listen to it.
3. Open **Train model**. Leave the third session held out and click **Train model**.
4. Open **Results** to see measured accuracy, the confusion matrix, a Wilson interval, and single-clip predictions.
5. Export the dataset, model weights, or evaluation report.

**Demo audio is synthetic and deliberately separable. A high demo score verifies the software workflow; it does not demonstrate real-world keyboard recognition.** Synthetic data and your recordings have separate datasets, models, and reports.

## Record your own keyboard

1. Select **My recordings → Capture**. Start with 4–6 keys; the default is A, S, D, F, J, K.
2. Enter a session name and choose samples per key. Twenty-five is a useful starting point.
3. Click **Start recording** and allow the browser to use your microphone. Your operating system may also need to allow microphone access for that browser.
4. Press the highlighted key slowly, leaving at least 0.4 seconds between presses. The studio advances automatically when you have enough examples of that key.
5. Finish the session or click **Stop & save**. The microphone disconnects. Capture also stops if the tab becomes hidden.
6. Record another full session with the same keys. Keep the mic, keyboard, and desk fixed at first. Each recording start creates a new session, even if names are identical.
7. Inspect the clips in **Dataset**. Listen for missed clicks or background sounds. Remove bad examples; an Undo action is available for ten seconds. Clips can be downloaded individually as WAV.
8. Train while holding out one entire session. At least five training clips and two held-out clips per key are required; more examples give a more useful experiment.

Keydown timestamps provide labels and help locate the recording window during explicit capture. **The classifier receives only features derived from audio**, never the pressed key, timestamp, session name, or position. Recording is scoped to this page. It is not a system-wide keyboard listener.

### Use a phone or an existing recording

Choose **Import audio** on Capture. Each audio file should contain isolated repetitions of **one known key**, ideally spaced at least 0.4 seconds apart. WAV is recommended; other browser-decodable audio formats may work.

Select its key and recording session. The app detects transients, aligns 256 ms clips, and lets you listen to and uncheck candidates before adding them. Raise the detection threshold if noise produces extra clips. Quiet or heavily clipped candidates are omitted when accepted.

Import the other key files from the **same recording sitting into the same session**. Make a new session for a separate sitting. Do not split one original recording into artificially separate train/test sessions: the software cannot infer the true provenance of arbitrary imported audio. Limits: 30 MB / five minutes per file; up to 500 candidates per file.

## Training and evaluation

This is an accessible implementation of the core workflow, **not a reproduction of the paper's CoAtNet architecture or reported accuracy**.

- Audio: mono, resampled to 16 kHz; 4,096 samples (256 ms) per clip.
- Alignment: strongest short energy transient, with 24 ms before it.
- Spectrogram: 512-point FFT, Hann window, 128-sample hop, 32 triangular mel bands spanning 80–7,600 Hz, relative 60 dB dynamic range.
- Model input: spectrogram pooled to 32 bands × 16 time bins, flattened to 512 values.
- Network: 512 inputs → 32 ReLU hidden units → one softmax output per learned key. Trained from scratch in a dedicated Web Worker.
- Optimizer: mini-batch Adam, batch size 16, learning rate 0.0015, small L2 penalty; deterministic seeded initialization and shuffling.
- Augmentation: random ±1 time-bin shifts and two-band frequency masks, applied only to training inputs.
- Split: one entire session held out; all remaining sessions used for training. Normalization is fitted only on training features. The test session is first scored after the fixed training epochs finish.
- Leakage checks: reject matching train/test clip IDs and exact audio copies even if IDs differ. These checks cannot detect all related recordings, near-duplicates, or misleading session labels.
- Metrics: held-out top-1 accuracy, correct/total counts, per-key recall, cross-entropy loss, confusion matrix, uniform-guess baseline, and a Wilson 95% binomial interval. The interval is descriptive; correlated keystrokes may make it overoptimistic.

The learning curve displays **training** accuracy, not test accuracy. Repeatedly tuning settings after looking at the held-out score can overfit the test set. Collect an additional untouched session for a final check. Session variation, noise, typing style, overlapping presses, and microphone processing can reduce accuracy substantially. No accuracy is guaranteed.

**Try a single press** records three seconds and predicts from exactly one detected sound without using keyboard events. Predictions are limited to learned classes and the displayed probabilities are uncalibrated. An unseen key or unrelated click can still receive a confident label.

## Storage and exports

Recordings, sessions, model weights, and reports persist in **IndexedDB on this browser and origin**. There is no audio upload endpoint, analytics, remote model, CDN, or background network request. Clearing site data removes this workspace; private browsing or browser storage limits can prevent persistence.

- **Dataset JSON:** labeled PCM samples, session metadata, source type, and schema version. Import adds sessions whose IDs are new, preserving existing sessions. Existing session IDs are skipped; duplicate clip IDs or invalid samples are rejected. Dataset changes invalidate that source's model and report.
- **Model JSON:** feature version, weights, normalization, class labels, and training session/audio signatures. Imported models do not carry over an evaluation report; evaluate an unused session or try a new single press.
- **Report JSON:** evaluation counts, labels, matrix, uncertainty interval, and training provenance.
- **WAV:** 16-bit mono PCM for an individual selected clip.

JSON imports are capped at 100 MB, 100 sessions, and 5,000 clips. Export large experiments in manageable subsets if needed. Keep your dataset and model exports together so you retain the experimental context.

## Test

```sh
npm test
```

Uses Node's built-in test runner. Tests cover known-frequency FFT output, silence handling, feature bounds, resampling, onset segmentation, alignment, WAV encoding, malformed imports, duplicate recordings, session/class coverage, learning on synthetic audio, model serialization, and the invariant that changing held-out audio cannot change training weights or normalization.

Browser verification covers the demo workflow, dataset filters, worker training, held-out results, predictions, persistent storage, imports/exports, and responsive layouts. Real microphone quality and real-keyboard recognition must be checked on your hardware; the demo tests do not validate those claims.

## Project structure

```text
keytone/
  index.html                 App shell
  styles.css                 Responsive interface
  server.mjs                 Loopback-only static server
  src/
    app.js                   UI, workflow, capture labels, imports/exports
    audio.js                 Microphone lifecycle and audio decoding
    capture-worklet.js       Mono PCM capture off the UI thread
    dsp.js                   FFT, mel features, segmentation, WAV encoding
    model.js                 Neural network, training, evaluation, validation
    train-worker.js          Background training
    demo.js                  Reproducible, explicitly synthetic audio
    store.js                 IndexedDB and dataset schema
    ui.js                    Icons and signal/chart rendering
  tests/
    dsp.test.js
    model.test.js
```

## Sources

- Harrison, Toreini & Mehrnezhad (2023), [A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards](https://arxiv.org/abs/2308.01074). The authors report 95% accuracy for their smartphone recordings and 93% for Zoom in their own experiment. Those are not this app's benchmark results.
- MDN: [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet) and [getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

MIT licensed. Use intentional recordings of your own keyboard or recordings whose participants have agreed to the experiment.
