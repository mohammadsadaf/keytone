import { SAMPLE_RATE, resample, alignClip } from './dsp.js';

export class Microphone {
  constructor() { this.chunks = []; this.active = false; }
  async start() {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Open Keytone over HTTPS or at http://localhost:3000 in a browser with microphone support.');
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }, video: false });
      this.context = new AudioContext({ sampleRate: SAMPLE_RATE });
      await this.context.resume();
      await this.context.audioWorklet.addModule(new URL('./capture-worklet.js', import.meta.url));
      this.source = this.context.createMediaStreamSource(this.stream);
      this.analyser = this.context.createAnalyser(); this.analyser.fftSize = 2048;
      this.node = new AudioWorkletNode(this.context, 'keytone-capture');
      this.mute = this.context.createGain(); this.mute.gain.value = 0;
      this.node.port.onmessage = ({ data }) => {
        this.chunks.push(data);
        // Keep only the most recent 12 seconds; captures are extracted immediately.
        const cutoff = data.frame - this.context.sampleRate * 12;
        while (this.chunks.length && this.chunks[0].frame < cutoff) this.chunks.shift();
      };
      this.source.connect(this.analyser);
      this.source.connect(this.node); this.node.connect(this.mute); this.mute.connect(this.context.destination);
      this.active = true;
      this.label = this.stream.getAudioTracks()[0]?.label || 'Default microphone';
    } catch (error) {
      await this.stop();
      if (error.name === 'NotAllowedError') throw new Error('Microphone access was denied. Allow it in your browser’s site settings, then try again.');
      if (error.name === 'NotFoundError') throw new Error('No microphone found. Connect one, or try the synthetic demo.');
      throw error;
    }
  }
  range(start, end) {
    const rate = this.context.sampleRate, from = Math.floor(start * rate), to = Math.floor(end * rate);
    const samples = new Float32Array(Math.max(0, to - from));
    for (const chunk of this.chunks) {
      const a = Math.max(from, chunk.frame), b = Math.min(to, chunk.frame + chunk.samples.length);
      if (b > a) samples.set(chunk.samples.subarray(a - chunk.frame, b - chunk.frame), a - from);
    }
    return resample(samples, rate);
  }
  clipAt(time) {
    return alignClip(this.range(Math.max(0, time - .1), time + .26), Math.round(Math.min(.1, time) * SAMPLE_RATE));
  }
  async stop() {
    this.active = false;
    this.stream?.getTracks().forEach(track => track.stop());
    this.source?.disconnect(); this.node?.disconnect(); this.mute?.disconnect();
    if (this.context && this.context.state !== 'closed') await this.context.close();
  }
}

export async function decodeFile(file) {
  if (!file || file.size > 30 * 1024 * 1024) throw new Error('Choose an audio file smaller than 30 MB.');
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    if (buffer.duration > 300) throw new Error('Use an audio recording under 5 minutes.');
    const mono = new Float32Array(buffer.length);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const channel = buffer.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / buffer.numberOfChannels;
    }
    return resample(mono, buffer.sampleRate);
  } catch (error) {
    if (error.name === 'EncodingError') throw new Error('This audio format could not be decoded. Try a WAV file.');
    throw error;
  } finally { await context.close(); }
}

let playbackContext;
export async function playSamples(samples) {
  playbackContext ||= new AudioContext();
  await playbackContext.resume();
  const buffer = playbackContext.createBuffer(1, samples.length, SAMPLE_RATE);
  buffer.copyToChannel(samples, 0);
  const source = playbackContext.createBufferSource(); source.buffer = buffer;
  source.connect(playbackContext.destination); source.start();
}
