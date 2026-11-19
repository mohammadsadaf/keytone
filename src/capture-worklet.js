class CaptureProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.buffer = new Float32Array(1024); this.offset = 0; this.startFrame = 0; }
  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      if (this.offset === 0) this.startFrame = currentFrame + i;
      let value = 0;
      for (const channel of channels) value += channel[i] / channels.length;
      this.buffer[this.offset++] = value;
      if (this.offset === this.buffer.length) {
        this.port.postMessage({ samples: this.buffer, frame: this.startFrame });
        this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('keytone-capture', CaptureProcessor);
