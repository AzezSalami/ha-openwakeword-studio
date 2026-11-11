// Captures microphone audio and posts 16 kHz mono 16-bit PCM in 80 ms chunks.
class Recorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; // 1 when the context already runs at 16 kHz
    this.out = new Int16Array(1280);
    this.n = 0;
    this.acc = 0;
    this.count = 0;
    this.phase = 0;
  }

  emit(value) {
    const v = Math.max(-1, Math.min(1, value));
    this.out[this.n++] = v < 0 ? v * 32768 : v * 32767;
    if (this.n === this.out.length) {
      const chunk = this.out;
      this.port.postMessage(chunk.buffer, [chunk.buffer]);
      this.out = new Int16Array(1280);
      this.n = 0;
    }
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < channel.length; i++) this.emit(channel[i]);
      return true;
    }
    // Fallback: average each window of `ratio` input samples (crude low-pass + decimate).
    for (let i = 0; i < channel.length; i++) {
      this.acc += channel[i];
      this.count++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        this.emit(this.acc / this.count);
        this.acc = 0;
        this.count = 0;
      }
    }
    return true;
  }
}
registerProcessor("recorder", Recorder);
