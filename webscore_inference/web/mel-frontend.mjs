const SAMPLE_RATE = 16000;
const N_FFT = 2048;
const HOP_LENGTH = 160;
const N_MELS = 512;
const CHUNK_SAMPLES = 80000;
const MEL_EPS = 1e-6;

function hzToMelHtk(hz) {
  return 2595.0 * Math.log10(1.0 + hz / 700.0);
}

function melToHzHtk(mel) {
  return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
}

function buildHannWindow(size) {
  const win = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    win[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / size));
  }
  return win;
}

function buildMelFilterbank(sr = SAMPLE_RATE, nFft = N_FFT, nMels = N_MELS) {
  const nFreqs = (nFft >> 1) + 1;
  const fMax = sr / 2.0;
  const melMin = hzToMelHtk(0.0);
  const melMax = hzToMelHtk(fMax);

  const fPts = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    const m = melMin + ((melMax - melMin) * i) / (nMels + 1);
    fPts[i] = melToHzHtk(m);
  }

  const fb = new Float32Array(nMels * nFreqs);
  for (let k = 0; k < nFreqs; k++) {
    const freq = (fMax * k) / (nFreqs - 1);
    for (let m = 0; m < nMels; m++) {
      const fLeft = fPts[m];
      const fCenter = fPts[m + 1];
      const fRight = fPts[m + 2];
      const down = (freq - fLeft) / Math.max(fCenter - fLeft, 1e-12);
      const up = (fRight - freq) / Math.max(fRight - fCenter, 1e-12);
      const val = Math.max(0.0, Math.min(down, up));
      fb[m * nFreqs + k] = val;
    }
  }
  return fb;
}

class Radix2FFT {
  constructor(n) {
    this.n = n;
    this.re = new Float32Array(n);
    this.im = new Float32Array(n);
    this.rev = new Uint32Array(n);
    const bits = Math.round(Math.log2(n));
    for (let i = 0; i < n; i++) {
      let x = i;
      let r = 0;
      for (let b = 0; b < bits; b++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
      }
      this.rev[i] = r;
    }
    this.cosTable = new Float32Array(n >> 1);
    this.sinTable = new Float32Array(n >> 1);
    for (let i = 0; i < (n >> 1); i++) {
      const angle = (-2.0 * Math.PI * i) / n;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }
  }

  magnitudeSpectrum(frame, outMag) {
    const n = this.n;
    const re = this.re;
    const im = this.im;
    im.fill(0);
    for (let i = 0; i < n; i++) {
      re[this.rev[i]] = frame[i];
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = 0; j < half; j++) {
          const idx = j * step;
          const cos = this.cosTable[idx];
          const sin = this.sinTable[idx];
          const k = i + j + half;
          const tRe = re[k] * cos - im[k] * sin;
          const tIm = re[k] * sin + im[k] * cos;
          re[k] = re[i + j] - tRe;
          im[k] = im[i + j] - tIm;
          re[i + j] += tRe;
          im[i + j] += tIm;
        }
      }
    }
    const nFreqs = (n >> 1) + 1;
    for (let k = 0; k < nFreqs; k++) {
      outMag[k] = Math.hypot(re[k], im[k]);
    }
  }
}

export class WebMelProcessor {
  constructor() {
    this.window = buildHannWindow(N_FFT);
    this.melFb = buildMelFilterbank(SAMPLE_RATE, N_FFT, N_MELS);
    // HTK triangle filters have narrow support. Skip exact zero coefficients,
    // retaining the original increasing-frequency accumulation order.
    const nFreqs = (N_FFT >> 1) + 1;
    this.melRanges = Array.from({ length: N_MELS }, (_, m) => {
      let first = nFreqs, end = 0;
      for (let k = 0; k < nFreqs; k++) {
        if (this.melFb[m * nFreqs + k] !== 0) { first = Math.min(first, k); end = k + 1; }
      }
      return [first, end];
    });
    this.fft = new Radix2FFT(N_FFT);
    this._frameBuf = new Float32Array(N_FFT);
    this._magBuf = new Float32Array((N_FFT >> 1) + 1);
  }

  computeChunkLogMel(pcm5s) {
    const input = new Float32Array(CHUNK_SAMPLES);
    input.set(pcm5s.subarray(0, Math.min(pcm5s.length, CHUNK_SAMPLES)));

    const pad = N_FFT >> 1;
    const paddedLen = CHUNK_SAMPLES + 2 * pad;
    const padded = new Float32Array(paddedLen);
    padded.set(input, pad);
    for (let i = 0; i < pad; i++) {
      padded[pad - 1 - i] = input[i + 1];
      padded[pad + CHUNK_SAMPLES + i] = input[CHUNK_SAMPLES - 2 - i];
    }

    const numFrames = Math.floor((paddedLen - N_FFT) / HOP_LENGTH) + 1;
    const nFreqs = (N_FFT >> 1) + 1;
    const out = new Float32Array(numFrames * N_MELS);

    for (let f = 0; f < numFrames; f++) {
      const start = f * HOP_LENGTH;
      for (let i = 0; i < N_FFT; i++) {
        this._frameBuf[i] = padded[start + i] * this.window[i];
      }
      this.fft.magnitudeSpectrum(this._frameBuf, this._magBuf);
      const rowOffset = f * N_MELS;
      for (let m = 0; m < N_MELS; m++) {
        const fbOffset = m * nFreqs;
        let sum = 0.0;
        const [first, end] = this.melRanges[m];
        for (let k = first; k < end; k++) {
          sum += this._magBuf[k] * this.melFb[fbOffset + k];
        }
        out[rowOffset + m] = Math.log(sum + MEL_EPS);
      }
    }

    return { data: out, dims: [1, numFrames, N_MELS] };
  }
}

export function splitPcmInto5sChunks(pcm16k) {
  const chunks = [];
  for (let offset = 0; offset < pcm16k.length; offset += CHUNK_SAMPLES) {
    const slice = new Float32Array(CHUNK_SAMPLES);
    const end = Math.min(offset + CHUNK_SAMPLES, pcm16k.length);
    slice.set(pcm16k.subarray(offset, end));
    chunks.push({ pcm: slice, startSec: offset / SAMPLE_RATE });
  }
  return chunks;
}

export function downmixToMono(channels) {
  if (!channels.length) return new Float32Array(0);
  if (channels.length === 1) return channels[0];
  const len = channels[0].length;
  const mono = new Float32Array(len);
  const scale = 1.0 / channels.length;
  for (let ch = 0; ch < channels.length; ch++) {
    const buf = channels[ch];
    for (let i = 0; i < len; i++) mono[i] += buf[i] * scale;
  }
  return mono;
}

export function resampleLinear(pcm, srcRate, dstRate = SAMPLE_RATE) {
  if (srcRate === dstRate || pcm.length === 0) return pcm;
  const outLen = Math.max(1, Math.round((pcm.length * dstRate) / srcRate));
  const out = new Float32Array(outLen);
  const ratio = srcRate / dstRate;
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(pcm.length - 1, i0 + 1);
    const frac = pos - i0;
    out[i] = pcm[i0] * (1.0 - frac) + pcm[i1] * frac;
  }
  return out;
}

export async function decodeAudioTo16kMono(input) {
  if (input instanceof Float32Array) return input;
  const arrayBuf = input instanceof ArrayBuffer ? input : await input.arrayBuffer();
  const AudioCtx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!AudioCtx) {
    throw new Error("Web AudioContext unavailable; pass 16 kHz mono Float32Array directly.");
  }
  const ctx = new AudioCtx();
  try {
    const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
    const targetLen = Math.max(1, Math.ceil(decoded.duration * SAMPLE_RATE));
    if (typeof globalThis.OfflineAudioContext === "function") {
      const offline = new globalThis.OfflineAudioContext(1, targetLen, SAMPLE_RATE);
      const src = offline.createBufferSource();
      src.buffer = decoded;
      src.connect(offline.destination);
      src.start(0);
      const rendered = await offline.startRendering();
      return new Float32Array(rendered.getChannelData(0));
    }
    const channels = [];
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      channels.push(decoded.getChannelData(ch));
    }
    return resampleLinear(downmixToMono(channels), decoded.sampleRate, SAMPLE_RATE);
  } finally {
    if (typeof ctx.close === "function") await ctx.close();
  }
}
