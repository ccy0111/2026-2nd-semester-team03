// Mono BS.1770 integrated loudness, fixed 16 kHz inference audio.
const DB = -0.691;
const ABSOLUTE_GATE = 10 ** ((-70 - DB) / 10);
const SAMPLE_RATE = 16000;
const BLOCK = SAMPLE_RATE * .4;
const HOP = SAMPLE_RATE * .1;
const TARGET = -18;
const LIMIT = 10 ** (-1 / 20); // −1 dBFS sample-peak headroom (not true peak)
const MAX_GAIN = 10 ** (12 / 20);

function biquad(b0, b1, b2, a0, a1, a2) {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  return x => {
    const y = (b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2) / a0;
    x2 = x1; x1 = x; y2 = y1; y1 = y;
    return y;
  };
}

function kWeightFilters() {
  const k = Math.tan(Math.PI * 1681.974450955533 / SAMPLE_RATE);
  const q = .7071752369554196;
  const vh = 10 ** (3.999843853973347 / 20);
  const vb = vh ** .4996667741545416;
  const shelf = biquad(vh + vb*k/q + k*k, 2*(k*k-vh), vh - vb*k/q + k*k,
    1 + k/q + k*k, 2*(k*k-1), 1-k/q+k*k);
  const h = Math.tan(Math.PI * 38.13547087602444 / SAMPLE_RATE);
  const hq = .5003270373238773;
  const a0 = 1+h/hq+h*h;
  const highpass = biquad(1,-2,1,1,2*(h*h-1)/a0,(1-h/hq+h*h)/a0);
  return x => highpass(shelf(x));
}

export function measureLoudness(samples) {
  const weight = kWeightFilters();
  const blocks = [], ring = new Float64Array(BLOCK);
  let sum = 0, peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const x = samples[i];
    if (!Number.isFinite(x)) throw new RangeError('Non-finite input PCM');
    peak = Math.max(peak, Math.abs(x));
    const y = weight(x), slot = i % BLOCK;
    sum += y*y - ring[slot];
    ring[slot] = y*y;
    if (i >= BLOCK-1 && (i+1-BLOCK) % HOP === 0 && sum/BLOCK >= ABSOLUTE_GATE) blocks.push(sum/BLOCK);
  }
  if (!blocks.length) return {lufs: null, peak};
  const gate = blocks.reduce((a,b) => a+b, 0) / blocks.length / 10;
  let total = 0, count = 0;
  for (const b of blocks) if (b >= gate) {total += b; count++;}
  return {lufs: DB + 10*Math.log10(total/count), peak};
}

export function normalizeInferenceLoudness(samples) {
  const {lufs, peak} = measureLoudness(samples);
  // No valid 400 ms block: no loudness boost; still enforce headroom if over peak.
  const desired = lufs === null ? 1 : 10 ** ((TARGET-lufs) / 20);
  const gain = Math.min(desired, MAX_GAIN, peak ? LIMIT/peak : Infinity);
  return gain === 1 ? samples : Float32Array.from(samples, x => x * gain);
}
