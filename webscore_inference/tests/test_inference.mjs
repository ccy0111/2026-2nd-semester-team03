import assert from "node:assert/strict";
import test from "node:test";

import {
  WebMelProcessor,
  downmixToMono,
  resampleLinear,
  splitPcmInto5sChunks,
} from "../web/mel-frontend.mjs";
import {
  buildTokenLookup,
  decodeChunkTokens,
  encodeNotesToMidi,
  encodeTiePrologue,
  finalizeOpenNotes,
} from "../web/midi-decoder.mjs";
import { WebScoreClient } from "../web/model-client.mjs";

function makeFakeTokenizer() {
  const tokens = ["<pad>", "<bos>", "<eos>", "<unk>"];
  for (let i = 0; i < 128; i++) tokens.push(`drum:${i}`);
  for (let i = 0; i < 130; i++) tokens.push(`inst:${i}`);
  tokens.push("vel:0", "vel:1", "tie");
  for (let i = 0; i < 128; i++) tokens.push(`pitch:${i}`);
  for (let i = 0; i <= 500; i++) tokens.push(`time:${i}`);
  return {
    vocab_size: tokens.length,
    pad_id: 0,
    bos_id: 1,
    eos_id: 2,
    unk_id: 3,
    id_to_token: tokens,
  };
}

test("mel processor outputs [1, 501, 512] finite log-mel", () => {
  const mel = new WebMelProcessor();
  const pcm = new Float32Array(80000);
  for (let i = 0; i < pcm.length; i++) {
    pcm[i] = 0.25 * Math.sin((2.0 * Math.PI * 440.0 * i) / 16000.0);
  }
  const res = mel.computeChunkLogMel(pcm);
  assert.deepEqual(res.dims, [1, 501, 512]);
  assert.equal(res.data.length, 501 * 512);
  for (let i = 0; i < res.data.length; i++) {
    assert.ok(Number.isFinite(res.data[i]));
  }
});

test("splitPcmInto5sChunks pads final partial chunk cleanly", () => {
  const chunks = splitPcmInto5sChunks(new Float32Array(100000));
  assert.equal(chunks.length, 2);
  assert.equal(chunks[0].startSec, 0);
  assert.equal(chunks[1].startSec, 5);
  assert.equal(chunks[1].pcm.length, 80000);
});

test("decodeChunkTokens handles tie continuity and encodes valid SMF MIDI", () => {
  const tok = makeFakeTokenizer();
  const lookup = buildTokenLookup(tok);
  const id = (k) => lookup.eventToId.get(k);

  const chunk0Tokens = [
    id("time:10"), id("inst:0"), id("vel:1"), id("pitch:60"),
    id("time:50"), id("drum:36"),
    lookup.eosId,
  ];
  const d0 = decodeChunkTokens(chunk0Tokens, lookup, { chunkStartSec: 0, openNotes: [] });
  assert.equal(d0.nextOpenNotes.length, 1);
  assert.equal(d0.completed.length, 1);

  const prologue = encodeTiePrologue(d0.nextOpenNotes, lookup);
  const chunk1Tokens = [
    ...prologue,
    id("time:20"), id("inst:0"), id("vel:0"), id("pitch:60"),
    lookup.eosId,
  ];
  const d1 = decodeChunkTokens(chunk1Tokens, lookup, {
    chunkStartSec: 5.0,
    openNotes: d0.nextOpenNotes,
  });
  const all = [...d0.completed, ...d1.completed, ...finalizeOpenNotes(d1.nextOpenNotes, 10.0)];
  assert.equal(all.length, 2);
  const piano = all.find((n) => !n.isDrum);
  assert.ok(Math.abs(piano.onset - 0.1) < 1e-6);
  assert.ok(Math.abs(piano.offset - 5.2) < 1e-6);

  const midi = encodeNotesToMidi(all);
  assert.equal(String.fromCharCode(...midi.slice(0, 4)), "MThd");
  assert.equal(String.fromCharCode(...midi.slice(14, 18)), "MTrk");
});

test("WebScoreClient executes chunk inference loop with mock ONNX runtime", async () => {
  const tok = makeFakeTokenizer();
  const lookup = buildTokenLookup(tok);
  const dim = 8;
  const layers = 2;
  const kvHeads = 2;
  const headDim = 2;

  class MockTensor {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  }

  let decodeCalls = 0;
  const sequenceToEmit = [
    lookup.eventToId.get("time:10"),
    lookup.eventToId.get("drum:36"),
    lookup.eosId,
  ];

  const client = new WebScoreClient(
    { Tensor: MockTensor },
    {
      tokenizer: tok,
      arch: { dim, num_layers: layers, num_heads: 4, num_kv_heads: kvHeads },
      encoderSession: {
        async run({ mel }) {
          return { cond_embeds: new MockTensor("float32", new Float32Array(mel.dims[1] * dim), [1, mel.dims[1], dim]) };
        },
      },
      embedSession: {
        async run({ input_ids }) {
          return { token_embeds: new MockTensor("float32", new Float32Array(input_ids.dims[1] * dim), [1, input_ids.dims[1], dim]) };
        },
      },
      decoderSession: {
        async run({ inputs_embeds, past_k }) {
          const seqLen = inputs_embeds.dims[1];
          const pastLen = past_k.dims[3];
          const totalLen = pastLen + seqLen;
          const logits = new Float32Array(seqLen * tok.vocab_size).fill(-10);
          const emitId = sequenceToEmit[decodeCalls++];
          logits[(seqLen - 1) * tok.vocab_size + emitId] = 10.0;
          const kvLen = layers * kvHeads * totalLen * headDim;
          return {
            logits: new MockTensor("float32", logits, [1, seqLen, tok.vocab_size]),
            present_k: new MockTensor("float32", new Float32Array(kvLen), [layers, 1, kvHeads, totalLen, headDim]),
            present_v: new MockTensor("float32", new Float32Array(kvLen), [layers, 1, kvHeads, totalLen, headDim]),
          };
        },
      },
    }
  );

  const res = await client.transcribeAudio(new Float32Array(80000));
  assert.equal(decodeCalls, 3);
  assert.equal(res.notes.length, 1);
  assert.equal(res.notes[0].isDrum, true);
  assert.equal(String.fromCharCode(...res.midiBytes.slice(0, 4)), "MThd");
});

test("downmixToMono and resampleLinear convert stereo 48kHz to mono 16kHz", () => {
  const left = new Float32Array(480).fill(0.4);
  const right = new Float32Array(480).fill(0.2);
  const mono = downmixToMono([left, right]);
  const resampled = resampleLinear(mono, 48000, 16000);
  assert.equal(resampled.length, 160);
  assert.ok(Math.abs(resampled[0] - 0.3) < 1e-5);
});
