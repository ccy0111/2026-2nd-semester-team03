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


test("MIDI export rejects excess melodic instruments rather than dropping notes", () => {
  const notes = Array.from({ length: 16 }, (_, program) => ({
    program, pitch: 60, onset: 0, offset: 1, isDrum: false,
  }));
  assert.throws(() => encodeNotesToMidi(notes), /at most 15 melodic instruments/);
  const supported = [...notes.slice(0, 15), { program: 128, pitch: 36, onset: 0, offset: 0.1, isDrum: true }];
  const midi = encodeNotesToMidi(supported);
  assert.equal(new DataView(midi.buffer).getUint16(10), 17); // tempo + all 16 tracks
});

test("model loading supports base directories and private per-file signed URLs", async (t) => {
  const tok = makeFakeTokenizer();
  const config = { dim: 8, num_layers: 2, num_heads: 4, num_kv_heads: 2 };
  const requests = [];
  const sessions = [];
  t.mock.method(globalThis, "fetch", async (url) => {
    requests.push(url);
    return { ok: true, json: async () => url.includes("tokenizer.json") ? tok : config };
  });
  const ort = { InferenceSession: { create: async (url, options) => {
    sessions.push({ url, options });
    return {};
  } } };
  await WebScoreClient.create(ort, { baseUrl: "https://models.example.test/piano/step-36000///" });
  assert.deepEqual(requests, [
    "https://models.example.test/piano/step-36000/tokenizer.json",
    "https://models.example.test/piano/step-36000/config.json",
  ]);
  assert.deepEqual(sessions.map((s) => s.url), [
    "https://models.example.test/piano/step-36000/mel_encoder.onnx",
    "https://models.example.test/piano/step-36000/token_embed.onnx",
    "https://models.example.test/piano/step-36000/decoder_kv.int8.onnx",
  ]);
  requests.length = sessions.length = 0;
  const names = ["tokenizer.json", "config.json", "mel_encoder.onnx", "token_embed.onnx", "decoder_kv.onnx"];
  const modelUrls = Object.fromEntries(names.map((name, i) => [name,
    `https://private.example.test/bucket/${name}?X-Amz-Signature=test%2B${i}&X-Amz-Expires=900`]));
  await WebScoreClient.create(ort, { modelUrls, useInt8: false });
  assert.deepEqual(requests, names.slice(0, 2).map((n) => modelUrls[n]));
  assert.deepEqual(sessions.map((s) => s.url), names.slice(2).map((n) => modelUrls[n]));
  assert.ok(sessions.every((s) => s.options.executionProviders.join() === "webgpu,wasm"));
});

test("missing signed URLs fail before network access without public fallback", async (t) => {
  t.mock.method(globalThis, "fetch", () => { throw new Error("Unexpected network request"); });
  await assert.rejects(WebScoreClient.create({}, { modelUrls: {
    "tokenizer.json": "https://private.example.test/tokenizer.json?signature=test",
  } }), /Missing model URL for config.json/);
  await assert.rejects(WebScoreClient.create({}, { baseUrl: "https://private.example.test/model?signature=test" }), /use modelUrls/);
  await assert.rejects(WebScoreClient.create({}, { baseUrl: "" }), /baseUrl must/);
});

test("expired metadata URL reports status without echoing the signature or loading weights", async (t) => {
  t.mock.method(globalThis, "fetch", async () => ({ ok: false, status: 403 }));
  const names = ["tokenizer.json", "config.json", "mel_encoder.onnx", "token_embed.onnx", "decoder_kv.int8.onnx"];
  const modelUrls = Object.fromEntries(names.map((n) => [n, `https://private.example.test/${n}?signature=secret-test`]));
  await assert.rejects(WebScoreClient.create({}, { modelUrls }), (err) => {
    assert.equal(err.message, "Failed to fetch tokenizer.json: 403");
    assert.ok(!err.message.includes("secret-test"));
    return true;
  });
});

test("network, parsing and runtime errors cannot escape with signed URLs", async (t) => {
  const secret = "https://private.example.test/model?X-Amz-Signature=secret-test";
  const names = ["tokenizer.json", "config.json", "mel_encoder.onnx", "token_embed.onnx", "decoder_kv.int8.onnx"];
  const modelUrls = Object.fromEntries(names.map((n) => [n, secret]));
  const assertSafe = (expected) => (err) => {
    assert.match(err.message, expected);
    assert.ok(!String(err.stack).includes("secret-test"));
    assert.equal(err.cause, undefined);
    return true;
  };
  const fetchMock = t.mock.method(globalThis, "fetch", async () => { throw new Error(secret); });
  await assert.rejects(WebScoreClient.create({}, { modelUrls }), assertSafe(/network request failed/));
  fetchMock.mock.mockImplementation(async () => ({ ok: true, json: async () => { throw new Error(secret); } }));
  await assert.rejects(WebScoreClient.create({}, { modelUrls }), assertSafe(/Invalid model metadata/));
  fetchMock.mock.mockImplementation(async () => ({ ok: true, json: async () => ({}) }));
  const ort = { InferenceSession: { create: async () => { throw new Error(secret, { cause: new Error(secret) }); } } };
  await assert.rejects(WebScoreClient.create(ort, { modelUrls }), assertSafe(/Failed to load mel_encoder.onnx/));
});

test("last partial chunk drops padding notes and clips sustained notes to audio end", async () => {
  const tok = makeFakeTokenizer(), lookup = buildTokenLookup(tok);
  const id = x => lookup.eventToId.get(x);
  const client = new WebScoreClient({}, {tokenizer:tok, arch:{}});
  client.transcribeChunk = async () => [id('inst:0'),id('vel:1'),id('time:50'),id('pitch:60'),id('time:200'),id('pitch:64'),lookup.eosId];
  const r = await client.transcribePcm(new Float32Array(16000));
  assert.equal(r.notes.length, 1);
  assert.equal(r.notes[0].pitch, 60);
  assert.equal(r.notes[0].offset, 1);
});

test("token limit without EOS is an explicit error", async () => {
 const tok=makeFakeTokenizer();
 class Tensor {constructor(type,data,dims){Object.assign(this,{type,data,dims});}}
 const client=new WebScoreClient({Tensor},{tokenizer:tok,arch:{dim:4,num_layers:1,num_heads:1,num_kv_heads:1},
  encoderSession:{run:async()=>({cond_embeds:new Tensor('float32',new Float32Array(4),[1,1,4])})},
  embedSession:{run:async()=>({token_embeds:new Tensor('float32',new Float32Array(4),[1,1,4])})},
  decoderSession:{run:async({inputs_embeds})=>{const n=inputs_embeds.dims[1];const logits=new Float32Array(n*tok.vocab_size).fill(-10);logits[(n-1)*tok.vocab_size+4]=10;return {logits:new Tensor('float32',logits,[1,n,tok.vocab_size]),present_k:new Tensor('float32',new Float32Array(4),[1,1,1,1,4]),present_v:new Tensor('float32',new Float32Array(4),[1,1,1,1,4])};}}
 });
 await assert.rejects(client.transcribeChunk(new Float32Array(80000),{maxGenLen:1}),/EOS/);
});
