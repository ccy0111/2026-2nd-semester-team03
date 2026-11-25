import test from "node:test";
import assert from "node:assert/strict";
import { WebScoreClient } from "../web/model-client.mjs";

const tokenizer = {
  pad_id: 0,
  bos_id: 1,
  eos_id: 2,
  unk_id: 3,
  id_to_token: [
    "<pad>", "<bos>", "<eos>", "<unk>", "tie",
    "inst:0", "pitch:60", "time:0", "vel:1", "time:100", "vel:0", "inst:1",
  ],
};

const chunk0Seq = [7, 5, 8, 6, 9, 10, 6, 2];
const chunk1Seq = [7, 5, 8, 6, 9, 10, 6, 2];
const audio2Chunks = new Float32Array(16000 * 10);
const audio6Chunks = new Float32Array(16000 * 30);

const ort = {
  Tensor: class {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  },
};

function makeClient({
  nativeImpl,
  batchArtifact = true,
  gpuBatchWidth = 8,
  gpuKvBindingBytes = 128 * 1024 * 1024,
} = {}) {
  let nativeCalls = 0;
  let nativeReleases = 0;
  let fallbackLoads = 0;
  let stdRuns = 0;
  let stdReleases = 0;

  const nativeSession = {
    nativeWebGpuDecoder: true,
    nativeMaxBatch: 8,
    projectionFusion: true,
    inputMetadata: [
      { name: "inputs_embeds", shape: ["batch", "seq_len", 2] },
      { name: "past_k", shape: [1, "batch", 1, "past", 2] },
      { name: "past_v", shape: [1, "batch", 1, "past", 2] },
    ],
    async generateBlockNative(args) {
      nativeCalls++;
      return nativeImpl(args, nativeCalls);
    },
    async run() {
      throw new Error("Native WebGPU decoder requires generateBlockNative");
    },
    async release() {
      nativeReleases++;
    },
  };

  let stdStep = 0;
  const stdSession = {
    inputMetadata: [
      { name: "inputs_embeds", shape: ["batch", "seq_len", 2] },
      { name: "past_k", shape: [1, "batch", 1, "past", 2] },
      { name: "past_v", shape: [1, "batch", 1, "past", 2] },
    ],
    async run(feed) {
      stdRuns++;
      const [b, s] = feed.inputs_embeds.dims;
      const past = feed.past_k.dims[3];
      if (past === 0) stdStep = 0;
      const tokId = chunk0Seq[stdStep++] ?? 2;
      const V = tokenizer.id_to_token.length;
      const logitsData = new Float32Array(b * s * V).fill(-10);
      for (let bi = 0; bi < b; bi++) {
        logitsData[((bi * s) + (s - 1)) * V + tokId] = 10;
      }
      return {
        present_k: { dims: [1, b, 1, past + s, 2], location: "gpu-buffer", dispose() {} },
        present_v: { dims: [1, b, 1, past + s, 2], location: "gpu-buffer", dispose() {} },
        logits: { dims: [b, s, V], data: logitsData, location: "cpu", dispose() {} },
      };
    },
    async release() {
      stdReleases++;
    },
  };

  const c = new WebScoreClient(ort, {
    tokenizer,
    arch: { dim: 2, num_heads: 1, num_kv_heads: 1, num_layers: 1 },
    gpuKvCache: true,
    cacheEmbeddings: true,
    decoderSession: nativeSession,
  });

  c.mel = {
    computeChunkLogMel() {
      return { data: new Float32Array(3), dims: [1, 3] };
    },
  };
  c.encoderSession = {
    inputMetadata: [{ name: "mel", shape: ["batch", 3] }],
    async run({ mel }) {
      const b = mel.dims[0];
      return { cond_embeds: new ort.Tensor("float32", new Float32Array(b * 3 * 2), [b, 3, 2]) };
    },
    async release() {},
  };
  c.embedSession = {
    async run({ input_ids }) {
      return { token_embeds: new ort.Tensor("float32", new Float32Array(input_ids.dims[1] * 2), [1, input_ids.dims[1], 2]) };
    },
    async release() {},
  };
  c.embeddingTable = new Float32Array(tokenizer.id_to_token.length * 2);
  c.tokenSelector = { setMask() {}, async run() { throw new Error("CPU logits used"); }, dispose() {} };
  c.gpuKvBindingBytes = gpuKvBindingBytes;
  c.localLayerKv = true;
  c.localProjectionFusion = true;
  c.gpuSerialFeedbackBlock = 8;
  c.gpuBatchPairs = true;
  c.gpuBatchWidth = gpuBatchWidth;
  c.gpuFeedbackBlock = 0;
  c._batchArtifact = batchArtifact;
  c._fallbackDecoderFactory = async () => {
    fallbackLoads++;
    return stdSession;
  };

  return {
    client: c,
    stats: () => ({ nativeCalls, nativeReleases, fallbackLoads, stdRuns, stdReleases }),
  };
}

test("prelude-forcing falls back to standard FP16 when native decoder throws at runtime", async () => {
  const { client, stats } = makeClient({
    nativeImpl() {
      throw new Error("simulated WebGPU native kernel error");
    },
  });

  const res = await client.transcribePcm(audio2Chunks, { boundaryMode: "prelude-forcing" });
  assert.equal(res.chunkStatuses.length, 2);
  assert.ok(res.chunkStatuses.every((s) => s.status === "ok"));
  assert.equal(res.notes.length, 2);

  const st = stats();
  assert.equal(st.nativeCalls, 1);
  assert.equal(st.nativeReleases, 1);
  assert.equal(st.fallbackLoads, 1);
  assert.ok(st.stdRuns > 0, "standard FP16 decoder must execute after fallback");
  assert.equal(client.decoderSession.nativeWebGpuDecoder, undefined);
  assert.equal(client.localLayerKv, false);
  assert.equal(client.gpuSerialFeedbackBlock, 0);

  await client.dispose();
  assert.equal(stats().stdReleases, 1);
});

test("pitch-tie batched B=8 falls back to standard FP16 when native decoder throws at runtime", async () => {
  const { client, stats } = makeClient({
    gpuBatchWidth: 8,
    batchArtifact: true,
    nativeImpl() {
      throw new Error("simulated batched native kernel error");
    },
  });

  const res = await client.transcribePcm(audio6Chunks, { boundaryMode: "pitch-tie" });
  assert.equal(res.chunkStatuses.length, 6);
  assert.ok(res.chunkStatuses.every((s) => s.status === "ok"));
  assert.equal(res.notes.length, 6);

  const st = stats();
  assert.equal(st.nativeCalls, 1);
  assert.equal(st.nativeReleases, 1);
  assert.equal(st.fallbackLoads, 1);
  assert.ok(st.stdRuns > 0, "standard FP16 decoder must complete batched/serial pitch-tie");
  assert.ok(client.gpuBatchWidth <= 4, "batch width clamped for standard ORT FP16");

  await client.dispose();
});

test("pitch-tie non-batch artifact falls back cleanly to standard FP16 sequential", async () => {
  const { client, stats } = makeClient({
    gpuBatchWidth: 4,
    batchArtifact: false,
    nativeImpl() {
      throw new Error("simulated non-batch native kernel error");
    },
  });

  const res = await client.transcribePcm(audio2Chunks, { boundaryMode: "pitch-tie" });
  assert.equal(res.chunkStatuses.length, 2);
  assert.ok(res.chunkStatuses.every((s) => s.status === "ok"));
  assert.equal(res.notes.length, 2);

  const st = stats();
  assert.equal(st.fallbackLoads, 1);
  assert.equal(client.gpuBatchPairs, false);
  assert.ok(st.stdRuns > 0);
});

test("missing EOS (TokenLimitError) stays on fast native kernel without falling back to ORT FP16", async () => {
  const { client, stats } = makeClient({
    nativeImpl(args, callIdx) {
      // Call 1 (chunk 0 attempt 1): hits maxGenLen without EOS -> TokenLimitError
      // Call 2 (chunk 0 retry): recovers with EOS on the SAME fast native kernel!
      // Call 3 (chunk 1): normal EOS on the SAME fast native kernel!
      if (callIdx === 1) {
        return { tokens: [[7, 5, 8]], done: [false], decoderCalls: 3 };
      }
      return { tokens: [chunk1Seq], done: [true], decoderCalls: chunk1Seq.length };
    },
  });

  const res = await client.transcribePcm(audio2Chunks, { boundaryMode: "prelude-forcing", maxGenLen: 32 });
  assert.equal(res.chunkStatuses.length, 2);
  assert.equal(res.chunkStatuses[0].status, "recovered");
  assert.equal(res.chunkStatuses[1].status, "ok");

  const st = stats();
  assert.equal(st.nativeCalls, 3);
  assert.equal(st.nativeReleases, 0);
  assert.equal(st.fallbackLoads, 0, "EOS limit must NEVER trigger ORT FP16 fallback");
  assert.equal(st.stdRuns, 0);
  assert.equal(client.decoderSession.nativeWebGpuDecoder, true);

  await client.dispose();
  assert.equal(stats().nativeReleases, 1);
});

test("user abort inside native decoder propagates immediately without loading standard FP16", async () => {
  const controller = new AbortController();
  const reason = { why: "user-canceled" };
  const { client, stats } = makeClient({
    nativeImpl() {
      controller.abort(reason);
      throw new Error("aborted mid-block");
    },
  });

  await assert.rejects(
    client.transcribePcm(audio2Chunks, { boundaryMode: "prelude-forcing", signal: controller.signal }),
    (err) => err === reason
  );

  const st = stats();
  assert.equal(st.nativeCalls, 1);
  assert.equal(st.fallbackLoads, 0, "abort must NEVER trigger fallback decoder load");
  assert.equal(st.stdRuns, 0);
});


import {NativeWebGpuDecoder} from "../web/webgpu-native-decoder.mjs";
test("real native guard at 2048 triggers same-chunk FP16 fallback",async()=>{
 const {client,stats}=makeClient({nativeImpl(args){
   return NativeWebGpuDecoder.prototype.generateBlockNative.call({maxBatch:8}, {...args,vocabSize:894});
 }});
 const out=await client.transcribePcm(audio2Chunks,{boundaryMode:"prelude-forcing",maxGenLen:2048});
 assert.equal(out.notes.length,2);
 assert.equal(stats().fallbackLoads,1);
 assert.ok(stats().stdRuns>0);
});
test("cancel during native release does not strand decoder for next request",async()=>{
 const {client}=makeClient({nativeImpl(){throw Error("forced failure")}});
 const ac=new AbortController();
 const original=client.decoderSession.release.bind(client.decoderSession);
 client.decoderSession.release=async()=>{await original();ac.abort(new DOMException("Cancel","AbortError"));};
 await assert.rejects(client.transcribePcm(audio2Chunks,{boundaryMode:"prelude-forcing",signal:ac.signal}),{name:"AbortError"});
 const out=await client.transcribePcm(audio2Chunks,{boundaryMode:"prelude-forcing"});
 assert.equal(out.notes.length,2);
});
