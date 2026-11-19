import { WebMelProcessor, decodeAudioTo16kMono, splitPcmInto5sChunks } from "./mel-frontend.mjs";
import {
  buildTokenLookup,
  decodeChunkTokens,
  encodeNotesToMidi,
  encodeTiePrologue,
  finalizeOpenNotes,
} from "./midi-decoder.mjs";

function argmaxMasked(logitsRow, maskedSet) {
  let bestId = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < logitsRow.length; i++) {
    if (maskedSet.has(i)) continue;
    if (logitsRow[i] > bestScore) {
      bestScore = logitsRow[i];
      bestId = i;
    }
  }
  return bestId;
}

function buildCausalMask(seqLen, pastLen) {
  const total = pastLen + seqLen;
  const mask = new Float32Array(seqLen * total);
  for (let r = 0; r < seqLen; r++) {
    const maxCol = pastLen + r;
    const rowOff = r * total;
    for (let c = 0; c < total; c++) {
      mask[rowOff + c] = c <= maxCol ? 0.0 : -1e9;
    }
  }
  return mask;
}

export class WebScoreClient {
  constructor(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch, gpuKvCache = false, cacheEmbeddings = false, fusedGreedy = false, chunkedFeedback = false, feedbackReadInterval = 8, feedbackHostShapes = false, feedbackHalfKv = false }) {
    this.ort = ort;
    this.lookup = buildTokenLookup(tokenizer);
    this.encoderSession = encoderSession;
    this.embedSession = embedSession;
    this.decoderSession = decoderSession;
    this.arch = arch;
    this.gpuKvCache = gpuKvCache;
    this.cacheEmbeddings = cacheEmbeddings;
    this.fusedGreedy = fusedGreedy;
    this.chunkedFeedback = chunkedFeedback;
    this.feedbackReadInterval = feedbackReadInterval;
    this.feedbackHostShapes = feedbackHostShapes;
    this.feedbackHalfKv = feedbackHalfKv;
    this.embeddingTable = null;
    this.disposed = false;
    this.transcribing = false;
    this.mel = new WebMelProcessor();
    this.maskedIds = new Set([this.lookup.padId, this.lookup.bosId, this.lookup.unkId]);
  }

  static async create(ort, { baseUrl = "/models", useInt8 = true, modelUrls = null, executionProviders = ["webgpu", "wasm"], gpuKvCache = false, cacheEmbeddings = false, fusedGreedy = false, chunkedFeedback = false, feedbackReadInterval = 8, feedbackHostShapes = false, feedbackHalfKv = false } = {}) {
    if (!Array.isArray(executionProviders) || !executionProviders.length) throw new Error("Specify at least one execution provider");
    if (gpuKvCache && (executionProviders[0] !== "webgpu" || useInt8)) {
      throw new Error("GPU KV cache is experimental: select WebGPU first and a floating-point decoder");
    }
    if (fusedGreedy && useInt8) throw new Error("Fused greedy graph requires a separate floating-point model");
    if (chunkedFeedback && !fusedGreedy) throw new Error("Chunked feedback requires fusedGreedy");
    if (!Number.isInteger(feedbackReadInterval) || feedbackReadInterval < 1 || feedbackReadInterval > 64) throw new Error("feedbackReadInterval must be 1..64");
    if (feedbackHostShapes && !chunkedFeedback) throw new Error("Host shapes require chunked feedback");
    if (feedbackHalfKv && !feedbackHostShapes) throw new Error("FP16 KV requires host-shape feedback graph");
    const decFile = feedbackHalfKv ? "decoder_feedback_hostshapes_f16kv.onnx" : feedbackHostShapes ? "decoder_feedback_hostshapes.onnx" : chunkedFeedback ? "decoder_feedback.onnx" : fusedGreedy ? "decoder_greedy.onnx" : useInt8 ? "decoder_kv.int8.onnx" : "decoder_kv.onnx";
    const names = ["tokenizer.json", "config.json", "mel_encoder.onnx", ...(!fusedGreedy ? ["token_embed.onnx"] : []), decFile];
    // Private buckets supply independently signed URLs. Never append paths to
    // a signed URL or silently fall back to an unauthenticated public URL.
    let urls;
    if (modelUrls !== null) {
      urls = Object.fromEntries(names.map((name) => {
        const url = modelUrls?.[name];
        if (typeof url !== "string" || !url.trim()) {
          throw new Error(`Missing model URL for ${name}`);
        }
        return [name, url];
      }));
    } else {
      if (typeof baseUrl !== "string" || !baseUrl.trim() || /[?#]/.test(baseUrl)) {
        throw new Error("baseUrl must be a directory URL without query or fragment; use modelUrls for signed URLs.");
      }
      const cleanBase = baseUrl.replace(/\/+$/, "");
      urls = Object.fromEntries(names.map((name) => [name, `${cleanBase}/${name}`]));
    }
    // Never propagate provider/network errors containing signed bearer URLs.
    // Deliberately omit cause as it can also be serialized by application logs.
    const readMetadata = async (name) => {
      let response;
      try {
        response = await fetch(urls[name]);
      } catch {
        throw new Error(`Failed to fetch ${name}: network request failed`);
      }
      if (!response.ok) throw new Error(`Failed to fetch ${name}: ${response.status}`);
      try {
        return await response.json();
      } catch {
        throw new Error(`Invalid model metadata: ${name}`);
      }
    };
    const [tokenizer, arch] = await Promise.all([
      readMetadata("tokenizer.json"), readMetadata("config.json"),
    ]);
    const opts = { executionProviders };
    const loadSession = async (name) => {
      try {
        const sessionOpts = gpuKvCache && name === decFile
          ? { ...opts, ...(chunkedFeedback ? { extra: { "ep.webgpuexecutionprovider.enableInt64": "1" } } : {}), preferredOutputLocation: { ...(chunkedFeedback ? { next_ids: "gpu-buffer", updated_history: "gpu-buffer" } : { [fusedGreedy ? "next_token" : "logits"]: "cpu" }), present_k: "gpu-buffer", present_v: "gpu-buffer" } }
          : gpuKvCache && fusedGreedy && name === "mel_encoder.onnx"
            ? { ...opts, preferredOutputLocation: { cond_embeds: "gpu-buffer" } } : opts;
        return await ort.InferenceSession.create(urls[name], sessionOpts);
      } catch {
        throw new Error(`Failed to load ${name}: check access, URL expiry and model compatibility`);
      }
    };
    const settled = await Promise.allSettled([
      loadSession("mel_encoder.onnx"), fusedGreedy ? Promise.resolve(null) : loadSession("token_embed.onnx"), loadSession(decFile),
    ]);
    const failed = settled.find(r => r.status === "rejected");
    if (failed) {
      await Promise.allSettled(settled.filter(r => r.status === "fulfilled").map(r => r.value?.release?.()));
      throw failed.reason;
    }
    const [encoderSession, embedSession, decoderSession] = settled.map(r => r.value);

    try {
      return new WebScoreClient(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch, gpuKvCache, cacheEmbeddings, fusedGreedy, chunkedFeedback, feedbackReadInterval, feedbackHostShapes, feedbackHalfKv });
    } catch {
      await Promise.allSettled([encoderSession, embedSession, decoderSession].map(s => s?.release?.()));
      throw new Error("Invalid model metadata or interface");
    }
  }

  async embedTokenIds(ids) {
    if (!this.cacheEmbeddings) {
      const input_ids = new this.ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
      return (await this.embedSession.run({ input_ids })).token_embeds;
    }
    if (!this.embeddingTable) {
      const n = this.lookup.idToEvent.length;
      const input_ids = new this.ort.Tensor("int64", BigInt64Array.from({ length: n }, (_, i) => BigInt(i)), [1, n]);
      const table = (await this.embedSession.run({ input_ids })).token_embeds;
      if (table.dims[1] !== n || table.dims[2] !== this.arch.dim) throw new Error("Invalid embedding table shape");
      this.embeddingTable = new Float32Array(table.data);
    }
    const dim = this.arch.dim;
    const data = new Float32Array(ids.length * dim);
    ids.forEach((id, i) => data.set(this.embeddingTable.subarray(id * dim, (id + 1) * dim), i * dim));
    return new this.ort.Tensor("float32", data, [1, ids.length, dim]);
  }

  async transcribeChunk(pcm5s, { promptTokens = [], maxGenLen = 1024 } = {}) {
    if (this.disposed) throw new Error("Client has been disposed");
    const { data: melData, dims: melDims } = this.mel.computeChunkLogMel(pcm5s);
    const melTensor = new this.ort.Tensor("float32", melData, melDims);
    const encOut = await this.encoderSession.run({ mel: melTensor });
    const condEmbeds = encOut.cond_embeds;
    if (this.chunkedFeedback) return this._transcribeFeedback(condEmbeds, { promptTokens, maxGenLen });
    if (this.fusedGreedy) return this._transcribeFusedGreedy(condEmbeds, { promptTokens, maxGenLen });

    const seqIds = [this.lookup.bosId, ...promptTokens];
    const tokEmbeds = await this.embedTokenIds(seqIds);

    const condLen = condEmbeds.dims[1];
    const hiddenDim = condEmbeds.dims[2];
    const prefillLen = condLen + tokEmbeds.dims[1];
    const prefillData = new Float32Array(prefillLen * hiddenDim);
    prefillData.set(condEmbeds.data, 0);
    prefillData.set(tokEmbeds.data, condLen * hiddenDim);

    const headDim = Math.floor(this.arch.dim / this.arch.num_heads);
    let pastK = new this.ort.Tensor(
      "float32",
      new Float32Array(0),
      [this.arch.num_layers, 1, this.arch.num_kv_heads, 0, headDim]
    );
    let pastV = new this.ort.Tensor(
      "float32",
      new Float32Array(0),
      [this.arch.num_layers, 1, this.arch.num_kv_heads, 0, headDim]
    );

    let stepEmbeds = new this.ort.Tensor("float32", prefillData, [1, prefillLen, hiddenDim]);
    let posOffset = 0;
    const generated = [...promptTokens];

    try {
      for (let step = 0; step < maxGenLen; step++) {
        const seqLen = stepEmbeds.dims[1];
        const pastLen = pastK.dims[3];
        const maskTensor = new this.ort.Tensor("float32", buildCausalMask(seqLen, pastLen), [1, 1, seqLen, pastLen + seqLen]);
        const posTensor = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(posOffset)]), [1]);

        const decOut = await this.decoderSession.run({
          inputs_embeds: stepEmbeds,
          past_k: pastK,
          past_v: pastV,
          position_offset: posTensor,
          attn_mask: maskTensor,
        });

        if (this.gpuKvCache) {
          for (const old of [pastK, pastV]) if (old.location === "gpu-buffer") old.dispose();
        }
        pastK = decOut.present_k;
        pastV = decOut.present_v;
        posOffset += seqLen;

        const vocabSize = this.lookup.idToEvent.length;
        const logitsArr = decOut.logits.data;
        const logitRows = decOut.logits.dims[1];
        if (logitRows !== 1 && logitRows !== seqLen) throw new Error("Invalid decoder logits shape");
        const lastRow = logitsArr.subarray((logitRows - 1) * vocabSize, logitRows * vocabSize);
        const nextTok = argmaxMasked(lastRow, this.maskedIds);
        generated.push(nextTok);

        if (nextTok === this.lookup.eosId) return generated;

        stepEmbeds = await this.embedTokenIds([nextTok]);
      }

      throw new Error("Model did not emit EOS within the per-chunk token limit");
    } finally {
      if (this.gpuKvCache) {
        for (const tensor of [pastK, pastV]) if (tensor.location === "gpu-buffer") tensor.dispose();
      }
    }
  }

  async _transcribeFusedGreedy(condEmbeds, { promptTokens, maxGenLen }) {
    // Dedicated graph: embedded token IDs + encoder prefix + greedy selection.
    // No .data read of GPU prefix/KV; only one int64 token is read per step.
    const T = this.ort.Tensor;
    const ids = [this.lookup.bosId, ...promptTokens];
    const headDim = Math.floor(this.arch.dim / this.arch.num_heads);
    const emptyKvShape = [this.arch.num_layers, 1, this.arch.num_kv_heads, 0, headDim];
    let pastK = new T('float32', new Float32Array(0), emptyKvShape);
    let pastV = new T('float32', new Float32Array(0), emptyKvShape);
    let prefix = condEmbeds;
    let inputIds = new T('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const emptyPrefix = new T('float32', new Float32Array(0), [1, 0, this.arch.dim]);
    const nextIdData = new BigInt64Array(1);
    const nextIds = new T('int64', nextIdData, [1, 1]);
    const posData = new BigInt64Array(1);
    const position = new T('int64', posData, [1]);
    const prefixLength = condEmbeds.dims[1] + ids.length;
    // One-token causal rows are all zero. Reuse a single backing allocation.
    const zeroMask = new Float32Array(prefixLength + maxGenLen);
    const generated = [...promptTokens];
    let posOffset = 0;
    const release = t => { if (t?.location === 'gpu-buffer') t.dispose(); };
    try {
      for (let step = 0; step < maxGenLen; step++) {
        const seqLen = prefix.dims[1] + inputIds.dims[1];
        const pastLen = pastK.dims[3];
        const mask = new T('float32', step === 0 ? buildCausalMask(seqLen, pastLen) : zeroMask.subarray(0, pastLen + 1), [1, 1, seqLen, pastLen + seqLen]);
        posData[0] = BigInt(posOffset);
        const out = await this.decoderSession.run({ prefix_embeds: prefix, input_ids: inputIds,
          past_k: pastK, past_v: pastV, position_offset: position, attn_mask: mask });
        const oldK = pastK, oldV = pastV;
        pastK = out.present_k; pastV = out.present_v;
        // Never dispose an aliased output buffer before its next use.
        if (oldK !== pastK && oldK !== pastV) release(oldK);
        if (oldV !== pastK && oldV !== pastV && oldV !== oldK) release(oldV);
        if (prefix !== emptyPrefix) { release(prefix); prefix = emptyPrefix; }
        if (this.gpuKvCache && (pastK?.location !== 'gpu-buffer' || pastV?.location !== 'gpu-buffer')) throw new Error('Requested GPU-resident KV outputs are not on GPU');
        posOffset += seqLen;
        if (out.next_token?.dims.length !== 1 || out.next_token.dims[0] !== 1) throw new Error('Invalid greedy token output');
        const nextTok = Number(out.next_token.data[0]);
        if (!Number.isInteger(nextTok) || nextTok < 0 || nextTok >= this.lookup.idToEvent.length || this.maskedIds.has(nextTok)) throw new Error('Invalid greedy token ID');
        generated.push(nextTok);
        if (nextTok === this.lookup.eosId) return generated;
        nextIdData[0] = BigInt(nextTok); inputIds = nextIds;
      }
      throw new Error('Model did not emit EOS within the per-chunk token limit');
    } finally {
      release(pastK); if (pastV !== pastK) release(pastV);
      if (prefix !== emptyPrefix) release(prefix);
    }
  }

  async _transcribeFeedback(condEmbeds, { promptTokens, maxGenLen }) {
    // Explicit prototype: no per-token host readback. Still one ORT run per token.
    const T = this.ort.Tensor;
    const release = t => { if (t?.location === 'gpu-buffer') t.dispose(); };
    const emptyPrefix = new T('float32', new Float32Array(0), [1, 0, this.arch.dim]);
    const shape = [this.arch.num_layers, 1, this.arch.num_kv_heads, 0, this.arch.dim / this.arch.num_heads];
    const kvType = this.feedbackHalfKv ? 'float16' : 'float32';
    const KvArray = this.feedbackHalfKv ? (globalThis.Float16Array ?? Uint16Array) : Float32Array;
    let k = new T(kvType, new KvArray(0), shape), v = new T(kvType, new KvArray(0), shape);
    let ids = new T('int64', BigInt64Array.from([this.lookup.bosId, ...promptTokens].map(BigInt)), [1, 1 + promptTokens.length]);
    let history = new T('int64', new BigInt64Array(0), [1, 0]);
    let prefix = condEmbeds, position = 0, pending = 0;
    const generated = [...promptTokens];
    const zeros = new Float32Array(condEmbeds.dims[1] + ids.dims[1] + maxGenLen);
    try {
      for (let step = 0; step < maxGenLen; step++) {
        const seq = prefix.dims[1] + ids.dims[1], past = k.dims[3];
        const mask = new T('float32', step === 0 ? buildCausalMask(seq, past) : zeros.subarray(0, past + 1), [1, 1, seq, past + seq]);
        const hostShapes = this.feedbackHostShapes ? {
          shape_q: new T('int64', BigInt64Array.from([1, seq, this.arch.num_heads, this.arch.dim / this.arch.num_heads].map(BigInt)), [4]),
          shape_kv: new T('int64', BigInt64Array.from([1, seq, this.arch.num_kv_heads, this.arch.dim / this.arch.num_heads].map(BigInt)), [4]),
          shape_expand: new T('int64', BigInt64Array.from([1, this.arch.num_kv_heads, this.arch.num_heads / this.arch.num_kv_heads, past + seq, this.arch.dim / this.arch.num_heads].map(BigInt)), [5]),
          shape_expanded_kv: new T('int64', BigInt64Array.from([1, this.arch.num_heads, past + seq, this.arch.dim / this.arch.num_heads].map(BigInt)), [4]),
          shape_out: new T('int64', BigInt64Array.from([1, seq, this.arch.dim].map(BigInt)), [3]),
          position_range: new T('float32', Float32Array.from({length: seq}, (_, i) => i), [seq]),
        } : {};
        const out = await this.decoderSession.run({ ...hostShapes, prefix_embeds: prefix, input_ids: ids, token_history: history,
          past_k: k, past_v: v, position_offset: new T('int64', BigInt64Array.of(BigInt(position)), [1]), attn_mask: mask });
        const old = new Set([k, v, ids, history, prefix]);
        k = out.present_k; v = out.present_v; ids = out.next_ids; history = out.updated_history; prefix = emptyPrefix;
        const keep = new Set([k, v, ids, history, prefix]);
        for (const tensor of old) if (!keep.has(tensor)) release(tensor);
        if (!ids || !history || ids.dims.length !== 2 || ids.dims[0] !== 1 || ids.dims[1] !== 1 || history.dims.length !== 2 || history.dims[0] !== 1 || history.dims[1] !== pending + 1) throw new Error('Invalid feedback output shape');
        if (this.feedbackHalfKv && (k?.type !== 'float16' || v?.type !== 'float16')) throw new Error('Expected FP16 feedback KV outputs');
        if (this.gpuKvCache && [k, v, ids, history].some(t => t?.location !== 'gpu-buffer')) throw new Error('Requested feedback outputs are not on GPU');
        position += seq; pending++;
        if ((step + 1) % this.feedbackReadInterval === 0 || step + 1 === maxGenLen) {
          const data = history.location === 'gpu-buffer' ? await history.getData(true) : history.data;
          for (let j = 0; j < data.length; j++) {
            const token = Number(data[j]);
            if (!Number.isInteger(token) || token < 0 || token >= this.lookup.idToEvent.length || this.maskedIds.has(token)) throw new Error('Invalid feedback token ID');
            generated.push(token);
            // Up to interval-1 speculative tokens after EOS are intentionally discarded.
            if (token === this.lookup.eosId) return generated;
          }
          // getData converts ORT tensor to CPU. Reset history instead of uploading it.
          history = new T('int64', new BigInt64Array(0), [1, 0]); pending = 0;
        }
      }
      throw new Error('Model did not emit EOS within the per-chunk token limit');
    } finally {
      for (const tensor of new Set([k, v, ids, history, prefix])) release(tensor);
    }
  }

  async transcribePcm(pcm16k, { maxGenLen = 1024, onProgress = null } = {}) {
    if (this.disposed) throw new Error("Client has been disposed");
    if (this.transcribing) throw new Error("A transcription is already running");
    this.transcribing = true;
    try {
      const chunks = splitPcmInto5sChunks(pcm16k);
      const allNotes = [];
      const tokensByChunk = [];
      let openNotes = [];

      for (let idx = 0; idx < chunks.length; idx++) {
        const { pcm, startSec } = chunks[idx];
        const promptTokens = idx > 0 ? encodeTiePrologue(openNotes, this.lookup) : [];
        const tokens = await this.transcribeChunk(pcm, { promptTokens, maxGenLen });
        tokensByChunk.push(tokens);

        const decoded = decodeChunkTokens(tokens, this.lookup, { chunkStartSec: startSec, openNotes });
        allNotes.push(...decoded.completed);
        openNotes = decoded.nextOpenNotes;
        if (onProgress) onProgress({ completedChunks: idx + 1, totalChunks: chunks.length });
      }

      const endSec = pcm16k.length / 16000;
      allNotes.push(...finalizeOpenNotes(openNotes, endSec));
      // The final encoder input is padded to five seconds, but output must not be.
      const notes = allNotes.filter(n => n.onset >= 0 && n.onset < endSec)
        .map(n => ({ ...n, offset: Math.min(n.offset, endSec) }));
      const midiBytes = encodeNotesToMidi(notes);
      return { notes, midiBytes, tokensByChunk };
    } finally { this.transcribing = false; }
  }

  async dispose() {
    if (this.disposed) return;
    if (this.transcribing) throw new Error("Cannot dispose during transcription");
    this.disposed = true;
    this.embeddingTable = null;
    const results = await Promise.allSettled([this.encoderSession, this.embedSession, this.decoderSession].map(s => s?.release?.()));
    if (results.some(r => r.status === "rejected")) throw new Error("One or more model sessions could not be released");
  }

  async transcribeAudio(audioInput, opts = {}) {
    const pcm16k = await decodeAudioTo16kMono(audioInput);
    return this.transcribePcm(pcm16k, opts);
  }
}
