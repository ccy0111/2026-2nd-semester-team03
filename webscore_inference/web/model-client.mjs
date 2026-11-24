import { GpuTokenSelector } from "./gpu-token-selector.mjs";
import { boundedGpuBatchWidth, runBatchPair } from "./gpu-batch-pair.mjs";
import { WebMelProcessor, decodeAudioTo16kMono, splitPcmInto5sChunks } from "./mel-frontend.mjs";
import {
  buildTokenLookup,
  decodeChunkTokens,
  encodeNotesToMidi,
  encodeTiePrologue,
  finalizeOpenNotes,
} from "./midi-decoder.mjs";

// Only this explicit model-generation failure is eligible for chunk recovery.
export class TokenLimitError extends Error {
  constructor() {
    super("Model did not emit EOS within the per-chunk token limit");
    this.name = "TokenLimitError";
    this.code = "TOKEN_LIMIT_EXHAUSTED";
  }
}

function throwIfAborted(signal) {
  if (!signal?.aborted) return;
  if (signal.reason !== undefined) throw signal.reason;
  const error = new Error("Transcription aborted");
  error.name = "AbortError";
  throw error;
}

function validateTokenLimit(maxGenLen) {
  if (!Number.isInteger(maxGenLen) || maxGenLen < 1) throw new RangeError("maxGenLen must be a positive integer");
}

const DRUM_PROGRAM = 128;

function normalizeAllowedPrograms(programs) {
  if (programs === undefined) return undefined;
  if (!Array.isArray(programs) || !programs.length ||
      [...programs].some(program => !Number.isInteger(program) || program < 0 || program > DRUM_PROGRAM)) {
    throw new RangeError("allowedPrograms must be a non-empty array of integers from 0 to 128");
  }
  return [...new Set(programs)];
}

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
    mask.fill(-1e9, rowOff + maxCol + 1, rowOff + total);
  }
  return mask;
}

const chunkAcoustics = Symbol("chunk acoustics");

export class WebScoreClient {
  constructor(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch, gpuKvCache = false, cacheEmbeddings = false }) {
    this.ort = ort;
    this.lookup = buildTokenLookup(tokenizer);
    this.encoderSession = encoderSession;
    this.embedSession = embedSession;
    this.decoderSession = decoderSession;
    this.arch = arch;
    this.gpuKvCache = gpuKvCache;
    this.cacheEmbeddings = cacheEmbeddings;
    this.embeddingTable = null;
    this.disposed = false;
    this.transcribing = false;
    this.mel = new WebMelProcessor();
    this.maskedIds = new Set([this.lookup.padId, this.lookup.bosId, this.lookup.unkId]);
  }

  static async create(ort, { baseUrl = "/models", useInt8 = true, modelUrls = null, executionProviders = ["webgpu", "wasm"], gpuKvCache = false, cacheEmbeddings = false, gpuTokenSelection = false, gpuBatchPairs = false, gpuBatchWidth = 2, gpuFeedbackBlock = 0 } = {}) {
    if (!Array.isArray(executionProviders) || !executionProviders.length) throw new Error("Specify at least one execution provider");
    if (gpuKvCache && (executionProviders[0] !== "webgpu" || useInt8)) {
      throw new Error("GPU KV cache is experimental: select WebGPU first and an FP32 decoder");
    }
    if (gpuTokenSelection && !gpuKvCache) throw new Error("GPU token selection requires GPU KV cache");
    if (gpuFeedbackBlock !== 0 && gpuFeedbackBlock !== 4) throw new Error("Unsupported block feedback size");
    const decFile = useInt8 ? "decoder_kv.int8.onnx" : "decoder_kv.onnx";
    const names = ["tokenizer.json", "config.json", "mel_encoder.onnx", "token_embed.onnx", decFile];
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
          ? { ...opts, preferredOutputLocation: { logits: gpuTokenSelection ? "gpu-buffer" : "cpu", present_k: "gpu-buffer", present_v: "gpu-buffer" } }
          : opts;
        return await ort.InferenceSession.create(urls[name], sessionOpts);
      } catch {
        throw new Error(`Failed to load ${name}: check access, URL expiry and model compatibility`);
      }
    };
    const sessionNames = ["mel_encoder.onnx", "token_embed.onnx", decFile];
    const settled = [];
    if (executionProviders[0] === "webgpu") {
      // Native WebGPU EP rejects concurrent session creation, even for different graphs.
      for (const name of sessionNames) {
        try {
          settled.push({ status: "fulfilled", value: await loadSession(name) });
        } catch (reason) {
          settled.push({ status: "rejected", reason });
        }
      }
    } else {
      settled.push(...await Promise.allSettled(sessionNames.map(loadSession)));
    }
    const failed = settled.find(r => r.status === "rejected");
    if (failed) {
      await Promise.allSettled(settled.filter(r => r.status === "fulfilled").map(r => r.value?.release?.()));
      throw failed.reason;
    }
    const [encoderSession, embedSession, decoderSession] = settled.map(r => r.value);

    try {
      const client = new WebScoreClient(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch, gpuKvCache, cacheEmbeddings });
      if (gpuTokenSelection) {
        const device = await ort.env.webgpu.device;
        client.tokenSelector = await GpuTokenSelector.create(device, client.lookup.idToEvent.length);
        client.gpuKvBindingBytes = Math.min(device?.limits?.maxBufferSize ?? 0, device?.limits?.maxStorageBufferBindingSize ?? 0);
      }
      // Only supported by the existing dynamic-batch WebScore graph and native FP32 logits ABI.
      const hasBatch = (session, name, axis) => session.inputMetadata?.some(input =>
        input.name === name && (input.shape?.[axis] === "batch" || input.shape?.[axis] === 2));
      client.gpuBatchWidth = gpuBatchWidth === 4 ? 4 : 2;
      const packedRelease = "https://models.midistudio.app/web-student-kd/releases/v2-step-00046500-grpo100-fp16-v1";
      const batchArtifact = !useInt8 && names.every(name =>
        urls[name] === `${packedRelease}/${name === "config.json" ? "config.web-abi-v1.json" : name}`);
      client.gpuBatchPairs = Boolean(gpuBatchPairs && batchArtifact && client.tokenSelector &&
        hasBatch(encoderSession, "mel", 0) && hasBatch(decoderSession, "inputs_embeds", 0) &&
        hasBatch(decoderSession, "past_k", 1) && hasBatch(decoderSession, "past_v", 1));
      client.gpuFeedbackBlock = client.gpuBatchPairs && gpuFeedbackBlock === 4 ? 4 : 0;
      return client;
    } catch {
      await Promise.allSettled([encoderSession, embedSession, decoderSession].map(s => s?.release?.()));
      throw new Error("Invalid model metadata or interface");
    }
  }

  async embedTokenIds(ids) {
    if (!this.cacheEmbeddings) {
      const input_ids = new this.ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
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

  async transcribeChunk(pcm5s, { promptTokens = [], maxGenLen = 1024, signal = null, allowedPrograms,
    [chunkAcoustics]: acoustics = {},
  } = {}) {
    if (this.disposed) throw new Error("Client has been disposed");
    validateTokenLimit(maxGenLen);
    throwIfAborted(signal);
    const programs = normalizeAllowedPrograms(allowedPrograms);
    let maskedIds = this.maskedIds;
    if (programs !== undefined) {
      const allowed = new Set(programs);
      maskedIds = new Set(maskedIds);
      for (const { id, kind, value } of this.lookup.idToEvent) {
        if ((kind === "inst" && (value >= DRUM_PROGRAM || !allowed.has(value))) ||
            (kind === "drum" && !allowed.has(DRUM_PROGRAM))) maskedIds.add(id);
      }
      if (promptTokens.some(id => maskedIds.has(id))) {
        throw new RangeError("promptTokens contain tokens not permitted by allowedPrograms");
      }
    }
    // The acoustic prefix does not depend on the prompt. Keep only this chunk's
    // encoder result so an EOS retry starts fresh decoder KV without recomputing audio.
    if (acoustics.condEmbeds === undefined) {
      const { data: melData, dims: melDims } = this.mel.computeChunkLogMel(pcm5s);
      const melTensor = new this.ort.Tensor("float32", melData, melDims);
      const encOut = await this.encoderSession.run({ mel: melTensor });
      acoustics.condEmbeds = encOut.cond_embeds;
    }
    throwIfAborted(signal);
    const { condEmbeds } = acoustics;
    let selector = this.tokenSelectorDisabled ? null : this.tokenSelector;
    const downgradeSelector = () => { this.tokenSelectorDisabled = true; selector = null; };
    if (selector) {
      try { selector.setMask(maskedIds); } catch { downgradeSelector(); }
    }

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
    // Session.run is awaited before the next write; no concurrent use of this host input.
    const posData = new BigInt64Array(1);
    const posTensor = new this.ort.Tensor("int64", posData, [1]);
    const generated = [...promptTokens];
    // Single-token causal rows are all zero. Reuse bounded storage for their
    // views instead of allocating and clearing an O(past length) buffer per token.
    const decodeMask = maxGenLen > 1 ? new Float32Array(prefillLen + Math.min(maxGenLen - 1, 4096)) : null;

    try {
      for (let step = 0; step < maxGenLen; step++) {
        throwIfAborted(signal);
        const seqLen = stepEmbeds.dims[1];
        const pastLen = pastK.dims[3];
        const total = pastLen + seqLen;
        const maskData = step > 0 && seqLen === 1 && total <= decodeMask.length
          ? decodeMask.subarray(0, total) : buildCausalMask(seqLen, pastLen);
        const maskTensor = new this.ort.Tensor("float32", maskData, [1, 1, seqLen, total]);
        posData[0] = BigInt(posOffset);

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
        const logits = decOut.logits;
        let nextTok;
        try {
          throwIfAborted(signal);
          const vocabSize = this.lookup.idToEvent.length;
          const logitRows = logits.dims[1];
          if (logitRows !== 1 && logitRows !== seqLen) throw new Error("Invalid decoder logits shape");
          const start = (logitRows - 1) * vocabSize;
          if (selector && logits.location === "gpu-buffer" && logits.type === "float32") {
            try {
              nextTok = await selector.run(logits.gpuBuffer, start, logitRows * vocabSize * 4);
            } catch { downgradeSelector(); }
          }
          if (nextTok === undefined) {
            const values = logits.location === "gpu-buffer" ? await logits.getData(true) : logits.data;
            nextTok = argmaxMasked(values.subarray(start, start + vocabSize), maskedIds);
          }
          throwIfAborted(signal);
        } finally {
          if (logits.location === "gpu-buffer") logits.dispose();
        }
        if (programs !== undefined && nextTok < 0) throw new Error("No valid token available for allowedPrograms");
        generated.push(nextTok);

        if (nextTok === this.lookup.eosId) return generated;

        if (step + 1 < maxGenLen) stepEmbeds = await this.embedTokenIds([nextTok]);
      }

      throw new TokenLimitError();
    } finally {
      if (this.gpuKvCache) {
        for (const tensor of [pastK, pastV]) if (tensor.location === "gpu-buffer") tensor.dispose();
      }
    }
  }

  async transcribePcm(pcm16k, { maxGenLen = 1024, onProgress = null, signal = null, boundaryMode = "pitch-tie", allowedPrograms } = {}) {
    if (this.disposed) throw new Error("Client has been disposed");
    if (this.transcribing) throw new Error("A transcription is already running");
    validateTokenLimit(maxGenLen);
    if (!["legacy", "prelude-forcing", "pitch-tie"].includes(boundaryMode)) throw new RangeError("Unsupported boundaryMode");
    throwIfAborted(signal);
    // Snapshot the caller's list before any await (all chunks and retries use it).
    const programs = normalizeAllowedPrograms(allowedPrograms);
    const allowed = programs === undefined ? null : new Set(programs);
    const keepsProgram = note => allowed.has(note.isDrum ? DRUM_PROGRAM : note.program);
    const chunkOptions = programs === undefined ? { maxGenLen, signal } : { maxGenLen, signal, allowedPrograms: programs };
    this.transcribing = true;
    try {
      const chunks = splitPcmInto5sChunks(pcm16k);
      const allNotes = [];
      const tokensByChunk = [];
      const chunkStatuses = [];
      let openNotes = [];
      let paired = null;

      for (let idx = 0; idx < chunks.length; idx++) {
        throwIfAborted(signal);
        const { pcm, startSec } = chunks[idx];
        const acoustics = {};
        // Native score stitching remains sequential. Only first passes that do not
        // depend on a prior tie prompt are batched; EOS retry uses a fresh serial decoder.
        if (!paired && this.gpuBatchPairs && !this.batchPairsDisabled && !this.tokenSelectorDisabled &&
            boundaryMode === "pitch-tie" && maxGenLen >= 32 && idx + 1 < chunks.length) {
          let width = boundedGpuBatchWidth(this.gpuBatchWidth ?? 2, chunks.length - idx, maxGenLen, this.gpuKvBindingBytes);
          while (width >= 2) {
            try {
              const firstPass = await runBatchPair(this, chunks.slice(idx, idx + width).map(chunk => chunk.pcm),
                { ...chunkOptions, selector: this.tokenSelector });
              paired = { start: idx, ...firstPass };
              break;
            } catch {
              throwIfAborted(signal);
              if (width > 2) {
                this.gpuBatchWidth = 2;
                width = 2;
                console.warn("[WebScore] Wide GPU decode unavailable; trying batch width 2");
              } else {
                this.batchPairsDisabled = true;
                console.warn("[WebScore] Batched GPU decode unavailable; using sequential pitch-tie");
                break;
              }
            }
          }
        }
        const pairIndex = paired ? idx - paired.start : -1;
        const pairedTokens = pairIndex >= 0 ? paired.tokens[pairIndex] : null;
        const pairedEos = pairIndex >= 0 ? paired.eos[pairIndex] : null;
        if (paired && pairIndex === paired.eos.length - 1) paired = null;
        const forcePrelude = boundaryMode === "prelude-forcing" || (boundaryMode === "legacy" && idx > 0);
        const promptTokens = forcePrelude ? encodeTiePrologue(openNotes, this.lookup) : [];
        const chunkStatus = { index: idx, startSec, endSec: Math.min(startSec + 5, pcm16k.length / 16000), status: "ok", attempts: 1, boundaryReset: false };
        let tokens;
        try {
          if (pairedEos === false) throw new TokenLimitError();
          tokens = pairedEos === true ? pairedTokens :
            await this.transcribeChunk(pcm, { ...chunkOptions, promptTokens, [chunkAcoustics]: acoustics });
        } catch (error) {
          throwIfAborted(signal);
          if (!(error instanceof TokenLimitError)) throw error;
          // Never decode partial runaway tokens. A retry starts new KV inside
          // transcribeChunk while reusing only the unchanged acoustic prefix.
          allNotes.push(...finalizeOpenNotes(openNotes, startSec));
          openNotes = [];
          chunkStatus.boundaryReset = true;
          chunkStatus.reason = "token_limit";
          chunkStatus.attempts = 2;
          try {
            tokens = await this.transcribeChunk(pcm, { ...chunkOptions, promptTokens: encodeTiePrologue([], this.lookup), [chunkAcoustics]: acoustics });
            chunkStatus.status = "recovered";
          } catch (retryError) {
            throwIfAborted(signal);
            if (!(retryError instanceof TokenLimitError)) throw retryError;
            chunkStatus.status = "failed";
            tokens = [];
          }
        }
        throwIfAborted(signal);
        tokensByChunk.push(tokens);
        if (chunkStatus.status !== "failed") {
          const decoded = decodeChunkTokens(tokens, this.lookup, { chunkStartSec: startSec, openNotes, boundaryMode });
          // Legacy parsing can infer program 0 without an explicit inst token.
          // Guard both output and carried notes before preparing the next prompt.
          allNotes.push(...(allowed ? decoded.completed.filter(keepsProgram) : decoded.completed));
          openNotes = allowed ? decoded.nextOpenNotes.filter(keepsProgram) : decoded.nextOpenNotes;
        }
        chunkStatuses.push(chunkStatus);
        if (onProgress) onProgress({ completedChunks: idx + 1, totalChunks: chunks.length, chunkStatus: { ...chunkStatus } });
      }

      const endSec = pcm16k.length / 16000;
      allNotes.push(...finalizeOpenNotes(openNotes, endSec));
      // The final encoder input is padded to five seconds, but output must not be.
      const notes = allNotes.filter(n => n.onset >= 0 && n.onset < endSec)
        .map(n => ({ ...n, offset: Math.min(n.offset, endSec) }));
      const midiBytes = encodeNotesToMidi(notes);
      return { notes, midiBytes, tokensByChunk, chunkStatuses };
    } finally { this.transcribing = false; }
  }

  async dispose() {
    if (this.disposed) return;
    if (this.transcribing) throw new Error("Cannot dispose during transcription");
    this.disposed = true;
    this.embeddingTable = null;
    this.tokenSelector?.dispose();
    const results = await Promise.allSettled([this.encoderSession, this.embedSession, this.decoderSession].map(s => s?.release?.()));
    if (results.some(r => r.status === "rejected")) throw new Error("One or more model sessions could not be released");
  }

  async transcribeAudio(audioInput, opts = {}) {
    throwIfAborted(opts?.signal);
    const pcm16k = await decodeAudioTo16kMono(audioInput);
    throwIfAborted(opts?.signal);
    return this.transcribePcm(pcm16k, opts);
  }
}
