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

  static async create(ort, { baseUrl = "/models", useInt8 = true, modelUrls = null, executionProviders = ["webgpu", "wasm"], gpuKvCache = false, cacheEmbeddings = false } = {}) {
    if (!Array.isArray(executionProviders) || !executionProviders.length) throw new Error("Specify at least one execution provider");
    if (gpuKvCache && (executionProviders[0] !== "webgpu" || useInt8)) {
      throw new Error("GPU KV cache is experimental: select WebGPU first and an FP32 decoder");
    }
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
          ? { ...opts, preferredOutputLocation: { logits: "cpu", present_k: "gpu-buffer", present_v: "gpu-buffer" } }
          : opts;
        return await ort.InferenceSession.create(urls[name], sessionOpts);
      } catch {
        throw new Error(`Failed to load ${name}: check access, URL expiry and model compatibility`);
      }
    };
    const settled = await Promise.allSettled([
      loadSession("mel_encoder.onnx"), loadSession("token_embed.onnx"), loadSession(decFile),
    ]);
    const failed = settled.find(r => r.status === "rejected");
    if (failed) {
      await Promise.allSettled(settled.filter(r => r.status === "fulfilled").map(r => r.value?.release?.()));
      throw failed.reason;
    }
    const [encoderSession, embedSession, decoderSession] = settled.map(r => r.value);

    try {
      return new WebScoreClient(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch, gpuKvCache, cacheEmbeddings });
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

  async transcribeChunk(pcm5s, { promptTokens = [], maxGenLen = 1024, signal = null } = {}) {
    if (this.disposed) throw new Error("Client has been disposed");
    validateTokenLimit(maxGenLen);
    throwIfAborted(signal);
    const { data: melData, dims: melDims } = this.mel.computeChunkLogMel(pcm5s);
    const melTensor = new this.ort.Tensor("float32", melData, melDims);
    const encOut = await this.encoderSession.run({ mel: melTensor });
    const condEmbeds = encOut.cond_embeds;

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
        throwIfAborted(signal);

        const vocabSize = this.lookup.idToEvent.length;
        const logitsArr = decOut.logits.data;
        const logitRows = decOut.logits.dims[1];
        if (logitRows !== 1 && logitRows !== seqLen) throw new Error("Invalid decoder logits shape");
        const lastRow = logitsArr.subarray((logitRows - 1) * vocabSize, logitRows * vocabSize);
        const nextTok = argmaxMasked(lastRow, this.maskedIds);
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

  async transcribePcm(pcm16k, { maxGenLen = 1024, onProgress = null, signal = null, boundaryMode = "pitch-tie" } = {}) {
    if (this.disposed) throw new Error("Client has been disposed");
    if (this.transcribing) throw new Error("A transcription is already running");
    validateTokenLimit(maxGenLen);
    if (!["legacy", "prelude-forcing", "pitch-tie"].includes(boundaryMode)) throw new RangeError("Unsupported boundaryMode");
    throwIfAborted(signal);
    this.transcribing = true;
    try {
      const chunks = splitPcmInto5sChunks(pcm16k);
      const allNotes = [];
      const tokensByChunk = [];
      const chunkStatuses = [];
      let openNotes = [];

      for (let idx = 0; idx < chunks.length; idx++) {
        throwIfAborted(signal);
        const { pcm, startSec } = chunks[idx];
        const forcePrelude = boundaryMode === "prelude-forcing" || (boundaryMode === "legacy" && idx > 0);
        const promptTokens = forcePrelude ? encodeTiePrologue(openNotes, this.lookup) : [];
        const chunkStatus = { index: idx, startSec, endSec: Math.min(startSec + 5, pcm16k.length / 16000), status: "ok", attempts: 1, boundaryReset: false };
        let tokens;
        try {
          tokens = await this.transcribeChunk(pcm, { promptTokens, maxGenLen, signal });
        } catch (error) {
          throwIfAborted(signal);
          if (!(error instanceof TokenLimitError)) throw error;
          // Never decode partial runaway tokens. A retry starts new KV inside
          // transcribeChunk, while reusing only the loaded inference sessions.
          allNotes.push(...finalizeOpenNotes(openNotes, startSec));
          openNotes = [];
          chunkStatus.boundaryReset = true;
          chunkStatus.reason = "token_limit";
          chunkStatus.attempts = 2;
          try {
            tokens = await this.transcribeChunk(pcm, { promptTokens: encodeTiePrologue([], this.lookup), maxGenLen, signal });
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
          allNotes.push(...decoded.completed);
          openNotes = decoded.nextOpenNotes;
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
