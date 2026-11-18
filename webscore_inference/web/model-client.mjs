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
  constructor(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch }) {
    this.ort = ort;
    this.lookup = buildTokenLookup(tokenizer);
    this.encoderSession = encoderSession;
    this.embedSession = embedSession;
    this.decoderSession = decoderSession;
    this.arch = arch;
    this.mel = new WebMelProcessor();
    this.maskedIds = new Set([this.lookup.padId, this.lookup.bosId, this.lookup.unkId]);
  }

  static async create(ort, { baseUrl = "/models", useInt8 = true, modelUrls = null } = {}) {
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
    const [tokRes, cfgRes] = await Promise.all([
      fetch(urls["tokenizer.json"]),
      fetch(urls["config.json"]),
    ]);
    if (!tokRes.ok) throw new Error(`Failed to fetch tokenizer.json: ${tokRes.status}`);
    if (!cfgRes.ok) throw new Error(`Failed to fetch config.json: ${cfgRes.status}`);

    const tokenizer = await tokRes.json();
    const arch = await cfgRes.json();
    const opts = { executionProviders: ["webgpu", "wasm"] };
    const [encoderSession, embedSession, decoderSession] = await Promise.all([
      ort.InferenceSession.create(urls["mel_encoder.onnx"], opts),
      ort.InferenceSession.create(urls["token_embed.onnx"], opts),
      ort.InferenceSession.create(urls[decFile], opts),
    ]);

    return new WebScoreClient(ort, { tokenizer, encoderSession, embedSession, decoderSession, arch });
  }

  async transcribeChunk(pcm5s, { promptTokens = [], maxGenLen = 1024 } = {}) {
    const { data: melData, dims: melDims } = this.mel.computeChunkLogMel(pcm5s);
    const melTensor = new this.ort.Tensor("float32", melData, melDims);
    const encOut = await this.encoderSession.run({ mel: melTensor });
    const condEmbeds = encOut.cond_embeds;

    const seqIds = [this.lookup.bosId, ...promptTokens];
    const idsTensor = new this.ort.Tensor("int64", BigInt64Array.from(seqIds.map(BigInt)), [1, seqIds.length]);
    const embOut = await this.embedSession.run({ input_ids: idsTensor });
    const tokEmbeds = embOut.token_embeds;

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

      pastK = decOut.present_k;
      pastV = decOut.present_v;
      posOffset += seqLen;

      const vocabSize = this.lookup.idToEvent.length;
      const logitsArr = decOut.logits.data;
      const lastRow = logitsArr.subarray((seqLen - 1) * vocabSize, seqLen * vocabSize);
      const nextTok = argmaxMasked(lastRow, this.maskedIds);
      generated.push(nextTok);

      if (nextTok === this.lookup.eosId) break;

      const nextIdTensor = new this.ort.Tensor("int64", BigInt64Array.from([BigInt(nextTok)]), [1, 1]);
      const nextEmb = await this.embedSession.run({ input_ids: nextIdTensor });
      stepEmbeds = nextEmb.token_embeds;
    }

    return generated;
  }

  async transcribePcm(pcm16k, { maxGenLen = 1024, onProgress = null } = {}) {
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

    allNotes.push(...finalizeOpenNotes(openNotes, chunks.length * 5.0));
    const midiBytes = encodeNotesToMidi(allNotes);
    return { notes: allNotes, midiBytes, tokensByChunk };
  }

  async transcribeAudio(audioInput, opts = {}) {
    const pcm16k = await decodeAudioTo16kMono(audioInput);
    return this.transcribePcm(pcm16k, opts);
  }
}
