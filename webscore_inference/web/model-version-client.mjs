import { WebScoreClient } from './model-client.mjs';
import { DEFAULT_MANIFEST_URL } from './model-config.mjs';

const FILES = ['config.json', 'tokenizer.json', 'mel_encoder.onnx', 'token_embed.onnx'];

function parseManifest(value, sourceUrl) {
  if (!value || value.schemaVersion !== 1 || value.format !== 'webscore-onnx-v1' ||
      typeof value.version !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value.version) ||
      !['int8', 'fp32'].includes(value.precision)) {
    throw new Error('Unsupported model manifest');
  }
  const names = [...FILES, value.precision === 'int8' ? 'decoder_kv.int8.onnx' : 'decoder_kv.onnx'];
  const modelUrls = {};
  for (const name of names) {
    if (typeof value.files?.[name] !== 'string' || !value.files[name].trim()) throw new Error('Incomplete model manifest');
    let url;
    try { url = new URL(value.files[name], sourceUrl); } catch { throw new Error('Invalid model URL'); }
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if ((url.protocol !== 'https:' && !(local && url.protocol === 'http:')) || url.username || url.password || url.hash) {
      throw new Error('Invalid model URL');
    }
    modelUrls[name] = url.href;
  }
  return { version: value.version, useInt8: value.precision === 'int8', modelUrls };
}

function validateClient(client) {
  const a = client?.arch, l = client?.lookup;
  if (!a || !l || a.model_type !== 'webscore-gqa' || a.sample_rate !== 16000 ||
      a.n_fft !== 2048 || a.frame_rate !== 100 || a.n_mel_bins !== 512 ||
      !['dim', 'num_layers', 'num_heads', 'num_kv_heads', 'card'].every(k => Number.isInteger(a[k]) && a[k] > 0) ||
      a.dim % a.num_heads || a.num_heads % a.num_kv_heads || l.idToEvent?.length !== a.card ||
      ![['padId','pad_token_id'],['bosId','bos_token_id'],['eosId','eos_token_id'],['unkId','unk_token_id']].every(([k,v]) => l[k] === a[v] && Number.isInteger(l[k]) && l[k] >= 0 && l[k] < a.card)) {
    throw new Error('Incompatible model metadata');
  }
  const special = ['padId', 'bosId', 'eosId', 'unkId'];
  if (new Set(special.map(k => l[k])).size !== 4 || !special.every(k => l.idToEvent[l[k]]?.kind === `<${k.slice(0, -2)}>`)) throw new Error('Incompatible special tokens');
  const contracts = [
    [client.encoderSession, ['mel'], ['cond_embeds']],
    [client.embedSession, ['input_ids'], ['token_embeds']],
    [client.decoderSession, ['inputs_embeds','past_k','past_v','position_offset','attn_mask'], ['logits','present_k','present_v']],
  ];
  if (!contracts.every(([s,ins,outs]) => s && s.inputNames?.length === ins.length && ins.every(n=>s.inputNames?.includes(n)) && outs.every(n=>s.outputNames?.includes(n)))) {
    throw new Error('Incompatible model interface');
  }
}

/** Model-only updater. All inference/update work is serialized to protect sessions. */
export class VersionedWebScoreClient {
  static async create(ort, options = {}) {
    const instance = new VersionedWebScoreClient(ort, options);
    await instance.checkForUpdate();
    if (instance.pollIntervalMs > 0) instance._schedule();
    return instance;
  }

  constructor(ort, { manifestUrl = DEFAULT_MANIFEST_URL, clientOptions = {}, pollIntervalMs = 60000,
    checkBeforeInference = true, manifestTimeoutMs = 15000, onUpdate = null, onUpdateError = null,
    fetchImpl = (...args) => globalThis.fetch(...args), clientFactory = WebScoreClient.create.bind(WebScoreClient) } = {}) {
    const base = globalThis.location?.href;
    let parsed;
    try { parsed = base ? new URL(manifestUrl, base) : new URL(manifestUrl); } catch { throw new Error('Invalid manifest URL'); }
    if (!manifestUrl || (parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && ['localhost','127.0.0.1','[::1]'].includes(parsed.hostname))) || parsed.username || parsed.password || parsed.hash) throw new Error('Invalid manifest URL');
    if (!Number.isFinite(pollIntervalMs) || (pollIntervalMs !== 0 && pollIntervalMs < 1000)) throw new Error('Polling interval must be zero or at least 1000ms');
    if (['baseUrl','modelUrls','useInt8'].some(k => k in clientOptions)) throw new Error('Model locations and precision must come from the manifest');
    if (!Number.isFinite(manifestTimeoutMs) || manifestTimeoutMs <= 0) throw new Error('Manifest timeout must be positive');
    this.manifestTimeoutMs = manifestTimeoutMs;
    this.ort = ort;
    this.manifestUrl = parsed.href;
    this.clientOptions = { ...clientOptions };
    this.pollIntervalMs = pollIntervalMs;
    this.checkBeforeInference = checkBeforeInference;
    this.onUpdate = onUpdate;
    this.onUpdateError = onUpdateError;
    // The default must call native fetch with the browser global as its receiver.
    this.fetchImpl = fetchImpl;
    this.clientFactory = clientFactory;
    this.client = null;
    this.modelVersion = null;
    this._queue = Promise.resolve();
    this._closed = false;
    this._timer = null;
  }

  _enqueue(work) {
    if (this._closed) return Promise.reject(new Error('Model client is disposed'));
    const result = this._queue.then(work);
    this._queue = result.catch(() => {});
    return result;
  }

  _notify(callback, value) {
    // User callbacks must not turn a committed update into an apparent failure.
    try { Promise.resolve(callback?.(value)).catch(() => {}); } catch { /* callback failures are isolated */ }
  }

  async _update() {
    let manifest;
    const controller = new AbortController();
    let timer;
    try {
      const request = (async () => {
        const response = await this.fetchImpl(this.manifestUrl, { cache: 'no-store', credentials: 'same-origin', signal: controller.signal });
        if (!response.ok) throw new Error('Manifest request failed');
        return parseManifest(await response.json(), this.manifestUrl);
      })();
      const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => {
          controller.abort();
          reject(new Error('Manifest request timed out'));
        }, this.manifestTimeoutMs);
      });
      manifest = await Promise.race([request, timeout]);
    } catch {
      throw new Error('Unable to read a compatible model manifest');
    } finally {
      clearTimeout(timer);
    }
    if (manifest.version === this.modelVersion) return { updated: false, version: this.modelVersion };
    let replacement;
    try {
      replacement = await this.clientFactory(this.ort, { ...this.clientOptions, useInt8: manifest.useInt8, modelUrls: manifest.modelUrls });
      if (!replacement || typeof replacement.transcribePcm !== 'function' || typeof replacement.dispose !== 'function') {
        throw new Error('Incompatible model client');
      }
      validateClient(replacement);
    } catch {
      try { await replacement?.dispose?.(); } catch { /* retain old model */ }
      throw new Error('Unable to load the new model; the previous version was kept');
    }
    const old = this.client;
    this.client = replacement;
    this.modelVersion = manifest.version;
    try { await old?.dispose(); } catch { /* new version is already committed */ }
    const result = { updated: true, version: this.modelVersion };
    this._notify(this.onUpdate, result);
    return result;
  }

  checkForUpdate() { return this._enqueue(() => this._update()); }

  _schedule() {
    if (this._closed || !this.pollIntervalMs) return;
    this._timer = setTimeout(async () => {
      try { await this.checkForUpdate(); }
      catch { if (!this._closed) this._notify(this.onUpdateError, new Error('Model update failed; the previous version was kept')); }
      finally { this._schedule(); }
    }, this.pollIntervalMs);
    this._timer.unref?.();
  }

  _transcribe(method, input, options) {
    return this._enqueue(async () => {
      if (this.checkBeforeInference) {
        try { await this._update(); }
        catch { this._notify(this.onUpdateError, new Error('Model update failed; the previous version was kept')); }
      }
      if (!this.client) throw new Error('No model is loaded');
      return this.client[method](input, options);
    });
  }

  transcribePcm(input, options) { return this._transcribe('transcribePcm', input, options); }
  transcribeAudio(input, options) { return this._transcribe('transcribeAudio', input, options); }

  async dispose() {
    if (this._closed) return this._queue;
    this._closed = true;
    clearTimeout(this._timer);
    const cleanup = this._queue.then(async () => {
      await this.client?.dispose();
      this.client = null;
    });
    this._queue = cleanup.catch(() => {});
    return cleanup;
  }
}
