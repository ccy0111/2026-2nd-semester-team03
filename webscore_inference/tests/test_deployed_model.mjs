import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createDeployedModel, MODEL_MANIFEST_URL, INITIAL_MODEL_VERSION, INITIAL_MODEL_BASE_URL } from '../web/deployed-model.mjs';

// Exercise the real wrapper and default factory, mocking only network and ORT.
test('deployed entry point loads public CDN manifest and uses complete INT8 WASM model set', async (t) => {
  const config = JSON.parse(await readFile(new URL('../config.json', import.meta.url)));
  const tokenizer = JSON.parse(await readFile(new URL('../tokenizer.json', import.meta.url)));
  const files = ['config.json', 'tokenizer.json', 'mel_encoder.onnx', 'token_embed.onnx', 'decoder_kv.int8.onnx'];
  const requests = [], sessions = [];
  const manifest = { schemaVersion: 1, format: 'webscore-onnx-v1', version: INITIAL_MODEL_VERSION,
    precision: 'int8', files: Object.fromEntries(files.map(n => [n, INITIAL_MODEL_BASE_URL + n])) };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    requests.push({ url, options });
    let data;
    if (url === MODEL_MANIFEST_URL) data = manifest;
    else if (url === manifest.files['config.json']) data = config;
    else if (url === manifest.files['tokenizer.json']) data = tokenizer;
    else throw new Error('Unexpected URL');
    return { ok: true, json: async () => data };
  });
  const ort = { InferenceSession: { create: async (url, options) => {
    const kind = url.endsWith('mel_encoder.onnx') ? 0 : url.endsWith('token_embed.onnx') ? 1 : 2;
    const inputNames = [['mel'], ['input_ids'], ['inputs_embeds', 'past_k', 'past_v', 'position_offset', 'attn_mask']][kind];
    const outputNames = [['cond_embeds'], ['token_embeds'], ['logits', 'present_k', 'present_v']][kind];
    const session = { url, options, inputNames, outputNames, released: 0,
      async release() { this.released++; } };
    sessions.push(session); return session;
  } } };
  const client = await createDeployedModel(ort, { pollIntervalMs: 0 });
  try {
    assert.equal(client.modelVersion, INITIAL_MODEL_VERSION);
    assert.equal(requests[0].url, MODEL_MANIFEST_URL);
    assert.equal(requests[0].options.cache, 'no-store');
    assert.deepEqual(sessions.map(s => s.url), files.slice(2).map(n => manifest.files[n]));
    assert.ok(sessions.every(s => s.options.executionProviders.join() === 'wasm'));
    await client.checkForUpdate();
    assert.equal(sessions.length, 3);
  } finally { await client.dispose(); }
  assert.ok(sessions.every(s => s.released === 1));
});
