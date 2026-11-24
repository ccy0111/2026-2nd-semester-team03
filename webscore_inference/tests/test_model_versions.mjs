import test from 'node:test';
import assert from 'node:assert/strict';
import { VersionedWebScoreClient } from '../web/model-version-client.mjs';
import { DEFAULT_MANIFEST_URL } from '../web/model-config.mjs';

const manifest = version => ({ schemaVersion: 1, format: 'webscore-onnx-v1', version, precision: 'int8', files: Object.fromEntries(['config.json','tokenizer.json','mel_encoder.onnx','token_embed.onnx','decoder_kv.int8.onnx'].map(n => [n, `./${version}/${n}?signature=private`])) });
function harness(extra = {}) {
  const state = { current: manifest('v1'), calls: [], clients: [], failLoad: false, failFetch: false };
  const options = { manifestUrl: 'https://models.example/current.json', pollIntervalMs: 0,
    fetchImpl: async (url, opts) => { state.calls.push(opts); if (state.failFetch) throw new Error(url); return { ok: true, json: async () => state.current }; },
    clientFactory: async (_, opts) => {
      if (state.failLoad) throw new Error(opts.modelUrls['config.json']);
      const client = { arch: { model_type:'webscore-gqa',sample_rate:16000,n_fft:2048,frame_rate:100,n_mel_bins:512,dim:4,num_layers:1,num_heads:2,num_kv_heads:1,card:4,pad_token_id:0,bos_token_id:1,eos_token_id:2,unk_token_id:3 }, lookup:{idToEvent:['<pad>','<bos>','<eos>','<unk>'].map(kind=>({kind})),padId:0,bosId:1,eosId:2,unkId:3}, encoderSession:{inputNames:['mel'],outputNames:['cond_embeds']}, embedSession:{inputNames:['input_ids'],outputNames:['token_embeds']}, decoderSession:{inputNames:['inputs_embeds','past_k','past_v','position_offset','attn_mask'],outputNames:['logits','present_k','present_v']}, opts, disposed: false, transcribePcm: async () => opts.modelUrls['config.json'], transcribeAudio: async () => 'audio', dispose: async () => { client.disposed = true; } };
      state.clients.push(client); return client;
    }, ...extra };
  return { state, options };
}
test('loads complete version set, bypasses manifest cache, skips same-version reload', async () => {
  const {state,options} = harness(); const client = await VersionedWebScoreClient.create({},options);
  assert.equal(client.modelVersion,'v1'); assert.equal(state.clients[0].opts.useInt8,true);
  assert.ok(Object.values(state.clients[0].opts.modelUrls).every(u => u.includes('/v1/')));
  assert.equal(state.calls[0].cache,'no-store');
  await client.checkForUpdate(); assert.equal(state.clients.length,1); await client.dispose();
});
test('uses the configured CDN current manifest when no URL is supplied', async () => {
  const { state, options } = harness();
  delete options.manifestUrl;
  let requestedUrl;
  const fetchManifest = options.fetchImpl;
  options.fetchImpl = async (url, init) => {
    requestedUrl = url;
    return fetchManifest(url, init);
  };
  const client = await VersionedWebScoreClient.create({}, options);
  try {
    assert.equal(requestedUrl, DEFAULT_MANIFEST_URL);
    assert.equal(client.modelVersion, 'v1');
    assert.equal(state.clients.length, 1);
  } finally {
    await client.dispose();
  }
});
test('the default native-style fetch uses the global receiver for startup and refresh', async () => {
  const { state, options } = harness();
  delete options.fetchImpl;
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async function (url, init) {
    assert.equal(this, globalThis);
    calls.push({ url, init });
    return { ok: true, json: async () => state.current };
  };
  let client;
  try {
    client = await VersionedWebScoreClient.create({}, options);
    state.current = manifest('v2');
    await client.checkForUpdate();
    assert.equal(client.modelVersion, 'v2');
    assert.deepEqual(calls.map(({ url }) => url), [options.manifestUrl, options.manifestUrl]);
    for (const { init } of calls) {
      assert.equal(init.cache, 'no-store');
      assert.equal(init.credentials, 'same-origin');
      assert.ok(init.signal instanceof AbortSignal);
    }
  } finally {
    globalThis.fetch = originalFetch;
    await client?.dispose();
  }
});

test('an explicit injected fetch is used even when the global fetch is unavailable', async () => {
  const { state, options } = harness();
  const originalFetch = globalThis.fetch;
  let client;
  globalThis.fetch = undefined;
  try {
    client = await VersionedWebScoreClient.create({}, options);
    assert.equal(client.modelVersion, 'v1');
    assert.equal(state.calls.length, 1);
  } finally {
    globalThis.fetch = originalFetch;
    await client?.dispose();
  }
});

test('supports an explicit backend manifest URL instead of the CDN default', async () => {
  const { options } = harness({ manifestUrl: 'http://127.0.0.1:8000/api/model-manifest' });
  const client = await VersionedWebScoreClient.create({}, options);
  try {
    assert.equal(client.manifestUrl, 'http://127.0.0.1:8000/api/model-manifest');
  } finally {
    await client.dispose();
  }
});
test('replaces only after complete load and disposes old sessions', async () => {
  const {state,options}=harness(); const client=await VersionedWebScoreClient.create({},options);
  state.current=manifest('v2'); await client.checkForUpdate();
  assert.equal(client.modelVersion,'v2'); assert.equal(state.clients[0].disposed,true); await client.dispose();
});
test('failed update retains old model and does not expose signed URLs', async () => {
  const {state,options}=harness(); const client=await VersionedWebScoreClient.create({},options);
  state.current=manifest('v2'); state.failLoad=true;
  await assert.rejects(client.checkForUpdate(), e => !e.message.includes('signature') && !e.cause);
  assert.equal(client.modelVersion,'v1'); assert.equal(state.clients[0].disposed,false);
  assert.match(await client.transcribePcm(new Float32Array(1)),/\/v1\//); await client.dispose();
});
test('malformed, incompatible, partial, and unsafe manifests never replace old model', async () => {
  const {state,options}=harness(); const client=await VersionedWebScoreClient.create({},options);
  for (const bad of [{...manifest('v2'),schemaVersion:2},{...manifest('v2'),format:'other'},{...manifest('v2'),files:{}},{...manifest('v2'),files:{...manifest('v2').files,'config.json':'javascript:alert(1)'}}]) {
    state.current=bad; await assert.rejects(client.checkForUpdate()); assert.equal(client.modelVersion,'v1');
  }
  await client.dispose();
});
test('update waits for active inference and next inference sees the new version', async () => {
  const {state,options}=harness({checkBeforeInference:false}); const client=await VersionedWebScoreClient.create({},options);
  let finish; let started; const start=new Promise(r=>started=r);
  state.clients[0].transcribePcm=async()=>{started(); return new Promise(r=>finish=r);};
  const inference=client.transcribePcm([]); await start;
  state.current=manifest('v2'); const update=client.checkForUpdate(); await Promise.resolve();
  assert.equal(state.clients[0].disposed,false); finish('old result'); assert.equal(await inference,'old result'); await update;
  assert.match(await client.transcribePcm([]),/\/v2\//); await client.dispose();
});
test('failed polling check does not block future checks or leak raw errors', async () => {
  const errors=[]; const {state,options}=harness({onUpdateError:e=>errors.push(e)}); const client=await VersionedWebScoreClient.create({},options);
  state.failFetch=true; await client.transcribePcm([]); assert.equal(errors.length,1); assert.ok(!errors[0].message.includes('https'));
  state.failFetch=false; state.current=manifest('v2'); await client.checkForUpdate(); assert.equal(client.modelVersion,'v2'); await client.dispose();
});
test('dispose waits for queued work, releases sessions, rejects new work', async () => {
  const {state,options}=harness(); const client=await VersionedWebScoreClient.create({},options);
  await client.dispose(); assert.equal(state.clients[0].disposed,true); await assert.rejects(client.transcribePcm([]),/disposed/); await client.dispose();
});
test('same version may renew signed URLs without unnecessary downloads; callbacks isolated', async () => {
  const {state,options}=harness({onUpdate:()=>{throw new Error('callback');}}); const client=await VersionedWebScoreClient.create({},options);
  state.current.files['config.json']+='renewed'; await client.checkForUpdate(); assert.equal(state.clients.length,1); await client.dispose();
});
test('constructor rejects mixed model overrides and invalid polling', () => {
  assert.throws(()=>new VersionedWebScoreClient({},harness({clientOptions:{useInt8:true}}).options));
  assert.throws(()=>new VersionedWebScoreClient({},harness({pollIntervalMs:1}).options));
});
test('incompatible loaded metadata is disposed while old sessions stay usable', async () => {
  const {state,options}=harness(); const factory=options.clientFactory;
  options.clientFactory=async(...args)=>{const c=await factory(...args); if(state.current.version==='v2') c.arch.sample_rate=44100; return c;};
  const client=await VersionedWebScoreClient.create({},options); state.current=manifest('v2');
  await assert.rejects(client.checkForUpdate()); assert.equal(state.clients[1].disposed,true); assert.equal(state.clients[0].disposed,false); await client.dispose();
});
test('automatic polling notices new version and can roll back to an older published version', async () => {
  const {state,options}=harness({pollIntervalMs:1000}); const client=await VersionedWebScoreClient.create({},options);
  try {
    state.current=manifest('v2'); await new Promise(r=>setTimeout(r,1100)); assert.equal(client.modelVersion,'v2');
    state.current=manifest('v1'); await client.checkForUpdate(); assert.equal(client.modelVersion,'v1');
  } finally { await client.dispose(); }
});
test('hung manifest times out and aborts without blocking subsequent inference or disposal', async () => {
  const {state,options}=harness({manifestTimeoutMs:20}); const client=await VersionedWebScoreClient.create({},options);
  let signal; client.fetchImpl=async(_,opts)=>{ signal=opts.signal; return new Promise(()=>{}); };
  await assert.rejects(client.checkForUpdate(),/compatible model manifest/); assert.equal(signal.aborted,true);
  assert.match(await client.transcribePcm([]),/\/v1\//); await client.dispose();
});
test('dispose really waits for active and queued inference and then rejects new work', async () => {
  const {state,options}=harness({checkBeforeInference:false}); const client=await VersionedWebScoreClient.create({},options);
  let finish, started; const start=new Promise(r=>started=r); let calls=0;
  state.clients[0].transcribePcm=async()=>{ if (++calls===1) { started(); await new Promise(r=>finish=r); } return calls; };
  const first=client.transcribePcm([]); await start; const second=client.transcribePcm([]); const disposal=client.dispose();
  assert.equal(state.clients[0].disposed,false); await assert.rejects(client.transcribePcm([]),/disposed/);
  finish(); await first; await second; await disposal; assert.equal(calls,2); assert.equal(state.clients[0].disposed,true);
});
test('async update callback rejections are isolated', async () => {
  const {options}=harness({onUpdate:async()=>{throw new Error('private callback error');}});
  const client=await VersionedWebScoreClient.create({},options); await new Promise(r=>setImmediate(r)); await client.dispose();
});
test('extra required inputs and incorrect special tokens reject the replacement', async () => {
  for (const corrupt of [c=>c.decoderSession.inputNames.push('extra'), c=>c.lookup.idToEvent[1].kind='<eos>', c=>{c.arch.eos_token_id=1;c.lookup.eosId=1;}]) {
    const {state,options}=harness(); const factory=options.clientFactory;
    options.clientFactory=async(...args)=>{const c=await factory(...args);if(state.current.version==='v2')corrupt(c);return c;};
    const client=await VersionedWebScoreClient.create({},options); state.current=manifest('v2'); await assert.rejects(client.checkForUpdate());
    assert.equal(client.modelVersion,'v1'); assert.equal(state.clients[1].disposed,true); await client.dispose();
  }
});

test('selects one explicit provider variant and rejects a mismatched variant without replacing the last client', async () => {
 const gpuFiles = Object.fromEntries(['config.json','tokenizer.json','mel_encoder.onnx','token_embed.onnx','decoder_kv.onnx'].map(n=>[n,`./fp16/${n}`]));
 const {state,options}=harness({manifestVariant:'webgpu'});
 state.current={...manifest('with-fp16'),variants:{webgpu:{precision:'fp16',files:gpuFiles}}};
 const c=await VersionedWebScoreClient.create({},options);
 try {
  assert.equal(state.clients[0].opts.useInt8,false);
  assert.match(state.clients[0].opts.modelUrls['decoder_kv.onnx'],/\/fp16\/decoder_kv\.onnx$/);
  assert.equal(state.clients[0].opts.modelUrls['decoder_kv.int8.onnx'],undefined);
  state.current=manifest('no-fp16');
  await assert.rejects(c.checkForUpdate(),/compatible model manifest/);
  assert.equal(state.clients.length,1);
  assert.equal(await c.transcribePcm(new Float32Array(1)),'https://models.example/fp16/config.json');
 } finally {await c.dispose();}
});
test('WASM intentionally selects INT8 even when the manifest also has a WebGPU variant', async () => {
 const {state,options}=harness({manifestVariant:'wasm'});
 state.current={...manifest('top-int8'),variants:{webgpu:{precision:'fp16',files:{}},wasm:{precision:'int8',files:manifest('explicit-int8').files}}};
 const c=await VersionedWebScoreClient.create({},options);
 try {
  assert.equal(state.clients[0].opts.useInt8,true);assert.match(state.clients[0].opts.modelUrls['decoder_kv.int8.onnx'],/\/explicit-int8\//);
 } finally {await c.dispose();}
});
