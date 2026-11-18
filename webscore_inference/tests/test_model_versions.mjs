import test from 'node:test';
import assert from 'node:assert/strict';
import { VersionedWebScoreClient } from '../web/model-version-client.mjs';

const manifest = version => ({ schemaVersion: 1, format: 'webscore-onnx-v1', version, precision: 'int8', files: Object.fromEntries(['config.json','tokenizer.json','mel_encoder.onnx','token_embed.onnx','decoder_kv.int8.onnx'].map(n => [n, `./${version}/${n}?signature=private`])) });
function harness(extra = {}) {
  const state = { current: manifest('v1'), calls: [], clients: [], failLoad: false, failFetch: false };
  const options = { manifestUrl: 'https://models.example/current.json', pollIntervalMs: 0,
    fetchImpl: async (url, opts) => { state.calls.push(opts); if (state.failFetch) throw new Error(url); return { ok: true, json: async () => state.current }; },
    clientFactory: async (_, opts) => {
      if (state.failLoad) throw new Error(opts.modelUrls['config.json']);
      const client = { arch: { model_type:'webscore-gqa',sample_rate:16000,n_fft:2048,frame_rate:100,n_mel_bins:512,dim:4,num_layers:1,num_heads:2,num_kv_heads:1,card:4,pad_token_id:0,bos_token_id:1,eos_token_id:2,unk_token_id:3 }, lookup:{idToEvent:Array(4),padId:0,bosId:1,eosId:2,unkId:3}, encoderSession:{inputNames:['mel'],outputNames:['cond_embeds']}, embedSession:{inputNames:['input_ids'],outputNames:['token_embeds']}, decoderSession:{inputNames:['inputs_embeds','past_k','past_v','position_offset','attn_mask'],outputNames:['logits','present_k','present_v']}, opts, disposed: false, transcribePcm: async () => opts.modelUrls['config.json'], transcribeAudio: async () => 'audio', dispose: async () => { client.disposed = true; } };
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
