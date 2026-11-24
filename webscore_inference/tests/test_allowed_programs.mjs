import test from 'node:test';
import assert from 'node:assert/strict';
import { WebScoreClient, TokenLimitError } from '../web/model-client.mjs';

const tokens = ['<pad>', '<bos>', '<eos>', '<unk>', 'inst:0', 'inst:24', 'inst:128',
  'inst:129', 'drum:36', 'pitch:60', 'vel:1', 'vel:0', 'time:0', 'time:50', 'tie'];
const tokenizer = { pad_id: 0, bos_id: 1, eos_id: 2, unk_id: 3, id_to_token: tokens };
class Tensor {
  constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
}
function makeClient() {
  const client = new WebScoreClient({ Tensor }, {
    tokenizer,
    arch: { dim: 1, num_heads: 1, num_kv_heads: 1, num_layers: 1 },
    encoderSession: {
      run: async () => ({ cond_embeds: new Tensor('float32', Float32Array.of(0), [1, 1, 1]) }),
    },
    decoderSession: {},
  });
  client.mel = { computeChunkLogMel: () => ({ data: Float32Array.of(0), dims: [1, 1, 1] }) };
  client.embedTokenIds = async ids => new Tensor('float32', new Float32Array(ids.length), [1, ids.length, 1]);
  return client;
}
const id = label => tokens.indexOf(label);
const pcm = () => new Float32Array(16000);

test('actual token loop masks excluded programs and percussion, then restores default on the same client', async () => {
  const c = makeClient();
  const candidates = [
    ['tie'],
    ['inst:24', 'inst:0', '<eos>'],
    ['vel:1'],
    ['drum:36', 'pitch:60'],
    ['time:50'],
    ['vel:0'],
    ['pitch:60'],
    ['<eos>'],
  ];
  let step;
  c.decoderSession.run = async () => {
    const data = new Float32Array(tokens.length).fill(-Infinity);
    candidates[step++].forEach((label, i) => { data[id(label)] = 10 - i; });
    const kv = new Tensor('float32', [], [1, 1, 1, 1, 1]);
    return { logits: new Tensor('float32', data, [1, 1, tokens.length]), present_k: kv, present_v: kv };
  };
  const run = options => { step = 0; return c.transcribePcm(pcm(), options); };
  const restricted = await run({ allowedPrograms: [0] });
  assert.deepEqual(restricted.notes.map(n => n.program), [0]);
  assert.deepEqual(restricted.tokensByChunk[0], ['tie', 'inst:0', 'vel:1', 'pitch:60', 'time:50', 'vel:0', 'pitch:60', '<eos>'].map(id));
  const auto = await run();
  assert.ok(auto.tokensByChunk[0].includes(id('inst:24')));
  assert.deepEqual(auto.notes.map(n => n.program), [128]);
  const pianoAndDrums = await run({ allowedPrograms: [0, 128] });
  assert.deepEqual(pianoAndDrums.notes.map(n => n.program), [128]);
  assert.ok(pianoAndDrums.tokensByChunk[0].includes(id('inst:0')));
  assert.ok(pianoAndDrums.tokensByChunk[0].includes(id('drum:36')));
});

test('uses one request snapshot on both EOS recovery and following chunks', async () => {
  const c = makeClient();
  const requested = [0];
  const calls = [];
  c.transcribeChunk = async (_, options) => {
    calls.push(options.allowedPrograms.slice());
    if (calls.length === 1) {
      requested[0] = 24;
      throw new TokenLimitError();
    }
    return ['tie', '<eos>'].map(id);
  };
  const r = await c.transcribePcm(new Float32Array(80001), { allowedPrograms: requested });
  assert.deepEqual(calls, [[0], [0], [0]]);
  assert.deepEqual(r.chunkStatuses.map(s => s.status), ['recovered', 'ok']);
});

test('legacy implicit program cannot leave a disallowed note, MIDI track or next-chunk prompt', async () => {
  const c = makeClient();
  let calls = 0;
  c.transcribeChunk = async (_, options) => {
    if (calls++ === 0) return ['tie', 'pitch:60', '<eos>'].map(id);
    assert.deepEqual(options.promptTokens, [id('tie')]);
    return ['tie', 'drum:36', '<eos>'].map(id);
  };
  const r = await c.transcribePcm(new Float32Array(96000), { boundaryMode: 'legacy', allowedPrograms: [128] });
  assert.deepEqual(r.notes.map(n => [n.program,n.isDrum]), [[128,true]]);
  assert.equal((r.midiBytes[10] << 8) | r.midiBytes[11], 2);
});

test('invalid lists and forbidden explicit prompts reject before inference', async () => {
  const c = makeClient();
  c.encoderSession.run = async () => assert.fail('invalid inputs must not run the model');
  for (const allowedPrograms of [[], null, [129], [-1], [1.5], ['0'], Array(1), new Uint8Array([0])]) {
    await assert.rejects(c.transcribePcm(pcm(), { allowedPrograms }), /allowedPrograms/);
  }
  await assert.rejects(c.transcribeChunk(pcm(), { allowedPrograms: [0], promptTokens: [id('inst:24')] }), /promptTokens/);
  assert.equal(c.transcribing, false);
});
