import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

test('worker forwards boundaryMode and preserves chunk status on the result', async () => {
  const src = await readFile(new URL('../../frontend/scripts/model.worker.js', import.meta.url), 'utf8');
  const messages = [];
  const calls = [];
  let listener;
  const context = vm.createContext({
    importScripts() {}, ort: { env: { wasm: {} } }, console,
    self: {
      addEventListener(type, callback) { assert.equal(type, 'message'); listener = callback; },
      postMessage(msg) { messages.push(JSON.parse(JSON.stringify(msg))); },
    },
    mockModel: {
      async transcribePcm(pcm, options) {
        calls.push({ pcm, options });
        return { notes: [], chunkStatuses: [{ index:0,status:'failed' }] };
      },
    },
  });
  vm.runInContext(src + '\nmodel = mockModel;', context);
  const pcm = new Float32Array(1);
  await listener({data:{type:'transcribe',pcm,maxGenLen:99,boundaryMode:'pitch-tie'}});
  assert.equal(calls[0].pcm,pcm);
  assert.equal(calls[0].options.boundaryMode,'pitch-tie');
  assert.equal(calls[0].options.maxGenLen,99);
  assert.deepEqual(messages[0],{type:'result',notes:[],chunkStatuses:[{index:0,status:'failed'}]});
  await listener({data:{type:'transcribe',pcm}});
  assert.equal(Object.hasOwn(calls[1].options,'boundaryMode'),false);
});
