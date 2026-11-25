import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import * as audio from '../../frontend/scripts/inference-loudness.mjs';
const {measureLoudness, normalizeInferenceLoudness} = audio;
const sine = (a, len=32000, hz=1000) => Float32Array.from({length:len}, (_,i) => a*Math.sin(2*Math.PI*hz*i/16000));
const near = (x,y,t=.05) => assert.ok(Math.abs(x-y)<t, `${x} vs ${y}`);

test('one constant gain for all samples reaches target, source untouched', () => {
  for (const a of [.7,.05]) {
    const x=sine(a), original=x.slice(), y=normalizeInferenceLoudness(x);
    near(measureLoudness(y).lufs,-18);
    const gain=y[4]/x[4];
    for(let i=0;i<x.length;i++) near(y[i],x[i]*gain,1e-7);
    assert.deepEqual(x,original);
  }
});
test('FFmpeg EBU R128 reference: 16 kHz mono 2 s 1 kHz sine with amplitude 0.5 meters -9.0 LUFS', () => {
  near(measureLoudness(sine(.5)).lufs,-9,.12);
});
test('short input and silence skip boosting; quiet gain and transient peak are capped', () => {
  for(const x of [new Float32Array(),new Float32Array(16000),sine(.006,6399),sine(1e-8)]) {
    assert.equal(normalizeInferenceLoudness(x),x);
  }
  const q=sine(.001);
  near(normalizeInferenceLoudness(q)[4]/q[4],10**(12/20),1e-6);
  q[25000]=1;
  const p=normalizeInferenceLoudness(q);
  near(measureLoudness(p).peak,10**(-1/20),1e-7);
  assert.ok(measureLoudness(p).lufs < -18);
  assert.throws(() => normalizeInferenceLoudness(Float32Array.of(NaN)),/Non-finite/);
});
test('worker message toggles preprocessing, preserving the original transfer and inference options by default', async () => {
  const path=new URL('../../frontend/scripts/model.worker.js',import.meta.url);
  const src=(await readFile(path,'utf8')).replace('import("./inference-loudness.mjs")','Promise.resolve(audio)');
  let listener;
  const received=[];
  const ctx=vm.createContext({
    audio, importScripts(){}, ort:{env:{wasm:{}}}, console,
    self:{addEventListener(_t,f){listener=f},postMessage(){}},
    engine:{async transcribePcm(pcm,options){received.push({pcm,options});return {notes:[],chunkStatuses:[]}}},
  });
  vm.runInContext(src+'\nmodel=engine;',ctx);
  const x=sine(.5), original=x.slice();
  for (const normalizeLoudness of [undefined,false,true]) {
    await listener({data:{type:'transcribe',pcm:x,boundaryMode:'prelude-forcing',normalizeLoudness}});
  }
  assert.equal(received.length,3);
  assert.notEqual(received[0].pcm,x);
  near(measureLoudness(received[0].pcm).lufs,-18);
  assert.equal(received[1].pcm,x);
  assert.notEqual(received[2].pcm,x);
  near(measureLoudness(received[2].pcm).lufs,-18);
  assert.equal(received[2].options.boundaryMode,'prelude-forcing');
  assert.deepEqual(x,original);
});
