import test from 'node:test';import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';import vm from 'node:vm';
import {cpuWasmThreadCount} from '../web/cpu-wasm-policy.mjs';
import {boundedGpuBatchWidth} from '../web/gpu-batch-pair.mjs';

const raw=await readFile(new URL('../../frontend/scripts/model.worker.js',import.meta.url),'utf8');
const adapter=(arch='rdna-3',fp16=true,software=false)=>({info:{architecture:arch,isFallbackAdapter:software},features:new Set(fp16?['shader-f16']:[])});
function setup({a,failGpu=false,failLoading=false,iso=false}={}){
 const steps=[],scripts=[],notices=[];
 let listener;
 const runtime={env:{webgpu:{},wasm:{}}};
 const versionFactory={ async create(ort,opts) {
   steps.push(structuredClone(opts));
   if(failLoading||(failGpu&&opts.manifestVariant==='webgpu'))throw new Error('unsanitized opaque provider failure');
   return {ort,client:{gpuKvCache:Boolean(opts.clientOptions.gpuKvCache),tokenSelector:opts.clientOptions.gpuTokenSelection?{}:null,
     gpuBatchPairs:Boolean(opts.clientOptions.gpuBatchPairs),gpuBatchWidth:opts.clientOptions.gpuBatchWidth}};
 }};
 const ctx=vm.createContext({versionFactory,cpuModule:{cpuWasmThreadCount},crossOriginIsolated:iso,SharedArrayBuffer,
  navigator:{hardwareConcurrency:12,...(a?{gpu:{requestAdapter:async()=>a}}:{})},ort:runtime,
  console:{warn(){},error(){}},
  importScripts(x){scripts.push(x);},
  self:{addEventListener(_,f){listener=f},postMessage(o){notices.push(JSON.parse(JSON.stringify(o)))}},
 });
 const src=raw.replace('import(MODEL_MODULE_URL)','Promise.resolve({VersionedWebScoreClient: versionFactory})')
   .replace('import("/inference/cpu-wasm-policy.mjs")','Promise.resolve(cpuModule)');
 vm.runInContext(src,ctx);
 return {load:()=>listener({data:{type:'load'}}),steps,scripts,notices,runtime};
}

test('hardware RDNA3 FP16 dispatches once and explicitly gates B4 plus K4',async()=>{
 const h=setup({a:adapter()});await Promise.all([h.load(),h.load()]);
 assert.equal(h.steps.length,1);assert.match(h.scripts[0],/@1\.31\.1\/dist\/ort\.webgpu\.min\.js$/);
 assert.deepEqual(h.steps[0],{manifestVariant:'webgpu',clientOptions:{executionProviders:['webgpu'],gpuKvCache:true,cacheEmbeddings:true,gpuTokenSelection:true,gpuBatchPairs:true,gpuBatchWidth:4,gpuFeedbackBlock:4}});
 assert.equal(h.runtime.env.wasm.numThreads,1);assert.equal(h.runtime.env.webgpu.adapter.info.architecture,'rdna-3');
 assert.equal(h.notices[0].runtime.provider,'webgpu');
 assert.equal(h.notices[0].runtime.gpuBatchWidth,4);
});
test('non RDNA3 exact provider may use GPU KV but not batch or block',async()=>{
 const h=setup({a:adapter('other')});await h.load();
 assert.equal(h.steps[0].clientOptions.gpuBatchPairs,false);
 assert.equal(h.steps[0].clientOptions.gpuFeedbackBlock,0);
 assert.equal(h.notices[0].runtime.gpuBatchWidth,1);
});
test('no device, software adapter or absent shader-f16 selects INT8 WASM without attempting FP16',async()=>{
 for(const a of [null,adapter('whatever',false),adapter('software',true,true)]){
  const h=setup({a,iso:true});await h.load();
  assert.deepEqual(h.steps,[{manifestVariant:'wasm',clientOptions:{executionProviders:['wasm']}}]);
  assert.match(h.scripts[0],/@1\.31\.1\/dist\/ort\.min\.js$/);
  assert.equal(h.notices[0].runtime.numThreads,4);
 }
});
test('initial GPU failure uses separately qualified WASM INT8 and never changes the bundle version',async()=>{
 const h=setup({a:adapter(),failGpu:true});await h.load();
 assert.deepEqual(h.steps.map(x=>x.manifestVariant),['webgpu','webgpu','webgpu','wasm']);
 assert.equal(h.steps[1].clientOptions.gpuTokenSelection,undefined);
 assert.equal(h.steps[2].clientOptions.gpuKvCache,undefined);
 assert.equal(h.notices[0].runtime.precision,'int8');assert.equal(h.notices[0].runtime.numThreads,1);
});
test('KV/device guard refuses unsupported token counts or binding sizes, retains a lower safe width',()=>{
 assert.equal(boundedGpuBatchWidth(4,20,1024,128*2**20),4);
 assert.equal(boundedGpuBatchWidth(4,20,1024,64*2**20),3);
 assert.equal(boundedGpuBatchWidth(4,2,1024,128*2**20),2);
 assert.equal(boundedGpuBatchWidth(4,20,2048,128*2**20),0);
 assert.equal(boundedGpuBatchWidth(4,20,1024,0),0);
});
