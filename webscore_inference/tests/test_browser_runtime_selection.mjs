import test from 'node:test';import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';import vm from 'node:vm';
import {cpuWasmThreadCount} from '../web/cpu-wasm-policy.mjs';
import {boundedGpuBatchWidth} from '../web/gpu-batch-pair.mjs';

const raw=await readFile(new URL('../../frontend/scripts/model.worker.js',import.meta.url),'utf8');
const adapter=(arch='rdna-3',fp16=true,software=false)=>({info:{architecture:arch,isFallbackAdapter:software},features:new Set(fp16?['shader-f16']:[])});
function setup({a,failGpu=false,failLoading=false,iso=false,hasJspi=false,failJspi=false,invalidJspiModel=false,host="127.0.0.1"}={}){
 const steps=[],scripts=[],notices=[],bridges=[];
 let jspiReleased=0;
 let listener;
 const runtime={env:{webgpu:{},wasm:{}}},jspiRuntime={env:{webgpu:{},wasm:{}}};
 const versionFactory={ async create(ort,opts) {
   steps.push(structuredClone(opts));
   bridges.push(ort===jspiRuntime?"jspi":"standard");
   if(ort===jspiRuntime&&failJspi)throw new Error("private provider-failure detail");
   if(failLoading||(failGpu&&opts.manifestVariant==='webgpu'))throw new Error('unsanitized opaque provider failure');
   return {ort,async dispose(){if(ort===jspiRuntime)jspiReleased++;},client:{
     localProjectionFusion:ort===jspiRuntime&&!invalidJspiModel,gpuSerialFeedbackBlock:ort===jspiRuntime?opts.clientOptions.gpuSerialFeedbackBlock:0,gpuKvCache:Boolean(opts.clientOptions.gpuKvCache),tokenSelector:opts.clientOptions.gpuTokenSelection?{}:null,
     gpuBatchPairs:Boolean(opts.clientOptions.gpuBatchPairs),gpuBatchWidth:opts.clientOptions.gpuBatchWidth}};
 }};
 const ctx=vm.createContext({versionFactory,cpuModule:{cpuWasmThreadCount},jspiRuntime,
  location:{hostname:host},WebAssembly:hasJspi?{Suspending(){},promising(){}}:{},crossOriginIsolated:iso,SharedArrayBuffer,
  navigator:{hardwareConcurrency:12,...(a?{gpu:{requestAdapter:async()=>a}}:{})},ort:runtime,
  console:{warn(){},error(){}},
  importScripts(x){scripts.push(x);},
  self:{addEventListener(_,f){listener=f},postMessage(o){notices.push(JSON.parse(JSON.stringify(o)))}},
 });
 const src=raw.replace('import(MODEL_MODULE_URL)','Promise.resolve({VersionedWebScoreClient: versionFactory})')
   .replace('import("/inference/cpu-wasm-policy.mjs")','Promise.resolve(cpuModule)')
   .replace('import(`${ORT_BASE}ort.jspi.min.mjs`)','Promise.resolve(jspiRuntime)');
 vm.runInContext(src,ctx);
 return {load:()=>listener({data:{type:'load'}}),steps,scripts,notices,runtime,bridges,jspiRuntime,get jspiReleased(){return jspiReleased;}};
}

test('hardware WebGPU FP16 dispatches once and enables universal batching and fusion',async()=>{
 const h=setup({a:adapter()});await Promise.all([h.load(),h.load()]);
 assert.equal(h.steps.length,1);assert.match(h.scripts[0],/@1\.31\.1\/dist\/ort\.webgpu\.min\.js$/);
 assert.deepEqual(h.steps[0],{manifestVariant:'webgpu',clientOptions:{executionProviders:['webgpu'],gpuKvCache:true,cacheEmbeddings:true,gpuTokenSelection:true,gpuBatchPairs:true,gpuBatchWidth:8,gpuFeedbackBlock:4,localLayerKv:true,gpuSerialFeedbackBlock:8,localProjectionFusion:true}});
 assert.equal(h.runtime.env.wasm.numThreads,1);assert.equal(h.runtime.env.webgpu.adapter.info.architecture,'rdna-3');
 assert.equal(h.notices[0].runtime.provider,'webgpu');
 assert.equal(h.notices[0].runtime.gpuBatchWidth,8);
});
test('non-RDNA3 hardware WebGPU GPUs (NVIDIA, Apple Silicon, Intel) also enable universal fast path',async()=>{
 const h=setup({a:adapter('turing')});await h.load();
 assert.equal(h.steps[0].clientOptions.gpuBatchPairs,true);
 assert.equal(h.steps[0].clientOptions.gpuFeedbackBlock,4);
 assert.equal(h.steps[0].clientOptions.localProjectionFusion,true);
 assert.equal(h.notices[0].runtime.gpuBatchWidth,8);
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

test('native JSPI works on any hardware WebGPU GPU and host when JSPI is available',async()=>{
 const yes=setup({a:adapter(),hasJspi:true});await yes.load();
 assert.deepEqual(yes.bridges,['jspi']);assert.deepEqual(yes.scripts,[]);
 assert.equal(yes.notices[0].runtime.runtimeBridge,'jspi');
 assert.equal(yes.steps[0].clientOptions.gpuSerialFeedbackBlock,8);
 assert.equal(yes.notices[0].runtime.serialPreludeFeedbackBlock,8);
 assert.equal(yes.jspiRuntime.env.wasm.numThreads,1);
 const otherGpu=setup({a:adapter('apple-gpu'),hasJspi:true,host:'app.example'});await otherGpu.load();
 assert.deepEqual(otherGpu.bridges,['jspi']);assert.equal(otherGpu.notices[0].runtime.runtimeBridge,'jspi');
});

test('JSPI load or exact interface failure falls back to standard native (not INT8)',async()=>{
 for(const opts of [{failJspi:true},{invalidJspiModel:true}]){
  const h=setup({a:adapter(),hasJspi:true,...opts});await h.load();
  assert.deepEqual(h.bridges,['jspi','standard']);
  assert.equal(h.scripts.length,1);
  assert.equal(h.notices[0].runtime.provider,'webgpu');
  assert.equal(h.notices[0].runtime.runtimeBridge,'asyncify');
  assert.equal(h.jspiReleased, opts.invalidJspiModel ? 1 : 0);
 }
});
