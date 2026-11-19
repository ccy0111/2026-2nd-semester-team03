import test from 'node:test';
import assert from 'node:assert/strict';
import {WebScoreClient} from '../web/model-client.mjs';
const tokenizer={pad_id:0,bos_id:1,eos_id:2,unk_id:3,id_to_token:['<pad>','<bos>','<eos>','<unk>','tie']};
function harness({eos=10,fail=0,hostShapes=false,halfKv=false}={}) {
 let runs=0,reads=0;const allocated=[];
 class Tensor {
  constructor(type,data,dims){this.type=type;this.hidden=data;this.dims=dims;this.location='cpu';this.freed=false;}
  get data(){assert.equal(this.location,'cpu','No per-token GPU .data access');return this.hidden;}
  async getData(release){reads++;assert.equal(release,true);this.dispose();this.location='cpu';return this.hidden;}
  dispose(){assert.equal(this.freed,false,'No double disposal');this.freed=true;}
 }
 const gpu=(data,dims)=>{const t=new Tensor('int64',data,dims);t.location='gpu-buffer';allocated.push(t);return t;};
 let previous;
 const dec={async run(f){runs++;
 if(hostShapes){
  const seq=f.prefix_embeds.dims[1]+f.input_ids.dims[1],total=f.past_k.dims[3]+seq;
  for(const [name,value] of Object.entries({shape_q:[1,seq,1,1],shape_kv:[1,seq,1,1],shape_expand:[1,1,1,total,1],shape_expanded_kv:[1,1,total,1],shape_out:[1,seq,1]}))assert.deepEqual(Array.from(f[name].data,Number),value);
  assert.deepEqual(Array.from(f.position_range.data),Array.from({length:seq},(_,i)=>i));
 }else assert.equal(f.position_range,undefined);
 if(fail===runs)throw new Error('injected');if(previous)assert.equal(f.input_ids,previous,'Direct GPU token feedback');
  if(halfKv){assert.equal(f.past_k.type,'float16');assert.equal(f.past_v.type,'float16');}
 const id=runs===eos?2:4;const next=gpu(BigInt64Array.of(BigInt(id)),[1,1]);previous=next;
  const hist=BigInt64Array.from([...f.token_history.hidden,BigInt(id)]);const len=f.past_k.dims[3]+f.prefix_embeds.dims[1]+f.input_ids.dims[1];
  const k=gpu([], [1,1,1,len,1]),v=gpu([], [1,1,1,len,1]);if(halfKv){k.type='float16';v.type='float16';}
  return {next_ids:next,updated_history:gpu(hist,[1,hist.length]),present_k:k,present_v:v};}};
 const c=new WebScoreClient({Tensor},{tokenizer,arch:{dim:1,num_layers:1,num_heads:1,num_kv_heads:1},decoderSession:dec,gpuKvCache:true,fusedGreedy:true,chunkedFeedback:true,feedbackReadInterval:8,feedbackHostShapes:hostShapes,feedbackHalfKv:halfKv});
 return {c,prefix:gpu([], [1,2,1]),stats:()=>({runs,reads,live:allocated.filter(t=>!t.freed).length})};
}
test('GPU feedback reads only each 8 tokens, truncates after first EOS, disposes outputs',async()=>{
 const h=harness();const result=await h.c._transcribeFeedback(h.prefix,{promptTokens:[],maxGenLen:32});assert.deepEqual(result,[...Array(9).fill(4),2]);assert.deepEqual(h.stats(),{runs:16,reads:2,live:0});
});
test('GPU feedback final partial read checks EOS at generation cap',async()=>{
 const h=harness({eos:10});await h.c._transcribeFeedback(h.prefix,{promptTokens:[],maxGenLen:10});assert.deepEqual(h.stats(),{runs:10,reads:2,live:0});
});
test('GPU feedback error releases tensors without any token readback',async()=>{
 const h=harness({fail:5});await assert.rejects(h.c._transcribeFeedback(h.prefix,{promptTokens:[],maxGenLen:16}),/injected/);assert.deepEqual(h.stats(),{runs:5,reads:0,live:0});
});
test('GPU feedback missing EOS still enforces cap and releases tensors',async()=>{
 const h=harness({eos:100});await assert.rejects(h.c._transcribeFeedback(h.prefix,{promptTokens:[],maxGenLen:10}),/did not emit EOS/);assert.deepEqual(h.stats(),{runs:10,reads:2,live:0});
});

test('native int64 session option is confined to explicitly opted-in GPU feedback decoder',async()=>{
 const previousFetch=globalThis.fetch;
 globalThis.fetch=async url=>({ok:true,json:async()=>url.endsWith('tokenizer.json')?tokenizer:{dim:1,num_heads:1,num_layers:1,num_kv_heads:1}});
 try {
  for(const [chunkedFeedback,gpuKvCache] of [[true,true],[true,false],[false,true],[false,false]]) {
   const calls=[];
   const ort={InferenceSession:{create:async(url,options)=>{calls.push({url,options});return {release:async()=>{}};}}};
   const c=await WebScoreClient.create(ort,{baseUrl:'/test',useInt8:false,fusedGreedy:true,chunkedFeedback,gpuKvCache,executionProviders:gpuKvCache?['webgpu']:['wasm']});
   for(const {url,options} of calls){
    if(chunkedFeedback&&gpuKvCache&&url.endsWith('decoder_feedback.onnx')){
     assert.deepEqual(options.extra,{'ep.webgpuexecutionprovider.enableInt64':'1'});
     assert.equal(options.preferredOutputLocation.next_ids,'gpu-buffer');
     assert.equal(options.preferredOutputLocation.updated_history,'gpu-buffer');
    }else assert.equal(options.extra,undefined);
   }
   await c.dispose();
  }
 }finally{globalThis.fetch=previousFetch;}
});

test('shared host shapes cover prefill, incremental positions and history resets',async()=>{const h=harness({hostShapes:true});await h.c._transcribeFeedback(h.prefix,{promptTokens:[4],maxGenLen:16});assert.deepEqual(h.stats(),{runs:16,reads:2,live:0});});

test('FP16 feedback KV uses explicit half-type state and retains feedback semantics',async()=>{const h=harness({hostShapes:true,halfKv:true});const tokens=await h.c._transcribeFeedback(h.prefix,{promptTokens:[],maxGenLen:16});assert.equal(tokens.length,10);assert.deepEqual(h.stats(),{runs:16,reads:2,live:0});});
test('FP16 KV requires the dedicated host-shape graph option',async()=>{
 await assert.rejects(WebScoreClient.create({}, {useInt8:false,fusedGreedy:true,chunkedFeedback:true,feedbackHalfKv:true}),/FP16 KV requires/);
});
test('baseline factory needs no optimized assets and missing opted-in graph is explicit',async()=>{
 const previousFetch=globalThis.fetch;const names=[];
 globalThis.fetch=async url=>({ok:true,json:async()=>url.endsWith('tokenizer.json')?tokenizer:{dim:1,num_heads:1,num_layers:1,num_kv_heads:1}});
 try {
  const ort={InferenceSession:{create:async url=>{names.push(url.split('/').at(-1));return {release:async()=>{}};}}};
  const c=await WebScoreClient.create(ort,{baseUrl:'/test'});
  assert.deepEqual(names.sort(),['decoder_kv.int8.onnx','mel_encoder.onnx','token_embed.onnx']);await c.dispose();
  await assert.rejects(WebScoreClient.create(ort,{useInt8:false,fusedGreedy:true,chunkedFeedback:true,feedbackHostShapes:true,modelUrls:{'config.json':'/config.json','tokenizer.json':'/tokenizer.json','mel_encoder.onnx':'/mel_encoder.onnx'}}),/Missing model URL for decoder_feedback_hostshapes.onnx/);
 }finally{globalThis.fetch=previousFetch;}
});
