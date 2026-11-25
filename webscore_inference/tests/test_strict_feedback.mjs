import test from 'node:test';
import assert from 'node:assert/strict';
import { WebScoreClient } from '../web/model-client.mjs';
import {GpuBlockFeedback} from '../web/gpu-block-feedback.mjs';

const tokenizer={pad_id:0,bos_id:1,eos_id:2,unk_id:3,id_to_token:[
 '<pad>','<bos>','<eos>','<unk>','tie','inst:0','pitch:60','time:0','vel:1','time:100','vel:0','inst:1'
]};
const first=[7,5,8,6,2],second=[9,5,10,6,2];
const audio=new Float32Array(160000);
const ort={Tensor:class {constructor(type,data,dims){Object.assign(this,{type,data,dims});}}};
function mock(block, {failure=false,abort=null,forceNoEos=false}={}) {
 const c=new WebScoreClient(ort,{tokenizer,arch:{dim:2,num_heads:1,num_kv_heads:1,num_layers:1},gpuKvCache:true});
 c.localLayerKv=true;c.gpuSerialFeedbackBlock=block;c.serialFeedbackStats={maps:0,mapMs:0};
 let idx=0, step=0, encoderCalls=0,bufferDisposals=0,selectorCalls=0;
 const prefillPrompts=[];
 c.mel={computeChunkLogMel(){return {data:new Float32Array(3),dims:[1,3]}}};
 c.encoderSession={async run(){encoderCalls++;return {cond_embeds:new ort.Tensor('float32',new Float32Array(6),[1,3,2])};}};
 c.tokenSelector={setMask(){},async run(logit){selectorCalls++;return logit.id;}};
 c.embedTokenIds=async ids=>{
   if(ids[0]===1)prefillPrompts.push([...ids]);
   return new ort.Tensor('float32',new Float32Array(ids.length*2),[1,ids.length,2]);
 };
 const mk=(shape)=>({dims:shape,location:'gpu-buffer',dispose(){bufferDisposals++}});
 c.decoderSession={async run(feed){
   const [b,s]=feed.inputs_embeds.dims, past=feed.past_k.dims[3];
   assert.equal(b,1);
   if(past===0)idx=encoderCalls-1,step=0;
   const stack=idx===0?first:second;
   const id=forceNoEos?7:(stack[step++]??2);
   return {present_k:mk([1,1,1,past+s,2]),present_v:mk([1,1,1,past+s,2]),logits:{...mk([1,s,tokenizer.id_to_token.length]),type:'float32',gpuBuffer:{id}}};
 }};
 let list=[],terminal=null,disposed=0;
 const feed={input:new ort.Tensor('float32',new Float32Array(2),[1,1,2]),
   emit(t){const id=terminal??t.gpuBuffer.id;if(id===2)terminal=2;list.push(id);},
   async read(n){if(abort)abort.controller.abort(abort.reason);if(failure)throw Error('simulated GPU map failure');return Int32Array.from(list.splice(0,n));},
   dispose(){disposed++;}
 };
 c.hook=async function(){
  terminal=null;list=[];
  return feed;
 };
 c.audit=()=>({prefillPrompts,encoderCalls,bufferDisposals,selectorCalls,disposed});
 return c;
}
async function trial(c,fn) {
 const saved=GpuBlockFeedback.create;
 try {GpuBlockFeedback.create=c.hook;return await fn(c);}
 finally {GpuBlockFeedback.create=saved;}
}
test('two nonempty changing strict prefills, extra EOS rows discarded, exact MIDI and one map per block',async()=>{
 const old=mock(0),a=await old.transcribePcm(audio,{boundaryMode:'prelude-forcing',allowedPrograms:[0]});
 for(const block of [4,8]){
  const c=mock(block),b=await trial(c,x=>x.transcribePcm(audio,{boundaryMode:'prelude-forcing',allowedPrograms:[0]}));
  assert.deepEqual(b,a);
  assert.deepEqual(c.audit().prefillPrompts,[[1,4],[1,5,6,4]]);
  assert.equal(c.audit().disposed,2);
  assert.equal(c.audit().selectorCalls,0);
  assert.equal(c.serialFeedbackStats.maps,block===4?4:2);
 }
});
test('block map failure restarts same chunk from fresh KV set, retains only encoder prefix',async()=>{
 const a=await mock(0).transcribePcm(audio,{boundaryMode:'prelude-forcing'});
 const c=mock(4,{failure:true}),b=await trial(c,x=>x.transcribePcm(audio,{boundaryMode:'prelude-forcing'}));
 assert.deepEqual(b,a);assert.ok(c.serialFeedbackDisabled);
 assert.equal(c.audit().encoderCalls,2);assert.equal(c.audit().selectorCalls,10);assert.equal(c.audit().disposed,1);
 assert.deepEqual(c.audit().prefillPrompts,[[1,4],[1,4],[1,5,6,4]]);
});
test('cancellation during sparse read does not retry or advance any chunk',async()=>{
 const controller=new AbortController(),reason={why:'Canceled'};
 const c=mock(4,{abort:{controller,reason}});
 let progress=0;
 await assert.rejects(trial(c,x=>x.transcribePcm(audio,{boundaryMode:'prelude-forcing',signal:controller.signal,onProgress(){progress++;}})),v=>v===reason);
 assert.equal(progress,0);assert.equal(c.audit().encoderCalls,1);assert.equal(c.audit().disposed,1);assert.equal(c.audit().selectorCalls,0);
});
test('a partial block at the limit keeps the typed EOS retry and matching failed status',async()=>{
 const old=mock(0,{forceNoEos:true}),a=await old.transcribePcm(audio,{boundaryMode:'prelude-forcing',maxGenLen:3});
 const c=mock(4,{forceNoEos:true}),b=await trial(c,x=>x.transcribePcm(audio,{boundaryMode:'prelude-forcing',maxGenLen:3}));
 assert.deepEqual(b,a);assert.equal(c.audit().encoderCalls,2);assert.equal(c.audit().disposed,4);
 assert.equal(c.serialFeedbackStats.maps,4);
});
test('pitch-tie and legacy never request the strict feedback lane',async()=>{
 for(const boundaryMode of ['pitch-tie','legacy']){
  const c=mock(4);
  c.hook=()=>{throw new Error('unexpected feedback route');};
  const b=await trial(c,x=>x.transcribePcm(audio,{boundaryMode}));
  assert.equal(c.serialFeedbackDisabled,undefined);assert.equal(c.audit().disposed,0);
  assert.equal(b.chunkStatuses.length,2);
 }
});
