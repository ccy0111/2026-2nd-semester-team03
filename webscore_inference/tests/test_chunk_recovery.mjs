import test from 'node:test';
import assert from 'node:assert/strict';
import {WebScoreClient,TokenLimitError} from '../web/model-client.mjs';
const tok={pad_id:0,bos_id:1,eos_id:2,unk_id:3,id_to_token:['<pad>','<bos>','<eos>','<unk>','inst:0','pitch:60','vel:1','vel:0','time:0','time:100','tie','pitch:62']};
const held=[10,8,4,6,5,2], ended=[10,8,4,6,11,9,7,11,2];
function client(run){const c=new WebScoreClient({}, {tokenizer:tok,arch:{}});c.transcribeChunk=run;return c;}
test('normal chunks preserve carry and add ok statuses without extra calls',async()=>{
 const calls=[];const c=client(async(_,opts)=>{calls.push(opts);return calls.length===1?held:[4,5,10,9,4,7,5,2];});
 const r=await c.transcribePcm(new Float32Array(160000),{boundaryMode:"legacy"});assert.equal(calls.length,2);assert.deepEqual(calls[1].promptTokens,[4,5,10]);assert.deepEqual(r.chunkStatuses.map(x=>x.status),['ok','ok']);assert.equal(r.notes[0].offset,6);
});
test('recover only affected chunk, drop old carry and preserve previous/following notes',async()=>{
 let n=0;const calls=[],progress=[];const c=client(async(_,opts)=>{calls.push(opts);n++;if(n===2)throw new TokenLimitError();return n===1?held:ended;});
 const r=await c.transcribePcm(new Float32Array(240000),{maxGenLen:17,onProgress:x=>progress.push(x)});
 assert.equal(n,4);assert.deepEqual(calls[2].promptTokens,[10]);assert.ok(calls.every(x=>x.maxGenLen===17));assert.deepEqual(r.chunkStatuses.map(x=>x.status),['ok','recovered','ok']);assert.equal(r.notes.find(x=>x.pitch===60).offset,5);assert.deepEqual(r.notes.filter(x=>x.pitch===62).map(x=>x.onset),[5,10]);assert.equal(progress.length,3);assert.equal(progress[1].chunkStatus.boundaryReset,true);
});
for(const failedIndex of [0,1,2])test(`double failure at chunk ${failedIndex} skips only that interval`,async()=>{
 let call=0;const calls=[];const c=client(async(_,o)=>{calls.push(o);const current=call++;if(current===failedIndex||current===failedIndex+1)throw new TokenLimitError();return held;});
 const r=await c.transcribePcm(new Float32Array(240000),{boundaryMode:"legacy"});assert.equal(call,4);assert.deepEqual(r.tokensByChunk[failedIndex],[]);assert.equal(r.chunkStatuses[failedIndex].status,'failed');assert.equal(r.chunkStatuses[failedIndex].attempts,2);assert.equal(r.chunkStatuses.length,3);assert.ok(r.notes.every(n=>n.offset<=15));if(failedIndex<2)assert.deepEqual(calls[failedIndex+2].promptTokens,[10]);
});
test('a short final failed chunk reports actual time instead of padding',async()=>{
 const c=client(async()=>{throw new TokenLimitError();});const r=await c.transcribePcm(new Float32Array(16000));assert.equal(r.chunkStatuses[0].endSec,1);assert.deepEqual(r.notes,[]);assert.equal(r.tokensByChunk.length,1);
});
for(const error of [new Error('network'),new Error('Model did not emit EOS within the per-chunk token limit')])test(`non-cap errors propagate: ${error.message}`,async()=>{
 let calls=0;const c=client(async()=>{calls++;throw error;});await assert.rejects(c.transcribePcm(new Float32Array(80000)),x=>x===error);assert.equal(calls,1);assert.equal(c.transcribing,false);
});
test('abort during failed chunk never retries or returns a partial success',async()=>{
 const ac=new AbortController();let calls=0;const c=client(async()=>{calls++;ac.abort();throw new TokenLimitError();});await assert.rejects(c.transcribePcm(new Float32Array(80000),{signal:ac.signal}),{name:'AbortError'});assert.equal(calls,1);assert.equal(c.transcribing,false);
});
test('retry runtime error and retry abort both propagate, no skip',async()=>{
 for(const abort of [false,true]){const ac=new AbortController();let n=0;const error=new Error('device lost');const c=client(async()=>{if(++n===1)throw new TokenLimitError();if(abort)ac.abort();throw error;});await assert.rejects(c.transcribePcm(new Float32Array(80000),{signal:ac.signal}),x=>abort?x.name==='AbortError':x===error);assert.equal(n,2);assert.equal(c.transcribing,false);}
});
test('pre-abort, invalid limits and progress failures do not trigger recovery',async()=>{
 let calls=0;const c=client(async()=>{calls++;return ended;});const ac=new AbortController();ac.abort();await assert.rejects(c.transcribePcm(new Float32Array(80000),{signal:ac.signal}),{name:'AbortError'});for(const maxGenLen of [0,-1,1.5,Infinity])await assert.rejects(c.transcribePcm(new Float32Array(80000),{maxGenLen}),RangeError);assert.equal(calls,0);await assert.rejects(c.transcribePcm(new Float32Array(80000),{onProgress:()=>{throw new Error('UI');}}),/UI/);assert.equal(calls,1);assert.equal(c.transcribing,false);
});
test('real chunk loop uses typed cap error and never embeds an unused final token',async()=>{
 class Tensor{constructor(type,data,dims){Object.assign(this,{type,data,dims});}}
 let embeddings=0,runs=0;
 const c=new WebScoreClient({Tensor},{tokenizer:tok,arch:{dim:1,num_heads:1,num_kv_heads:1,num_layers:1},encoderSession:{run:async()=>({cond_embeds:new Tensor('float32',Float32Array.of(0),[1,1,1])})},decoderSession:{run:async()=>{runs++;const logits=new Float32Array(tok.id_to_token.length);logits[5]=1;return {logits:new Tensor('float32',logits,[1,1,logits.length]),present_k:new Tensor('float32',[],[1,1,1,2,1]),present_v:new Tensor('float32',[],[1,1,1,2,1])};}}});
 c.mel={computeChunkLogMel:()=>({data:Float32Array.of(0),dims:[1,1,1]})};c.embedTokenIds=async ids=>{embeddings++;return new Tensor('float32',new Float32Array(ids.length),[1,ids.length,1]);};
 await assert.rejects(c.transcribeChunk(new Float32Array(80000),{maxGenLen:1}),x=>x instanceof TokenLimitError&&x.code==='TOKEN_LIMIT_EXHAUSTED');assert.equal(runs,1);assert.equal(embeddings,1);
});
test('runaway positive-velocity partial tokens never leak notes into a skipped chunk',async()=>{
 let call=0;const runaway=[8,4,6,11,11,11,11,11];
 const c=client(async()=>{call++;if(call===2||call===3){const error=new TokenLimitError();error.partialTokens=runaway;throw error;}return call===1?held:ended;});
 const r=await c.transcribePcm(new Float32Array(240000));assert.equal(r.chunkStatuses[1].status,'failed');assert.deepEqual(r.tokensByChunk[1],[]);assert.equal(r.notes.length,2);assert.ok(r.notes.every(n=>n.onset<5||n.onset>=10));assert.equal(r.notes.find(n=>n.pitch===60).offset,5);
});
