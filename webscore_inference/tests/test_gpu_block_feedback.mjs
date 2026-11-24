import test from 'node:test';
import assert from 'node:assert/strict';
import {runBatchPair} from '../web/gpu-batch-pair.mjs';
import {consumeGpuBlock} from '../web/gpu-block-feedback.mjs';

globalThis.GPUBufferUsage={STORAGE:1,COPY_SRC:2,COPY_DST:4,MAP_READ:8,UNIFORM:16};
globalThis.GPUMapMode={READ:1};
const B=4,V=9,D=2,E=2;
const commands=[[2],[7,2],[4,8,2],[7,8,7,8,2]];
const lut=Float32Array.from({length:V*D},(_,i)=>i+.25);
const pick=(values,mask,base)=>{
 let score=-Infinity,idx=-1;
 for(let j=0;j<V;j++)if(mask.has(j)===false&&values[base+j]>score){score=values[base+j];idx=j;}
 return idx;
};
function fixture({failMap=false,abort=null,dropAt=-1}={}){
 const calls={positions:[],gpuEmbeds:[],maps:0,baselineReadbacks:0},made=[];
 class Buffer {
  constructor(size,usage){this.buffer=new ArrayBuffer(size);this.usage=usage;this.destroyed=false;}
  async mapAsync(){calls.maps++;abort?.abort();if(failMap)throw new Error('Map failure');}
  getMappedRange(offset,size){return this.buffer.slice(offset,offset+size);}
  unmap(){}
  destroy(){this.destroyed=true;}
 }
 const device={
  createShaderModule:()=>({getCompilationInfo:async()=>({messages:[]})}),
  createComputePipelineAsync:async()=>({getBindGroupLayout:()=>({})}),
  createBuffer:({size,usage})=>{const b=new Buffer(size,usage);made.push(b);return b;},
  createBindGroup:({entries})=>entries.map(e=>e.resource.buffer),
  queue:{
   writeBuffer:(b,off,value)=>new Uint8Array(b.buffer).set(new Uint8Array(value.buffer,value.byteOffset,value.byteLength),off),
   submit:list=>list.flat().forEach(fn=>fn()),
  },
  createCommandEncoder:()=>{
   const f=[];
   return {
    beginComputePass:()=>{
     let g;
     return {setPipeline:()=>{},setBindGroup:(_,v)=>{g=v;},dispatchWorkgroups:(n)=>{
      f.push(()=>{
       const [src,valid,cfg,hist,end,table,ready]=g.map(x=>x.buffer);
       const [v,offset,stride,width,d,slot,eos]=new Uint32Array(cfg);
       assert.equal(n,width);assert.equal(v,V);assert.equal(d,D);
       const scores=new Float32Array(src), mask=new Set(Array.from(new Uint32Array(valid).entries()).filter(([,x])=>!x).map(([i])=>i));
       const history=new Int32Array(hist), state=new Int32Array(end), out=new Float32Array(ready);
       for(let row=0;row<n;row++){
        let t=state[row];
        if(t===-2)t=pick(scores,mask,offset+row*stride);
        if(t===-1||t===eos)state[row]=t;
        if(calls.positions.length-1!==dropAt)history[slot*n+row]=t;
        const j=t>=0?t:eos;out.set(new Float32Array(table).subarray(j*d,(j+1)*d),row*d);
       }
      });
     },end:()=>{}};
    },
    copyBufferToBuffer:(a,x,b,y,n)=>{f.push(()=>new Uint8Array(b.buffer,y,n).set(new Uint8Array(a.buffer,x,n)));},
    finish:()=>f,
   };
  },
 };
 class Tensor{
  constructor(type,data,dims,buf=null){Object.assign(this,{type,dims,gpuBuffer:buf,location:buf?'gpu-buffer':'cpu',value:data});}
  get data(){if(this.gpuBuffer)throw Error('GPU input forced host readback');return this.value;}
  dispose(){this.disposed=true;}
  static fromGpuBuffer(buf,{dataType,dims}){return new Tensor(dataType,null,dims,buf);}
 }
 const enc=(data,dims)=>new Tensor('float32',data,dims);
 const gpu=(data,dims)=>{
  const b=new Buffer(data.byteLength,1);
  new Float32Array(b.buffer).set(data);
  return new Tensor('float32',null,dims,b);
 };
 const c={
  ort:{env:{webgpu:{device}},Tensor},
  lookup:{bosId:1,eosId:E,idToEvent:Array.from({length:V},(_,id)=>({id,kind:id===5?'inst':id===6?'drum':'pitch',value:id===5?16:id}))},
  maskedIds:new Set([0,1,3]),arch:{dim:D,num_heads:1,num_layers:1,num_kv_heads:1},
  mel:{computeChunkLogMel:()=>({data:new Float32Array(2),dims:[1,1,2]})},
  encoderSession:{run:async()=>({cond_embeds:enc(new Float32Array(B*2*D),[B,2,D])})},
  embedTokenIds:async ids=>enc(Float32Array.from(ids.flatMap(i=>[lut[i*D],lut[i*D+1]])),[1,ids.length,D]),
  embeddingTable:lut,
  decoderSession:{run:async({inputs_embeds:emb,past_k:past,position_offset:pos})=>{
   const stored=past.dims[3],step=Math.max(0,stored-2),n=emb.dims[1];
   assert.equal(Number(pos.data[0]),stored);calls.positions.push(stored);
   if(emb.gpuBuffer)calls.gpuEmbeds.push(Array.from(new Float32Array(emb.gpuBuffer.buffer)));
   const vals=new Float32Array(B*n*V).fill(NaN);
   for(let row=0;row<B;row++){
    const base=((row+1)*n-1)*V;
    // Prefix has non-final time rows; wrong offsets yield NaN/no selection.
    vals[base+5]=Infinity; vals[base+6]=Infinity;
    vals[base+(commands[row][step]??E)]=11;
   }
   const kv=()=>gpu(new Float32Array(B*(stored+n)*D),[1,B,1,stored+n,D]);
   return {logits:gpu(vals,[B,n,V]),present_k:kv(),present_v:kv()};
  }},
 };
 const selector={
  setMask(mask){this.mask=mask;},
  async runRows(src,offset,stride,width){calls.baselineReadbacks++;return Int32Array.from({length:width},(_,i)=>pick(new Float32Array(src.buffer),this.mask,offset+i*stride));},
 };
 return {c,calls,selector,made};
}
const wave=Array.from({length:B},()=>new Float32Array(1));
const opts=(f,extras)=>({allowedPrograms:[0],selector:f.selector,maxGenLen:8,...extras});
test('block feedback preserves exact per-lane EOS tokens and mask; next embedding stays resident',async()=>{
 const prior=fixture(), expected=await runBatchPair(prior.c,wave,opts(prior));
 assert.deepEqual(expected.tokens,commands);
 assert.equal(prior.calls.baselineReadbacks,5);
 for(const k of [1,2,4,8]){
  const f=fixture();f.c.gpuFeedbackBlock=k;
  const got=await runBatchPair(f.c,wave,opts(f));
  assert.deepEqual(got.tokens,expected.tokens);assert.deepEqual(got.eos,expected.eos);
  assert.equal(f.calls.maps,Math.ceil(5/k));
  assert.equal(f.calls.baselineReadbacks,0);
  assert.ok(f.calls.gpuEmbeds.length>0);
  assert.deepEqual(f.calls.gpuEmbeds[0],Array.from([2,7,4,7].flatMap(i=>[lut[i*D],lut[i*D+1]])));
  assert.ok(f.made.every(b=>b.destroyed));
 }
});
test('partial block at cap preserves unended lanes and never speculates past cap',async()=>{
 const f=fixture();f.c.gpuFeedbackBlock=4;
 const got=await runBatchPair(f.c,wave,opts(f,{maxGenLen:3}));
 assert.deepEqual(got.tokens,commands.map(t=>t.slice(0,3)));assert.deepEqual(got.eos,[true,true,true,false]);
 assert.equal(got.decoderCalls,3);assert.equal(f.calls.maps,1);
});
test('validation failure redoes only the group using the baseline selector, not partial outputs',async()=>{
 for(const bad of [{failMap:true},{dropAt:1}]){
  const f=fixture(bad);f.c.gpuFeedbackBlock=4;
  const got=await runBatchPair(f.c,wave,opts(f));
  assert.deepEqual(got.tokens,commands);assert.equal(f.c.gpuFeedbackDisabled,true);
  assert.equal(f.c.gpuFeedbackStats.failed,1);assert.equal(f.calls.baselineReadbacks,5);
  assert.ok(f.made.every(b=>b.destroyed));
 }
});
test('abort during a readback does not rerun or stitch a partial group',async()=>{
 const ctrl=new AbortController();const f=fixture({abort:ctrl});f.c.gpuFeedbackBlock=4;
 await assert.rejects(runBatchPair(f.c,wave,opts(f,{signal:ctrl.signal})),{name:'AbortError'});
 assert.equal(f.calls.baselineReadbacks,0);assert.ok(f.made.every(b=>b.destroyed));
});
test('host accepts only in-range unmasked active history and ignores frozen trailing rows',()=>{
 const state={tokens:[[],[]],done:[false,false]};
 consumeGpuBlock(state,Int32Array.of(2,7,-34,2),2,V,new Set([0]));
 assert.deepEqual(state,{tokens:[[2],[7,2]],done:[true,true]});
 for(const x of [-1,0,V])assert.throws(()=>consumeGpuBlock({tokens:[[]],done:[false]},Int32Array.of(x),2,V,new Set([0])));
});
