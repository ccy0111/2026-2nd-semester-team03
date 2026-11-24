// Opt-in checkpoint experiment. Keep GPU token IDs / embeddings for a short block.
const shaders = new WeakMap();
const shader = `
struct Params { vocab: u32, offset: u32, stride: u32, width: u32, hidden:u32, slot:u32, eos:u32, pad:u32 }
@group(0) @binding(0) var<storage, read> logits: array<u32>;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<uniform> cfg: Params;
@group(0) @binding(3) var<storage, read_write> history: array<i32>;
@group(0) @binding(4) var<storage, read_write> terminal: array<i32>;
@group(0) @binding(5) var<storage, read> lookup: array<u32>;
@group(0) @binding(6) var<storage, read_write> embedding: array<u32>;
var<workgroup> scores: array<u32,128>;
var<workgroup> indices: array<u32,128>;
var<workgroup> winner: i32;
const NONE: u32 = 0xffffffffu;
fn better(a: u32, ai: u32, b: u32, bi: u32) -> bool {
 return ai != NONE && (bi == NONE || a > b || (a == b && ai < bi));
}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
 let row=group.x;
 var best=0u;
 var idx=NONE;
 for (var i=lane; i<cfg.vocab; i=i+128u) {
  let raw=logits[cfg.offset+row*cfg.stride+i];
  let bits=select(raw,0u,(raw & 0x7fffffffu)==0u);
  let x=select(bits ^ 0x80000000u,~bits,(bits & 0x80000000u)!=0u);
  if (valid[i]!=0u && (bits & 0x7fffffffu)<=0x7f800000u &&
      bits!=0xff800000u && better(x,i,best,idx)) {best=x;idx=i;}
 }
 scores[lane]=best; indices[lane]=idx;
 workgroupBarrier();
 for(var hop=64u;hop>0u;hop=hop>>1u){
  if(lane<hop && better(scores[lane+hop],indices[lane+hop],scores[lane],indices[lane])){
    scores[lane]=scores[lane+hop];indices[lane]=indices[lane+hop];
  }
  workgroupBarrier();
 }
 if(lane==0u){
  let state=terminal[row];
  winner=select(state,bitcast<i32>(indices[0]),state==-2);
  if(state==-2 && (winner==-1 || winner==i32(cfg.eos))){terminal[row]=winner;}
  history[cfg.slot*cfg.width+row]=winner;
 }
 workgroupBarrier();
 let feed=select(cfg.eos,u32(winner),winner>=0);
 // Copy exact embedding bits, including signed zero.
 for(var j=lane;j<cfg.hidden;j=j+128u){
  embedding[row*cfg.hidden+j]=lookup[feed*cfg.hidden+j];
 }
}`;

async function pipelineFor(device) {
 if(!shaders.has(device))shaders.set(device,(async()=>{
  const module=device.createShaderModule({code:shader});
  const info=await module.getCompilationInfo();
  if(info.messages.some(m=>m.type==="error"))throw new Error(info.messages.filter(m=>m.type==="error").map(m=>m.message).join("; "));
  return device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
 })());
 return shaders.get(device);
}

export class GpuBlockFeedback {
 static async create(c,b,d,masked,k){
  if(![1,2,4,8].includes(k)||b<2||b>4)throw new Error("Unsupported GPU feedback size");
  if(typeof c.ort.Tensor.fromGpuBuffer!=="function")throw new Error("ORT external GPU tensors unavailable");
  const device=await c.ort.env.webgpu.device, pipe=await pipelineFor(device);
  const v=c.lookup.idToEvent.length;
  if(!(c.embeddingTable instanceof Float32Array)||c.embeddingTable.length!==v*d)throw new Error("FP32 embedding LUT unavailable");
  const f=new GpuBlockFeedback(device,pipe,b,d,v,c.lookup.eosId,k);
  try { f.initialize(c.ort.Tensor,c.embeddingTable,masked);return f; }
  catch (error) { f.dispose();throw error; }
 }
 constructor(device,pipeline,b,d,v,eos,k){
  Object.assign(this,{device,pipeline,b,d,v,eos,k,buffers:[]});
 }
 buffer(size,usage){
  const b=this.device.createBuffer({size,usage});this.buffers.push(b);return b;
 }
 initialize(Tensor,lut,masked){
  const {STORAGE,COPY_SRC,COPY_DST,MAP_READ,UNIFORM}=GPUBufferUsage;
  const d=this.device;
  this.valid=this.buffer(this.v*4,STORAGE|COPY_DST);
  this.lut=this.buffer(lut.byteLength,STORAGE|COPY_DST);
  this.terminal=this.buffer(this.b*4,STORAGE|COPY_DST);
  this.config=this.buffer(32,UNIFORM|COPY_DST);
  this.history=this.buffer(this.b*this.k*4,STORAGE|COPY_SRC|COPY_DST);
  this.emptyHistory=new Int32Array(this.b*this.k).fill(-1);
  this.staging=this.buffer(this.b*this.k*4,MAP_READ|COPY_DST);
  this.embedding=this.buffer(this.b*this.d*4,STORAGE|COPY_SRC|COPY_DST);
  const valid=new Uint32Array(this.v).fill(1);
  for(const id of masked)if(id>=0&&id<this.v)valid[id]=0;
  d.queue.writeBuffer(this.valid,0,valid);
  d.queue.writeBuffer(this.terminal,0,new Int32Array(this.b).fill(-2));
  d.queue.writeBuffer(this.lut,0,lut);
  this.input=Tensor.fromGpuBuffer(this.embedding,{dataType:"float32",dims:[this.b,1,this.d]});
 }
 emit(logits,rows,slot){
  const buffer=logits.gpuBuffer,d=this.device;
  if(logits.type!=="float32"||logits.location!=="gpu-buffer"||!(buffer?.usage&GPUBufferUsage.STORAGE)||
     !Number.isInteger(slot)||slot<0||slot>=this.k)throw new Error("Invalid GPU feedback logits");
  if(slot===0)d.queue.writeBuffer(this.history,0,this.emptyHistory);
  d.queue.writeBuffer(this.config,0,Uint32Array.of(this.v,(rows-1)*this.v,rows*this.v,this.b,this.d,slot,this.eos,0));
  const bound=[{buffer,size:this.b*rows*this.v*4},...[this.valid,this.config,this.history,this.terminal,this.lut,this.embedding].map(buffer=>({buffer}))];
  const g=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:bound.map((resource,binding)=>({binding,resource}))});
  const cmd=d.createCommandEncoder();
  const pass=cmd.beginComputePass();
  pass.setPipeline(this.pipeline);pass.setBindGroup(0,g);pass.dispatchWorkgroups(this.b);pass.end();
  d.queue.submit([cmd.finish()]);
 }
 async read(count){
  if(!Number.isInteger(count)||count<1||count>this.k)throw new Error("Invalid GPU feedback read length");
  const bytes=count*this.b*4,d=this.device,cmd=d.createCommandEncoder();
  cmd.copyBufferToBuffer(this.history,0,this.staging,0,bytes);d.queue.submit([cmd.finish()]);
  await this.staging.mapAsync(GPUMapMode.READ,0,bytes);
  try{return new Int32Array(this.staging.getMappedRange(0,bytes)).slice();}
  finally{this.staging.unmap();}
 }
 dispose(){this.input?.dispose();for(const b of this.buffers)b.destroy();}
}

// Shared framing logic for sparse reads (step-major [step,row]) and tests.
export function consumeGpuBlock({tokens,done},history,eos,vocab,masked){
 const n=done.length;
 if(history.length%n)throw new Error("Incomplete GPU feedback block");
 for(let i=0;i<history.length;i++){
  const row=i%n;
  if(done[row])continue;
  const id=history[i];
  if(!Number.isInteger(id)||id<0||id>=vocab||masked?.has(id))throw new Error("Invalid GPU feedback token");
  tokens[row].push(id);
  done[row]=id===eos;
 }
}
