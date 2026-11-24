const shader = `
struct Params { count: u32, offset: u32, stride: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> logits: array<u32>;
@group(0) @binding(1) var<storage, read> valid: array<u32>;
@group(0) @binding(2) var<uniform> cfg: Params;
@group(0) @binding(3) var<storage, read_write> winner: array<i32>;
var<workgroup> scores: array<u32,128>;
var<workgroup> indices: array<u32,128>;
const NONE: u32 = 0xffffffffu;
fn better(a: u32, ai: u32, b: u32, bi: u32) -> bool {
  return ai != NONE && (bi == NONE || a > b || (a == b && ai < bi));
}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  var best = 0u;
  var idx = NONE;
  for (var i = lane; i < cfg.count; i = i + 128u) {
    let raw = logits[cfg.offset+group.x*cfg.stride+i];
    let bits = select(raw,0u,(raw & 0x7fffffffu) == 0u);
    // Compare ordered IEEE754 bits so subnormals and signed zero match CPU.
    let x = select(bits ^ 0x80000000u,~bits,(bits & 0x80000000u) != 0u);
    // JS ">" rejects NaNs and -Infinity (including when all candidates fail);
    // +Infinity ties and signed zero ties still favor the first token id.
    if (valid[i] != 0u && (bits & 0x7fffffffu) <= 0x7f800000u &&
        bits != 0xff800000u && better(x,i,best,idx)) {
      best = x;
      idx = i;
    }
  }
  scores[lane] = best;
  indices[lane] = idx;
  workgroupBarrier();
  for (var stride=64u; stride>0u; stride=stride>>1u) {
    if (lane < stride) {
      let j = lane+stride;
      if (better(scores[j],indices[j],scores[lane],indices[lane])) {
        scores[lane]=scores[j];
        indices[lane]=indices[j];
      }
    }
    workgroupBarrier();
  }
  if (lane==0u) { winner[group.x]=bitcast<i32>(indices[0]); }
}
`;

export class GpuTokenSelector {
  static async create(device, count) {
    const module = device.createShaderModule({code:shader});
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(x=>x.type==="error");
    if(errors.length) throw new Error(errors.map(x=>x.message).join("; "));
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    return new GpuTokenSelector(device,pipeline,count);
  }
  constructor(device,pipeline,count) {
    this.device=device; this.pipeline=pipeline; this.count=count;
    const {STORAGE,COPY_SRC,COPY_DST,MAP_READ,UNIFORM}=GPUBufferUsage;
    this.mask=device.createBuffer({size:count*4,usage:STORAGE|COPY_DST});
    this.config=device.createBuffer({size:16,usage:UNIFORM|COPY_DST});
    this.out=device.createBuffer({size:16,usage:STORAGE|COPY_SRC});
    this.staging=device.createBuffer({size:16,usage:COPY_DST|MAP_READ});
    this.masked=null;
  }
  setMask(set) {
    if(this.masked===set)return;
    this.masked=set;
    const mask=new Uint32Array(this.count).fill(1);
    for (const id of set) if (id>=0 && id<this.count) mask[id]=0;
    this.device.queue.writeBuffer(this.mask,0,mask);
  }
  async run(buffer, offset, byteLength) { return (await this.runRows(buffer,offset,0,1,byteLength))[0]; }
  async runRows(buffer, offset, stride, rows, byteLength) {
    if(!(buffer?.usage&GPUBufferUsage.STORAGE)||offset<0||rows<1||rows>4||byteLength<(offset+(rows-1)*stride+this.count)*4)
      throw new Error("Unsupported GPU logits binding");
    const d=this.device;
    d.queue.writeBuffer(this.config,0,Uint32Array.of(this.count,offset,stride,0));
    const g=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer,size:byteLength}},
      {binding:1,resource:{buffer:this.mask}},{binding:2,resource:{buffer:this.config}},
      {binding:3,resource:{buffer:this.out}},
    ]});
    const cmd=d.createCommandEncoder();
    const pass=cmd.beginComputePass();
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,g);pass.dispatchWorkgroups(rows);pass.end();
    const readBytes=rows*4;
    cmd.copyBufferToBuffer(this.out,0,this.staging,0,readBytes);
    d.queue.submit([cmd.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ,0,readBytes);
    try { return new Int32Array(this.staging.getMappedRange(0,readBytes)).slice(); }
    finally { this.staging.unmap(); }
  }
  dispose(){ for(const b of [this.mask,this.config,this.out,this.staging]) b.destroy(); }
}
