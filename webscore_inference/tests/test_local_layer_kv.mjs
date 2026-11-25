import {test} from 'node:test';
import assert from 'node:assert/strict';
import {bridge,perLayerOutputs,initLocalLayerDecoder,localLayerDecoderUrl,localProjectionDecoderUrl} from '../web/local-layer-kv.mjs';
const ort={Tensor:class{constructor(type,data,dims){Object.assign(this,{type,data,dims})}}};
const ins=['inputs_embeds','position_offset','attn_mask',...perLayerOutputs.map(x=>x.replace('present_','past_'))],outs=['logits',...perLayerOutputs];
function physical() {
 const calls=[],disposed=[];
 let bad=false, releases=0;
 const session={inputNames:ins,outputNames:outs,inputMetadata:ins.map(name=>({name})),async release(){releases++;},
  async run(f){calls.push(f);const [B,S]=f.inputs_embeds.dims,T=f.past_k_00.dims[2]+S;
    return {logits:{location:'gpu-buffer',dispose(){disposed.push('logits');}},...Object.fromEntries(perLayerOutputs.map(name=>
      [name,{type:'float16',id:name,location:bad&&name.endsWith('13')?'cpu':'gpu-buffer',dims:[B,4,T,64],dispose(){disposed.push(name);}}]))};
  }};
 return {session,calls,disposed,markBad(){bad=true},get releases(){return releases}};
}
const input=(b,s,k={type:'float32',dims:[14,b,4,0,64]},v=k)=>({inputs_embeds:{dims:[b,s,1024]},past_k:k,past_v:v,attn_mask:'mask',position_offset:'pos'});
test('per-layer wrapper at B4/B3/B1 preserves public logical interface and direct GPU history',async()=>{
 const r=physical(),b=bridge(ort,r.session);
 assert.deepEqual(b.inputNames,['inputs_embeds','past_k','past_v','position_offset','attn_mask']);assert.deepEqual(b.outputNames,['logits','present_k','present_v']);
 const min=await b.run(input(4,502)),y=await b.run(input(4,1,min.present_k,min.present_v));
 assert.deepEqual(r.calls[0].past_v_13.dims,[4,4,0,64]);assert.equal(r.calls[1].past_k_13.id,'present_k_13');
 assert.deepEqual(y.present_v.dims,[14,4,4,503,64]);
 for (const width of [3,1]) {
   const ans=await b.run(input(width,502));
   assert.deepEqual(ans.present_v.dims,[14,width,4,502,64]);ans.present_k.dispose();ans.present_v.dispose();
 }
 min.present_k.dispose();min.present_v.dispose();
 await assert.rejects(()=>b.run(input(4,1,min.present_k,min.present_v)),/released/);
 y.present_k.dispose();y.present_v.dispose();
 assert.equal(r.disposed.length,112);
});
test('malformed local session rejected and released once before packed fallback',async()=>{
 const r=physical();r.session.inputNames=[...ins,'bad-required'];
 let loads=0;
 await assert.rejects(()=>initLocalLayerDecoder({InferenceSession:{async create(url,opts){
   loads++;assert.equal(url,localLayerDecoderUrl);assert.equal(opts.preferredOutputLocation.present_k_00,'gpu-buffer');return r.session;
 }}},{executionProviders:['webgpu']}),/Unexpected per-layer/);
 assert.equal(loads,1);assert.equal(r.releases,1);
});
test('reject CPU output and release collected GPU output leases',async()=>{
 const r=physical(),b=bridge(ort,r.session);r.markBad();
 await assert.rejects(()=>b.run(input(4,502)),/came off GPU/);assert.equal(r.disposed.length,27);
});

test('fused projection is opt-in; failed fusion returns current per-layer session before packed fallback is needed', async () => {
 const good=physical(), rejected=physical();
 rejected.session.inputNames=[];
 const calls=[];
 const rt={InferenceSession:{async create(url){
  calls.push(url);
  if (url.includes("fused-range-")) throw new Error("ramp trial missing");
  return url===localProjectionDecoderUrl ? rejected.session : good.session;
 }}};
 const selected=await initLocalLayerDecoder(rt,{executionProviders:['webgpu']},true);
 assert.match(calls.shift(),/fused-range-ff3e6b78/);
 assert.deepEqual(calls,[localProjectionDecoderUrl,localLayerDecoderUrl]);
 assert.equal(selected.projectionFusion,false);
 assert.equal(rejected.releases,1);
 await selected.release();assert.equal(good.releases,1);
});
