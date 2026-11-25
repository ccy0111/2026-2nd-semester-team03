// Adapts only the pinned GRPO100 dynamic independent-batch per-layer half cache ABI.
// Keeps real GPU handles; no tensor contents are read or copied at this layer.
const kinds=["k","v"],indexes=Array.from({length:14},(_,i)=>String(i).padStart(2,"0"));
export const perLayerOutputs=kinds.flatMap(k=>indexes.map(i=>"present_"+k+"_"+i));

function lease(leases,layer,shape){
 const order={[Symbol.toStringTag]:"PerLayerGPUCache",type:"float16",location:"gpu-buffer",dims:shape};
 let done=false;
 order.dispose=()=>{if(!done){done=true;leases.delete(order);for(const x of layer)x.dispose();}};
 leases.set(order,layer);
 return order;
}
const logicalInputs=["inputs_embeds","past_k","past_v","position_offset","attn_mask"];
const logicalOutputs=["logits","present_k","present_v"];
const physicalInputs=[...logicalInputs.filter(n=>n!=="past_k"&&n!=="past_v"),...perLayerOutputs.map(n=>n.replace("present_","past_"))];
const physicalOutputs=["logits",...perLayerOutputs];
const sameNames=(actual,expected)=>actual?.length===expected.length&&expected.every(n=>actual.includes(n));
export const localLayerDecoderUrl="/api/local-preview/v2-grpo100-layerkv/decoder.onnx";
const rangeName="/Range_output_0_cast_to_fp16", ramp=Float32Array.from({length:2048},(_,i)=>i);
const positions=new WeakMap();
function rangeFor(ort,span){
 if(!Number.isSafeInteger(span)||span<1||span>ramp.length)throw Error("Outside exactly representable FP16 Range domain");
 let cache=positions.get(ort);if(!cache){cache=new Map();positions.set(ort,cache);}
 if(cache.has(span))return cache.get(span);
 const t=new ort.Tensor("float32",ramp.subarray(0,span),[span]);
 if(cache.size>=32)cache.delete(cache.keys().next().value);
 cache.set(span,t);return t;
}
export function bridge(ort,session,externalRange=false){
 const empties=new Map(),leases=new WeakMap();
 const wrap=(ts,dims)=>lease(leases,ts,dims);
 const names=new Set(session.inputMetadata.map(x=>x.name));
 const expectedInputs=externalRange?[...physicalInputs,rangeName]:physicalInputs;
 if(!sameNames(session.inputNames,expectedInputs)||!sameNames(session.outputNames,physicalOutputs)||
    expectedInputs.some(n=>!names.has(n)))throw Error("Unexpected per-layer decoder interface");
 const before=session.run.bind(session);
 return {inputNames:logicalInputs,outputNames:logicalOutputs,inputMetadata:[...session.inputMetadata,{name:"past_k",type:"float32",shape:[14,"batch",4,"past_seq_len",64]},{name:"past_v",type:"float32",shape:[14,"batch",4,"past_seq_len",64]}],release:()=>session.release(),
  async run(feeds){
   const {past_k,past_v,...next}=feeds;
   const B=next.inputs_embeds.dims[0],S=next.inputs_embeds.dims[1];
   const T=past_k?.dims[3];
   if(![1,2,3,4].includes(B)||!Number.isSafeInteger(S)||S<1||!Number.isSafeInteger(T)||T<0||past_v?.dims[3]!==T)throw Error("Invalid batch cache adapter request");
   for(const [k,old] of [["k",past_k],["v",past_v]]){
    if(old?.type!==(T>0?"float16":"float32")||String(old?.dims)!==String([14,B,4,T,64]))throw Error("Unexpected old KV metadata");
    if(T===0&&!empties.has(B))empties.set(B,new ort.Tensor("float16",new Uint16Array(0),[B,4,0,64]));
    const v=leases.get(old);
    if(T&&!v)throw Error("Invalid or released GPU layer state");
    for(const [z,i] of indexes.entries())next["past_"+k+"_"+i]=v?v[z]:empties.get(B);
   }
   if(externalRange)next[rangeName]=rangeFor(ort,S);
   const out=await before(next);
   const expected=[B,4,T+S,64];
   const valid=t=>t?.location==="gpu-buffer"&&t.type==="float16"&&String(t.dims)===String(expected);
   try{
    const k=indexes.map(i=>out["present_k_"+i]),v=indexes.map(i=>out["present_v_"+i]);
    if(!k.every(valid)||!v.every(valid)||out.logits?.location!=="gpu-buffer")throw Error("Per-layer state or logits came off GPU");
    return {present_k:wrap(k,[14,...expected]),present_v:wrap(v,[14,...expected]),logits:out.logits};
   }catch(e){
    for(const t of Object.values(out))if(t?.location==="gpu-buffer")t.dispose();
    throw e;
   }
  }
 };
}


// Only the caller with the exact released encoder/embedding/tokenizer pins may
// elect this same-origin preview asset. Fall back to the packed GPU model
// without retaining both GPU sessions; outer worker may still choose WASM INT8.
export const localProjectionDecoderUrl = "/inference/decoder-grpo100-fused-673e022b.onnx";
export async function initLocalLayerDecoder(ort, regularOptions, preferProjectionFusion = false) {
 const opts={...regularOptions,preferredOutputLocation:Object.fromEntries(physicalOutputs.map(n=>[n,"gpu-buffer"]))};
 async function load(url, projectionFusion, externalRange = false) {
  const session=await ort.InferenceSession.create(url, opts);
  try { return Object.assign(bridge(ort,session,externalRange), { projectionFusion, externalRange }); }
  catch(error) { await session.release(); throw error; }
 }
 if (preferProjectionFusion === true) {
  if (opts.extra?.ep?.webgpuexecutionprovider?.maxNumPendingDispatches === "64") {
   try {
    const {initNativeWebGpuDecoder} = await import("./webgpu-native-decoder.mjs");
    return await initNativeWebGpuDecoder(ort, opts);
   } catch (err) {
    console.warn("[WebScore] Native WebGPU fused decoder unavailable; falling back to ORT FP16.", err?.message);
   }
   const plainUrl = "/inference/decoder-grpo100-fused-range-ff3e6b78.onnx";
   let session;
   try {
    const {initNormFusedNative} = await import("./local-norm-session.mjs");
    session = await initNormFusedNative(ort, opts, plainUrl);
    const adapted = Object.assign(bridge(ort, session, true), { projectionFusion: true, externalRange: true });
    Object.defineProperty(adapted, "nativeNormActive", {get: () => session.kernelActive});
    return adapted;
   } catch (err) {
    await session?.release().catch(() => {});
    console.warn("[WebScore] Specialized FP16 kernel unavailable; using the regular FP16 graph.", err?.message);
   }
  }
  // The default preview remains the proven fused decoder if this optional
  // pinned asset is absent or initialization fails. Precision is unchanged.
  try { return await load("/inference/decoder-grpo100-fused-range-ff3e6b78.onnx", true, true); }
  catch { /* full-shape native fused fallback */ }
  try { return await load(localProjectionDecoderUrl, true); }
  catch { /* keep the already validated local per-layer GPU decoder */ }
 }
 return await load(localLayerDecoderUrl, false);
}
