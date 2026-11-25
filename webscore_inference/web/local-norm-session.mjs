import {kernelForMarker, validateAttentionScaleOnDevice} from "./local-norm-contract.mjs";

export const localNormDecoderUrl = "/inference/decoder-grpo100-rdna3-norm-eaeb37f4.onnx";

function checkContext(ort, opts, device) {
 const adapter=ort.env.webgpu.adapter;
 return ort.env.versions?.web==="1.31.1" && adapter?.info?.architecture==="rdna-3" &&
  !adapter.info.isFallbackAdapter && device?.features.has("shader-f16") && device.features.has("subgroups") &&
  opts.extra?.ep?.webgpuexecutionprovider?.maxNumPendingDispatches==="64" &&
  ["127.0.0.1","localhost","[::1]"].includes(globalThis.location?.hostname);
}

function installShaderScope(device, onInterception){
 const prev=Object.getOwnPropertyDescriptor(device,"createShaderModule");
 const create=device.createShaderModule;
 if(typeof create!=="function")throw Error("No WebGPU compiler");
 const compile=function(descriptor){
  if(descriptor.label!=="Pow")return create.call(device,descriptor);
  const code=kernelForMarker(descriptor.code);
  if(!code)throw Error("Unrecognized FP16 norm marker from the pinned runtime");
  onInterception();
  return create.call(device,{...descriptor,label:"ExactF16RMS",code});
 };
 Object.defineProperty(device,"createShaderModule",{configurable:true,value:compile});
 return ()=>{
  if(prev)Object.defineProperty(device,"createShaderModule",prev);
  else delete device.createShaderModule;
 };
}

export function scopeLocalDecoder(session, device, loadPlain) {
 let inspected=false, switching=false, plain, closed=false, ran=false;
 return {
  inputNames:session.inputNames, inputMetadata:session.inputMetadata,outputNames:session.outputNames,
  get kernelActive(){return inspected&&!plain;},
  async run(feeds){
   if(closed||switching)throw Error("Concurrent or released local decoder request");
   switching=true;
   try{
    if(plain)return await plain.run(feeds);
    let intercepted=false, restore, scopeOpen=false, scopeFailure, answer;
    try{
     // This decoder runs in the dedicated worker, not on the page-wide GPU.
     // Scope the override to this session invocation; other sessions do not see it.
     restore=installShaderScope(device,()=>{intercepted=true;});
     try{
      if(!inspected){device.pushErrorScope("validation");scopeOpen=true;}
      answer=await session.run(feeds);
     }
     finally{
      restore();
      if(scopeOpen)scopeFailure=await device.popErrorScope();
     }
     if(scopeFailure||(!inspected&&!intercepted))throw Error("FP16 shader validation failed on this browser");
     inspected=true;ran=true;
     return answer;
    }catch(e){
     for(const x of Object.values(answer??{})) x?.dispose?.();
     console.warn("[WebScore] Local FP16 kernel disabled; retaining native FP16 model.");
     plain=await loadPlain();
     // If a later shader variant fails, existing KV leases may belong to the
     // candidate. Keep that session alive until the parent releases both.
     if(!ran)await session.release();
     return await plain.run(feeds);
    }
   }finally{switching=false;}
  },
  async release(){
   if(closed)return;
   if(switching)throw Error("Cannot dispose during local decode");
   closed=true;
   if(!plain||ran)await session.release();
   if(plain)await plain.release();
  },
 };
}

export async function initNormFusedNative(ort,opts,plainUrl){
 const device=await ort.env.webgpu.device;
 if(!checkContext(ort,opts,device))throw Error("Outside verified local WebGPU configuration");
 await validateAttentionScaleOnDevice(ort,opts);
 const session=await ort.InferenceSession.create(localNormDecoderUrl,opts);
 return scopeLocalDecoder(session,device,()=>ort.InferenceSession.create(plainUrl,opts));
}
