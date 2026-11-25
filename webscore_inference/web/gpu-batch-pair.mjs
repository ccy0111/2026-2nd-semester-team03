// Batch two to four independent (pitch-tie) five-second acoustic chunks.
import {GpuBlockFeedback, consumeGpuBlock} from "./gpu-block-feedback.mjs";
// The immutable GRPO100 prefix has 501 acoustic frames plus BOS. Bound one
// packed FP32 K or V binding, not total VRAM; failures still retry at lower width.
export function boundedGpuBatchWidth(requested, remaining, maxGenLen, bindingBytes) {
 if(!(bindingBytes>0)||!Number.isSafeInteger(maxGenLen)||maxGenLen<32||maxGenLen>1024)return 0;
 const bytesPerRow=14*4*64*4*(502+maxGenLen);
 const width=Math.min(requested,remaining,Math.floor(bindingBytes/bytesPerRow));
 return width>=2?width:0;
}


export async function runBatchPair(c, pcms, options={}) {
 // Experimental, deliberately disabled unless a local bench opts in.
 const block=c.gpuFeedbackBlock;
 if ([1,2,4,8].includes(block) && !c.gpuFeedbackDisabled && options.selector) {
  const stats=c.gpuFeedbackStats??={groups:0,failed:0,decoderCalls:0,decoderAwaitMs:0,feedbackSubmits:0,feedbackSubmitMs:0,blockReadbacks:0,blockReadbackMs:0,terminalEvaluatedRows:0};
  try {
   const result=await runBatch(c,pcms,options,{block,stats});
   stats.groups++;
   return result;
  } catch(error) {
   options.signal?.throwIfAborted();
   if (c.gpuBatchPairs === false || pcms.length > (c.decoderSession?.nativeMaxBatch ?? 4)) throw error;
   stats.failed++;
   stats.failureName=error?.name??"Error";
   stats.failureReason=String(error?.message??"GPU feedback failed").replace(/https?:[^ ]+/g,"[URL]").slice(0,250);
   c.gpuFeedbackDisabled=true;
   // Retry the same independent group through the tested baseline, not a partial stitch.
  }
 }
 return runBatch(c,pcms,options,null);
}
async function runBatch(c, pcms, {allowedPrograms, maxGenLen=1024,selector=null,signal=null}={}, experiment) {
 const B=pcms.length;
 signal?.throwIfAborted();
 const maxB = c.decoderSession?.nativeMaxBatch ?? 4;
 if(B<2||B>maxB)throw new Error("Invalid independent chunk batch size");
 const m=pcms.map(p=>c.mel.computeChunkLogMel(p));
 const mel=new Float32Array(m[0].data.length*B);
 for(let row=0;row<B;row++)mel.set(m[row].data,row*m[0].data.length);
 const tensor=(type,data,dims)=>new c.ort.Tensor(type,data,dims);
 const out=await c.encoderSession.run({mel:tensor("float32",mel,[B,...m[0].dims.slice(1)])});
 const cond=out.cond_embeds;
 if(cond.dims[0]!==B)throw new Error("Encoder did not preserve batch dimension");
 const C=cond.dims[1], D=cond.dims[2], V=c.lookup.idToEvent.length, P=C+1;
 const bos=await c.embedTokenIds([c.lookup.bosId]);
 const prep=new Float32Array(B*P*D);
 for(let row=0;row<B;row++){
  prep.set(cond.data.subarray(row*C*D,(row+1)*C*D),row*P*D);
  prep.set(bos.data,row*P*D+C*D);
 }
 let embeds=tensor("float32",prep,[B,P,D]);
 const H=c.arch.dim/c.arch.num_heads,L=c.arch.num_layers,K=c.arch.num_kv_heads;
 const zero=()=>tensor("float32",new Float32Array(),[L,B,K,0,H]);
 let pk=zero(),pv=zero(),pos=0;
 const posData=new BigInt64Array(1);
 const posTensor=tensor("int64",posData,[1]);
 const tok=Array.from({length:B},()=>[]),done=new Array(B).fill(false);
 const allowed=allowedPrograms===undefined?null:new Set(allowedPrograms),masked=new Set(c.maskedIds);
 if(allowed)for(const {id,kind,value} of c.lookup.idToEvent){
   if((kind==="inst"&&(value>=128||!allowed.has(value)))||(kind==="drum"&&!allowed.has(128)))masked.add(id);
 }
 if (typeof c.decoderSession?.generateBlockNative === "function" && c.embeddingTable) {
  try {
   const nativeRes = await c.decoderSession.generateBlockNative({
    prefillData: prep,
    batchSize: B,
    prefillLen: P,
    maxGenLen,
    maskedIds: masked,
    eosId: c.lookup.eosId,
    vocabSize: V,
    embeddingTable: c.embeddingTable,
    blockSize: 64,
    signal,
    stats: experiment?.stats ?? null,
   });
   return { tokens: nativeRes.tokens, eos: nativeRes.done, decoderCalls: nativeRes.decoderCalls };
  } catch (error) {
   signal?.throwIfAborted();
   if (typeof c._fallbackFromNativeDecoder === "function" && await c._fallbackFromNativeDecoder(signal)) {
    const safeWidth = c.gpuKvBindingBytes > 0
     ? boundedGpuBatchWidth(c.gpuBatchWidth ?? 4, B, maxGenLen, c.gpuKvBindingBytes)
     : 4;
    if (c.gpuBatchPairs === false || B > safeWidth) throw error;
   } else {
    throw error;
   }
  }
 }
 const offsets=[];
 if(selector)selector.setMask(masked);
 const free=t=>{if(t?.location==="gpu-buffer")t.dispose()};
 const buffer=experiment?await GpuBlockFeedback.create(c,B,D,masked,experiment.block):null;
 const stats=experiment?.stats;
 let slot=0;
 try{
  for(let step=0;step<maxGenLen;step++){
    signal?.throwIfAborted();
    const n=embeds.dims[1], total=pk.dims[3]+n;
    const mask=new Float32Array(B*n*total);
    for(let b=0;b<B;b++)for(let t=0;t<n;t++) {
      const start=(b*n+t)*total;
      mask.fill(-1e9,start+pk.dims[3]+t+1,start+total);
    }
    posData[0]=BigInt(pos);offsets.push(pos);
    const start=stats?performance.now():0;
    const dec=await c.decoderSession.run({inputs_embeds:embeds,past_k:pk,past_v:pv,position_offset:posTensor,attn_mask:tensor("float32",mask,[B,1,n,total])});
    if(stats){stats.decoderCalls++;stats.decoderAwaitMs+=performance.now()-start;}
    free(pk);free(pv);pk=dec.present_k;pv=dec.present_v;pos+=n;
    if(pk.dims[1]!==B||pv.dims[1]!==B)throw new Error("Decoder cache did not preserve batch dimension");
    const [b,rows,v]=dec.logits.dims;
    if(b!==B||v!==V||(rows!==1&&rows!==n))throw new Error("Decoder logits did not preserve batch dimension");
    const logits=dec.logits;
    if(buffer){
     try {
      signal?.throwIfAborted();
      const t=performance.now();buffer.emit(logits,rows,slot++);stats.feedbackSubmits++;stats.feedbackSubmitMs+=performance.now()-t;
     } finally { free(logits); }
     if(slot===experiment.block || step+1===maxGenLen){
      const t=performance.now(),ids=await buffer.read(slot);
      stats.blockReadbacks++;stats.blockReadbackMs+=performance.now()-t;
      signal?.throwIfAborted();
      const accepted=tok.reduce((a,x)=>a+x.length,0);
      consumeGpuBlock({tokens:tok,done},ids,c.lookup.eosId,V,masked);
      stats.terminalEvaluatedRows+=ids.length-tok.reduce((a,x)=>a+x.length,0)+accepted;
      slot=0;
      if(done.every(Boolean)||step+1===maxGenLen)break;
     }
     embeds=buffer.input;
     continue;
    }
    let vals,selected;
    try{
      if(selector && logits.location==="gpu-buffer" && logits.type==="float32"){
        selected=await selector.runRows(logits.gpuBuffer,(rows-1)*V,rows*V,B,B*rows*V*4);
      }else{vals=logits.location==="gpu-buffer"?await logits.getData(true):logits.data}
      signal?.throwIfAborted();
    }finally{if(logits.location==="gpu-buffer")logits.dispose()}
    const stepIds=new Array(B).fill(c.lookup.eosId);
    for(let row=0;row<B;row++){
      if(done[row])continue;
      let best=-Infinity,id=-1;
      const offset=((row+1)*rows-1)*V;
      if(selected)id=selected[row];
      else for(let j=0;j<V;j++)if(!masked.has(j)&&vals[offset+j]>best){best=vals[offset+j];id=j}
      if(allowed&&id<0)throw new Error("No valid token available for allowedPrograms");
      tok[row].push(id);stepIds[row]=id;done[row]=id===c.lookup.eosId;
    }
    if(done.every(Boolean)||step+1===maxGenLen)break;
    if(stepIds.some(x=>x<0))throw new Error("Cannot batch invalid embedding id");
    const e=await c.embedTokenIds(stepIds);
    embeds=tensor("float32",e.data,[B,1,D]);
  }
  return {tokens:tok,eos:done,decoderCalls:offsets.length};
 }finally{free(pk);free(pv);buffer?.dispose()}
}
