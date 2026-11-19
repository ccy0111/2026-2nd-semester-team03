# Opt-in native FP16 WebGPU optimizer (local-only prototype)

This PR targets the step-53,500 model with **FP16 internal weights/operations**, using the native ONNX Runtime WebGPU entrypoint. It does not add a variant loader or change production INT8 inference, updater manifests, UI, backend, CDN, model weights or defaults. The supported prototype is selected directly through WebScoreClient. Float32 interfaces remain the default. The separately selected feedbackHalfKv graph keeps only K/V state in FP16; other interfaces remain unchanged.

## Reproduce the FP16 target

Use matching config.json, tokenizer.json, mel_encoder.onnx, token_embed.onnx and decoder_kv.onnx from the verified step-53,500 FP16 export (keep_io_types=True). Do not mix checkpoints or replace these inputs with INT8 exports. Graph generation needs Python, numpy and onnx (tested numpy/onnx in the export environment; no PyTorch/training execution is required).

```sh
python tools/build_feedback_graph.py --source-dir /models/53500-fp16 --output /models/local-optimizer/decoder_feedback.onnx
python tools/build_feedback_hostshapes.py --input /models/local-optimizer/decoder_feedback.onnx --output /models/local-optimizer/decoder_feedback_hostshapes.onnx
# Copy the same export's config.json, tokenizer.json and mel_encoder.onnx into local-optimizer.
```

Use the native `onnxruntime-web` 1.31.1 `ort.webgpu.min.mjs` entrypoint, with a shader-f16-capable adapter. This recipe does not upload or publish files. No weights or generated ONNX files belong in Git. The shape rewrite targets this known WebScore export structure, not arbitrary models.

```js
const client = await WebScoreClient.create(ort, {
  baseUrl: '/local-optimizer', useInt8: false, fusedGreedy: true,
  chunkedFeedback: true, feedbackHostShapes: true, feedbackReadInterval: 8,
  gpuKvCache: true, executionProviders: ['webgpu'],
});
```

With feedbackHostShapes:true, decoder_feedback_hostshapes.onnx is loaded. With it false, decoder_feedback.onnx is loaded for A/B. Neither option changes ordinary INT8 callers or the versioned wrapper.

## Original feedback-only A/B path

```js
await WebScoreClient.create(ort, {
  baseUrl: '/local-feedback-model', useInt8: false, fusedGreedy: true,
  chunkedFeedback: true, feedbackReadInterval: 8,
  gpuKvCache: true, executionProviders: ['webgpu'],
});
```

Directory contains config.json, tokenizer.json, mel_encoder.onnx and the **new** decoder_feedback.onnx. Existing decoder files are not compatible with this option. Standard loaders and automatic model version selection remain unchanged.

Generate the graph from matching standard FP16 (or FP32) encoder/embedding/decoder exports:

```sh
python tools/build_feedback_graph.py --source-dir /path/to/standard-export --output /path/to/prototype/decoder_feedback.onnx
```

The generator keeps floating-point I/O and existing arithmetic casts, combines token embedding with the decoder, projects only the final hidden row, masks PAD/BOS/UNK, and chooses the first ArgMax index. Input token_history is INT64 [1,N]; outputs next_ids INT64 [1,1], updated_history INT64 [1,N+1], and FP32 present_k/present_v. Remaining inputs match fused greedy: prefix_embeds, input_ids, past_k/v, position_offset, attn_mask.

The client reuses next_ids and KV tensors directly on GPU. Audio prefix is used only at prefill, followed by an empty prefix. Position/mask are generated using known shapes and uploaded without a GPU read. An eight-token history is downloaded only at a boundary or generation cap. ORT getData changes the tensor to CPU, so it is called with releaseData=true and history is reset to empty rather than uploaded again. The next_ids tensor is separate and continues GPU feedback. The first EOS ends the returned sequence; up to seven extra tokens may have been computed and are discarded. No quality decision is based on the discarded tail. A hard generation cap is still enforced.

There is still one awaited ORT session.run per token, multiple operators/kernels, input uploads, and periodic readback. This does not claim one driver submission, graph capture, no synchronization, or hardware acceleration. Dynamic KV concatenation still exists. FP16 requires a compatible runtime/device; there is no silent automatic fallback for this direct experimental option.

Verification: real step-53500 FP16 graph on CPU ORT (piano, shifted pitch/tempo, silence, tie prologue), real JS client with onnxruntime-web Node WASM (six-note piano), and separate GPU-tensor mocks testing read frequency, direct feedback identity, EOS truncation, partial final reads, error cleanup and token cap. Real model token parity is distinct from mock residency tests. Actual hardware WebGPU speed and GPU kernel coverage are not established by these tests.

## Native int64 WebGPU session option
Only when both chunkedFeedback and gpuKvCache are explicitly enabled, the feedback decoder session receives `extra: { 'ep.webgpuexecutionprovider.enableInt64': '1' }`. This preserves native GPU token IDs/history instead of int64 emulation requiring CPU placement. Other decoders, the encoder, and WASM/default paths receive no extra flag. CPU fallback is NOT disabled by default.

The native `ort.webgpu.min.mjs` entrypoint in onnxruntime-web 1.31.1 was separately tested with Node Dawn/SwiftShader: the full FP32 feedback graph placed all 1,191 nodes on WebGpuExecutionProvider with the option enabled and CPU fallback explicitly disabled in the test harness. This is software WebGPU, not a physical GPU or browser performance result. It does not verify FP16 hardware execution. Runtime entrypoint/version and operator support must be validated on the actual deployment target; this remains an explicitly selected prototype.

## Separate host-shape beta
`feedbackHostShapes: true` additionally requires chunkedFeedback and selects `decoder_feedback_hostshapes.onnx`. The original feedback artifact/path stays available for A/B. Generate it using `python tools/build_feedback_hostshapes.py --input decoder_feedback.onnx --output decoder_feedback_hostshapes.onnx`.

For batch 1, S = prefix length + token ID count and L = previous KV length + S. Shared CPU INT64 operands replace each layer's shape-only graph:
- shape_q = [1,S,num_heads,head_dim]
- shape_kv = [1,S,num_kv_heads,head_dim]
- shape_expand = [1,num_kv_heads,num_heads/num_kv_heads,L,head_dim]
- shape_expanded_kv = [1,num_heads,L,head_dim]
- shape_out = [1,S,dim]
- position_range = FLOAT32 arange(0,S); position offset is still added inside the model.

The generator prunes now-unreachable shape calculations and replaces Range with host-generated values, preserving downstream FP16 casts. Model weights, attention/KV arithmetic, precision, tokens and masking are unchanged. This targets internal shape readbacks observed in ORT, not just explicit JS reads. CPU FP16 tests cover all four fixtures with token parity; unit tests cover prefill, incremental shape values and history resets. Actual software-WebGPU mapAsync results must be reported separately from physical GPU results.

Measured independently in the strict FP32 Node Dawn/SwiftShader two-step probe: all 543 host-shape graph nodes placed on WebGPU; tokens 263 then 330 matched the original graph. Internal mapAsync calls before deliberate output checks fell from 98/98 per step to 0/0; queue.submit calls fell from 147/147 to 29/32. Eight deliberate final output-check readbacks are excluded from those internal counts. These narrow software-WebGPU measurements establish removal of the observed shape-readback boundaries, not browser/physical-GPU speed, full-audio GPU parity, or FP16 hardware support.

Later, the same native ORT Web version on Dawn with Mesa Lavapipe `shader-f16` reproduced the two-step readback result on the actual FP16 decoder (strict CPU fallback disabled, 1,202→553 WebGPU nodes). The five-second piano fixture through the actual optimized FP16 JS client matched 25 reference tokens and six notes. It ran 32 decoder steps, with four requested token-history readbacks and zero observed implicit decoder buffer maps. This is a sandbox software-adapter result, not measured browser speed or hardware support.

## Additional FP16 KV state option

After the two build steps above, run:

```sh
python tools/keep_fp16_kv.py /models/local-optimizer/decoder_feedback_hostshapes.onnx /models/local-optimizer/decoder_feedback_hostshapes_f16kv.onnx
```

Set `feedbackHalfKv: true` in addition to `feedbackHostShapes: true` to select the new artifact. This is only for matching FP16 exports. The generator rejects anything other than the expected FP32 boundary wrapping casts, removes the FP16→FP32→FP16 state round trip, and runs full ONNX type checking. The JS client initializes empty float16 state using native Float16Array when available, otherwise Uint16Array storage, and checks returned K/V tensor types.

For this model, two FP16 K/V tensors use 14,336 bytes per context token versus 28,672 with FP32 state. This halves the stored KV representation only. It does not guarantee halved peak memory or faster physical-device execution; attention temporaries, dynamic concatenation and weights remain. Verification found four CPU fixture token sequences unchanged and strict two-step software-WebGPU token parity with zero internal maps. Full-audio results should be reported independently.

### Isolated branch verification
On the same five-second piano fixture with software WebGPU (Dawn/Lavapipe, ORT Web 1.31.1, strict no CPU fallback), the FP16 graph before host shapes recorded 3,136 implicit buffer maps over 32 decoder runs. With host shapes this fell to zero; both runs performed four deliberate token-history reads and produced the same 25 tokens, six notes and exact MIDI bytes. These are WebGPU API observations, not hardware performance measurements.

The additional FP16 KV option also passed the full fixture through the isolated branch's real client with 25 exact tokens, byte-identical MIDI, GPU-resident decoder outputs, zero observed implicit decoder maps and four deliberate history readbacks. CPU token parity additionally covered piano, pitch/tempo shift, silence and a tie prologue. The KV buffer payload is half the FP32 variant by construction; this does not imply a 50% reduction in overall memory or improved hardware latency.
