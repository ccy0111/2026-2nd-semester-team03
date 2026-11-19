# Local opt-in fused greedy decoder experiment

This documents the original single-token-readback A/B path. For the current FP16 GPU-feedback/host-shape optimizer and its software-WebGPU validation, see [GPU feedback prototype](../webscore_inference/GPU_FEEDBACK_PROTOTYPE.md). Historical unmeasured statements below apply to this original path.

Not published, not a default, and not a measured WebGPU speedup. The legacy path
is unchanged unless `fusedGreedy: true` is explicitly selected. This experiment
requires a separately exported `decoder_greedy.onnx`; pointing it at the legacy
`decoder_kv.onnx` will not work.

```js
const client = await WebScoreClient.create(ort, {
  baseUrl: '/experimental-fused-model',
  useInt8: false,
  fusedGreedy: true,
  executionProviders: ['webgpu'],
  gpuKvCache: true,
});
```

Required files: config.json, tokenizer.json, mel_encoder.onnx,
decoder_greedy.onnx. No token_embed session or duplicate legacy decoder is loaded.
The versioned manifest wrapper is not wired to this experimental graph contract;
use the direct client above for controlled tests, not the deployed updater.

Graph contract:
- Inputs: prefix_embeds float32 `[B,P,D]`, input_ids int64 `[B,S]`, plus existing
  float32 past_k/past_v/attn_mask and int64 position_offset.
- Outputs: next_token int64 `[B]`, present_k/present_v float32.
- Internal token embedding, prefix concatenation and last-position greedy ArgMax.
  PAD/BOS/UNK must be masked before ArgMax, with smallest-index tie breaking to
  match the legacy JS decoder.
- Prefill passes encoder output + BOS/tie IDs. Later steps use an empty prefix and
  the last generated token ID. Float32 interface types must be preserved even if
  internal weights/operations use FP16.

With `gpuKvCache: true`, encoder output and decoder KV outputs request GPU storage.
The hot loop never reads their `.data`. Non-GPU KV output is an explicit error,
not a claimed residency success. Runtime-owned GPU tensors are released only
when their successor run completes, including EOS/errors. Subsequent one-token
causal masks share one zero-filled backing array; token/position storage is reused.

## Source-level accounting, not measured driver traffic

For the 14-layer, 16-query-head/4-KV-head, dim1024, card894 configuration:

- Float32 K+V bytes per context token = `2 * 14 * 4 * 64 * 4 = 28,672`.
- If a baseline run downloads 502-token KV and the next uploads 501-token KV,
  theoretical payload is `(502+501)*28,672 = 28,758,016` bytes (~28.76 MB) per
  generated token. GPU residency eliminates this *host round trip* when the
  requested locations are actually honored; GPU-side Concat/copies still exist.
- Legacy default uncached embedding path reads 4,096 embedding bytes to CPU and
  sends them back (8,192 bytes). Fused graph embeds the 8-byte token ID internally.
- Legacy last logits are `894*4 = 3,576` CPU bytes. Greedy output is one 8-byte
  int64 token (3,568 bytes less). A CPU token read/synchronization still remains.
- Encoder prefix `501*1024*4 = 2,052,096` bytes no longer needs CPU readback and
  JS prefill concatenation when encoder output is actually GPU resident.
- Session.run calls per incremental token drop from decoder+embedding (2) to
  fused decoder (1). The already-existing cached-embedding option also avoids
  embedding run calls, but still supplies CPU embeddings. **Session.run counts
  are not actual driver submission counts**, which remain unmeasured.

There is no static KV buffer, graph capture, or multi-token GPU loop here. Those
would require a different memory/graph design and cannot be inferred from this
change. CPU/mock tests validate control flow and buffer ownership only; real
WebGPU model correctness, location support, device loss, memory stability and
latency must be checked separately before rollout.
