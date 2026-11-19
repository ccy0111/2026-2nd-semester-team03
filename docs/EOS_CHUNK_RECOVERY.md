# Bounded recovery for a chunk that never emits EOS

This change is independent of the FP16 optimizer and teammate UI work. It does not change weights, model URLs, precision, defaults for successful decoding, CDN files, frontend code, or deployment.

`transcribeChunk` throws the exported `TokenLimitError` (`code: TOKEN_LIMIT_EXHAUSTED`) if the model fails to emit EOS within `maxGenLen`. Partial generated tokens are not returned. The final unused token is not sent through another embedding call after the generation budget is exhausted.

`transcribePcm`, and therefore `transcribeAudio` and the existing versioned-client forwarding methods, apply one shared policy:

1. Decode normally, preserving the existing carry-note prompt.
2. Only on `TokenLimitError`, discard that attempt. Close prior open notes at this chunk's start and clear carry state. Retry exactly once with bare `tie`, the same generation cap, and freshly initialized KV. Loaded sessions may be reused; autoregressive state is not reused.
3. If retry reaches EOS, use only its successful tokens and report `recovered`.
4. If retry also exhausts the cap, insert no notes or tokens for that interval, report `failed`, and continue the next chunk with empty carry state. Earlier/later successfully decoded notes remain available.

No forced EOS, unchecked partial MIDI, indefinite retry, increased token limit, or automatic re-download is used. Retry can shorten sustained notes across a boundary; a failed interval is a gap. Recovery and completion do not guarantee transcription accuracy.

## Result and progress contract

Existing `notes`, `midiBytes` and `tokensByChunk` remain. `tokensByChunk` stays aligned with audio chunks; a failed chunk has `[]`. Both note data and MIDI are built from the same accepted chunks. Additional `chunkStatuses` uses zero-based `index` and contains:

```js
{ index: 3, startSec: 15, endSec: 20,
  status: 'ok' /* or 'recovered' or 'failed' */,
  attempts: 1 /* or 2 */, boundaryReset: false /* or true */ }
```

Recovery/failed statuses additionally have `reason: 'token_limit'`. `endSec` is clipped to actual audio duration, not padded input length. The normal progress callback adds a `chunkStatus` snapshot alongside existing `completedChunks`/`totalChunks`. Completion means all chunks were processed; callers must inspect statuses before representing the result as fully successful. This PR does not alter any frontend display.

## Cancellation and unrelated failures

Optional `signal: AbortSignal` is accepted by transcription methods. Pre-aborted requests fail before inference. Cancellation is checked between decoder runs and before retry/decode/progress; an active ORT run cannot be synchronously interrupted. Once it resolves, cancellation propagates rather than being retried or converted to a successful/failed chunk. Existing GPU KV cleanup runs in `finally`, and the transcription busy flag is reset.

Network, model/runtime/device, invalid-input, callback, and metadata errors propagate. Matching an error message or arbitrary error code is deliberately insufficient to trigger recovery. `maxGenLen` must be a positive integer.

## Evidence and limits

The private actual-song sandbox investigation used step-53,500 INT8 and the baseline JS/WASM client with ORT Web 1.20.1. A 150–155 s chunk exhausted both 1,024 and 2,048 tokens, repeating a pitch without EOS. Bare-tie retry completed in 94 tokens, and continuation reached all 35 chunks; warm-session and fresh-session continuation produced identical MIDI hashes. ORT 1.31.1 showed a similar failure at 135–140 s and successful 106-token bare-tie recovery.

These are observations for one song, not an assertion of universal recovery. They do not reproduce the exact reported progress indicators: 49% with the older model and 97% with the newer model. The test uses fixed audio intervals, and FFmpeg 16 kHz PCM may differ from browser AudioContext resampling.

The real song, model weights, and private inference traces are not included in this PR. Tests cover normal carry, recovery, double failure at beginning/middle/end, padding duration, error filtering, cancellation, progress errors, and the actual token-cap loop. No generated models or frontend changes are needed for this policy.
