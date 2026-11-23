# Bounded recovery for a chunk that never emits EOS

This change does not change weights, model URLs, precision, default decoding, CDN files, or deployment. The existing browser worker and service forward the optional boundary mode and result statuses without changing the UI.

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
# Boundary modes

`transcribePcm(pcm, { boundaryMode })` and `transcribeAudio(file, { boundaryMode })`
take one optional mode. Calls without a mode use Pitch Tie:

- `legacy` (explicit compatibility mode): BOS only in the first chunk, forced held-note prelude in
  later chunks, with exact program and pitch matching.
- `prelude-forcing`: same exact matching, but force a bare tie delimiter on the
  first chunk as well. Later chunks receive the sorted held notes and delimiter.
- Omitted or `pitch-tie`: BOS only on every normal attempt. Match the model's leading
  declarations to held notes by exact program and pitch first, then by pitch
  only when one unmatched note and one unmatched declaration remain. The
  original note retains its program; a changed program is only a chunk-local
  alias for note-off or retrigger. Ambiguous and missing ties close at the
  boundary; orphan declarations and note-offs do not generate an onset.

Pitch Tie and explicit prelude forcing carry the program from a valid leading prelude into the
body, but begin with no velocity and no default program. Pitches without
both selectors established are ignored. A malformed prefix before a late
delimiter is discarded. With no delimiter, explicit body events still
decode, but no notes are tied across that boundary. Legacy parsing is unchanged.

The independent prompts of Pitch Tie permit batching with an appropriate
runtime, but choosing the mode alone does not imply batch execution or a
speedup. In all modes an exhausted chunk is retried once with a fresh KV cache,
empty held notes and a bare tie delimiter; no failed attempt's tokens are kept.
The empty delimiter on this exceptional retry does not force previous notes.

The existing browser `ModelService.transcribe(file, onProgress, { boundaryMode })`
forwards the optional choice and returns `{ notes, chunkStatuses }`. Existing
UI calls without a mode use Pitch Tie. Consumers may use the returned statuses to
distinguish a skipped chunk from silence.
