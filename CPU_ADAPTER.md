# CPU WebScore backend adapter

The existing FastAPI `Transcriber` / `ScoreImage` contract is unchanged. The
adapter returns UTF-8 SVG bytes (`image/svg+xml`); the backend retains and serves
the original uploaded MP3 separately. No weights or training artifacts are
included or downloaded. No training process is started or stopped.

## Installation and model placement

Install the existing backend requirements plus `numpy>=1.26,<3` and
`onnxruntime>=1.18,<2`. Install FFmpeg from its official distribution and ensure
both `ffmpeg` and `ffprobe` are on the server PATH. For tests install
`httpx>=0.28,<1`, then run `python -m unittest discover -s tests -v`.

Copy a compatible completed WebScore export into the directory passed to
`load_model` (the backend's models directory):

- `config.json` and `tokenizer.json` from that exact model export
- `mel_encoder.onnx`
- `token_embed.onnx`
- `decoder_kv.int8.onnx` (preferred) or `decoder_kv.onnx`
- any external-data sidecar files referenced by these ONNX exports

Do not copy an export while it is being written. The committed source-package
metadata is not automatically substituted for missing model metadata. Missing
or invalid model exports raise `ModelLoadError`; the existing backend remains
available but transcription returns 503. The backend logs the startup reason.
The adapter uses ONNX Runtime CPUExecutionProvider with two intra-op threads.

## Contract and bounded processing

MP3 format and audio codec are checked with ffprobe, rather than trusting the
filename. The FFmpeg processes permit only file/pipe protocols. Audio is decoded
to mono float32 at 16 kHz, then processed in five-second chunks. The frontend uses
a periodic 2048-sample Hann window, 160-sample hop, reflect-centered STFT, 512 HTK
mel magnitude filters and natural log with epsilon 1e-6. Encoder, embedding and
KV-decoder input/output names match the browser export contract.

The implementation limits decoded audio to 30 minutes and generation to 1024
new tokens per chunk. Exceeding the audio limit returns the backend's invalid
audio error; reaching the token limit without EOS raises an inference error
instead of returning a silently truncated score. Predictions inside final-chunk
padding are discarded. Note ties preserve note onsets across chunks. A score
preview above 20,000 note events raises a rendering error.

## Score preview limitations

This is a staff-notation preview, not publication-quality engraving. Timing is
quantized to sixteenth notes at fixed 120 BPM and 4/4. Noteheads, stems, flags,
ledger lines and duration ties are drawn on instrument-separated treble staffs.
Pitch spelling assumes C major with sharps; tempo/key detection, rests and
independent polyphonic voices are not inferred. Percussion is shown by MIDI
pitch rather than standardized percussion notation. Each staff's vertical space
adapts to its complete pitch range. These limitations are printed on the SVG.

## Verification status

Tests cover metadata validation, silent/short-input mel shape and values,
cross-chunk ties, SVG XML validity, mock ONNX KV shape/position handling,
audio/error mapping, and existing API SVG/original-MP3 behavior. They do **not**
validate transcription accuracy, real-model memory consumption, latency, or
numerical equivalence of actual exported ONNX inference. A completed compatible
model export and real MP3 fixtures are still required for that acceptance test.

An independent deterministic 80,000-sample broadband fixture comparison against
the committed JavaScript mel frontend passed at atol/rtol 1e-4 (maximum absolute
difference 4.7683716e-06; mean 8.041436e-08). This verifies that fixture, not all
possible inputs or the ONNX model itself.

A separate real FFmpeg smoke check decoded a generated 0.3-second MP3 into 4,800 mono samples, rejected WAV as unsupported, and rejected corrupt MP3 as invalid. No real model was used.
