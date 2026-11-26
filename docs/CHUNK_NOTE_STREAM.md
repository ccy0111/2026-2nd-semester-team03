# Streaming transcription notes

The existing final response and progress callback are unchanged. The UI can opt
into incremental piano-roll updates with:

```js
const finalized = [];
const result = await modelService.transcribe(file, onProgress, {
  onChunkNotes({ chunkStatus, finalizedNotes, provisionalNotes }) {
    for (const note of finalizedNotes) finalized.push(note);
    renderPreview(finalized, provisionalNotes, chunkStatus);
  },
});
renderFinal(result.notes);
```

- One ordered update per resolved five-second chunk after any EOS retry. On
  WebGPU, batched chunks may arrive together. No additional inference is run.
- Append each `finalizedNotes` delta once. Replace the entire provisional
  overlay with `provisionalNotes`; do not append or merge it into final notes.
  A carried note can later finalize with an onset in a previous chunk.
- A failed interval still sends a status update; show a gap, not confirmed
  silence. Its delta may close notes held from the previous chunk at the gap.
- The last update clears the overlay and flushes remaining notes. Replace the
  preview with the final result on completion, and clear preview state on a new
  run, cancellation, or error. Enable MIDI/PDF exports only after final success.
- The worker message is opt-in via `streamChunkNotes: true`; each message is
  `{ type: "chunk-notes", update }`. Direct inference callers may use
  `transcribePcm(pcm, { onChunkNotes })` with the same payload.
