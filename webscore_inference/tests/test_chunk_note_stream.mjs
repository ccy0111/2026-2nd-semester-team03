import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { WebScoreClient, TokenLimitError } from "../web/model-client.mjs";

const tokenizer = {
  pad_id: 0, bos_id: 1, eos_id: 2, unk_id: 3,
  id_to_token: ["<pad>", "<bos>", "<eos>", "<unk>", "inst:0", "pitch:60",
    "vel:1", "vel:0", "time:0", "time:100", "tie", "pitch:62"],
};
const held = [10, 8, 4, 6, 5, 2];
const close = [4, 5, 10, 9, 4, 7, 5, 2];
const shortFinal = [10, 8, 4, 6, 11, 9, 7, 11, 2];
const makeClient = run => Object.assign(new WebScoreClient({}, { tokenizer, arch: {} }), { transcribeChunk: run });

for (const boundaryMode of ["pitch-tie", "prelude-forcing", "legacy"]) {
  test(`${boundaryMode}: emit before next decode; streamed notes and final MIDI agree`, async () => {
    const pcm = new Float32Array(88000);
    const updates = [];
    let calls = 0;
    const streamed = await makeClient(async () => {
      if (!calls++) return held;
      assert.equal(updates.length, 1);
      return close;
    }).transcribePcm(pcm, { boundaryMode, onChunkNotes: update => updates.push(update) });
    let plainCalls = 0;
    const plain = await makeClient(async () => plainCalls++ ? close : held).transcribePcm(pcm, { boundaryMode });
    assert.deepEqual(streamed, plain);
    assert.deepEqual(updates.map(u => u.completedChunks), [1, 2]);
    assert.deepEqual(updates[0].finalizedNotes, []);
    assert.equal(updates[0].provisionalNotes[0].offset, 5);
    assert.deepEqual(updates[1].provisionalNotes, []);
    assert.deepEqual(updates.flatMap(u => u.finalizedNotes), streamed.notes);
    assert.equal(streamed.notes[0].offset, 5.5);
  });
}

test("EOS failure emits one gap and closes old carry without exposing runaway tokens", async () => {
  let calls = 0;
  const updates = [];
  const result = await makeClient(async () => {
    calls++;
    if (calls === 2 || calls === 3) throw new TokenLimitError();
    return calls === 1 ? held : shortFinal;
  }).transcribePcm(new Float32Array(176000), { onChunkNotes: u => updates.push(u) });
  assert.deepEqual(updates.map(u => u.chunkStatus.status), ["ok", "failed", "ok"]);
  assert.equal(updates[1].finalizedNotes[0].offset, 5);
  assert.deepEqual(updates[1].provisionalNotes, []);
  assert.deepEqual(updates.flatMap(u => u.finalizedNotes), result.notes);
});

test("stream callback cannot mutate final notes or statuses; abort stops later updates", async () => {
  const result = await makeClient(async () => held).transcribePcm(new Float32Array(16000), {
    onChunkNotes(u) { u.finalizedNotes[0].pitch = 99; u.chunkStatus.status = "failed"; },
  });
  assert.equal(result.notes[0].pitch, 60);
  assert.equal(result.chunkStatuses[0].status, "ok");
  const ac = new AbortController();
  let updates = 0;
  await assert.rejects(makeClient(async () => held).transcribePcm(new Float32Array(160000), {
    signal: ac.signal, onChunkNotes() { updates++; ac.abort(); },
  }), { name: "AbortError" });
  assert.equal(updates, 1);
});

test("worker and ModelService forward opt-in update before the existing result", async () => {
  const workerSource = await readFile(new URL("../../frontend/scripts/model.worker.js", import.meta.url), "utf8");
  const serviceSource = await readFile(new URL("../../frontend/scripts/ModelService.js", import.meta.url), "utf8");
  const delta = { completedChunks: 1, totalChunks: 2, finalizedNotes: [], provisionalNotes: [{ pitch: 60 }] };
  const final = { notes: [{ pitch: 60 }], chunkStatuses: [], midiBytes: Uint8Array.of(77, 84) };
  const events = [];
  let listener;
  const svcCtx = vm.createContext({ decodeMock: async () => new Float32Array(1) });
  vm.runInContext(serviceSource.replaceAll("export ", "").replace("await import(MEL_MODULE_URL)", "({ decodeAudioTo16kMono: decodeMock })") + "\nglobalThis.service = new ModelService();", svcCtx);
  const svc = svcCtx.service;
  const workerCtx = vm.createContext({
    self: { addEventListener(_, fn) { listener = fn; }, postMessage(m) { events.push(m.type); svc.handleMessage(m); } },
    mockModel: { async transcribePcm(_, opts) { opts.onChunkNotes?.(delta); return final; } },
  });
  vm.runInContext(workerSource + "\nmodel = mockModel;", workerCtx);
  svc.worker = { postMessage(data) { void listener({ data }); } };
  svc.ready = true;
  const updates = [];
  assert.deepEqual({ ...await svc.transcribe({}, null, { normalizeLoudness: false, onChunkNotes: u => updates.push(u) }) }, final);
  assert.deepEqual(events, ["chunk-notes", "result"]);
  assert.deepEqual(updates, [delta]);
  svc.worker = { postMessage() {} };
  const active = svc.transcribe({}, null, { normalizeLoudness: false });
  await assert.rejects(svc.transcribe({}, null, { normalizeLoudness: false }), /already running/);
  svc.handleMessage({ type: "result", ...final });
  await active;
  svc.worker = { postMessage(data) { void listener({ data }); } };
  await svc.transcribe({}, null, { normalizeLoudness: false });
  assert.deepEqual(events, ["chunk-notes", "result", "result"]);
});
