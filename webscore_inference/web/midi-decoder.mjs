const DRUM_PROGRAM = 128;
const CHUNK_SEC = 5.0;
const FRAME_RATE_HZ = 100.0;

export function buildTokenLookup(tokenizerJson) {
  const idToEvent = tokenizerJson.id_to_token.map((label, id) => {
    if (label.startsWith("<")) return { id, kind: label, value: 0 };
    if (label === "tie") return { id, kind: "tie", value: 0 };
    const [kind, rawVal] = label.split(":");
    return { id, kind, value: Number(rawVal) };
  });
  const eventToId = new Map();
  for (const ev of idToEvent) {
    eventToId.set(`${ev.kind}:${ev.value}`, ev.id);
  }
  return {
    padId: tokenizerJson.pad_id,
    bosId: tokenizerJson.bos_id,
    eosId: tokenizerJson.eos_id,
    unkId: tokenizerJson.unk_id,
    idToEvent,
    eventToId,
  };
}

export function encodeTiePrologue(openNotes, lookup) {
  if (!openNotes.length) {
    return [lookup.eventToId.get("tie:0")];
  }
  const sorted = [...openNotes].sort((a, b) => a.program - b.program || a.pitch - b.pitch);
  const tokens = [];
  let currentProg = null;
  for (const n of sorted) {
    if (n.program !== currentProg) {
      const id = lookup.eventToId.get(`inst:${n.program}`);
      if (id !== undefined) tokens.push(id);
      currentProg = n.program;
    }
    const pitchId = lookup.eventToId.get(`pitch:${n.pitch}`);
    if (pitchId !== undefined) tokens.push(pitchId);
  }
  tokens.push(lookup.eventToId.get("tie:0"));
  return tokens;
}

export function decodeChunkTokens(tokenIds, lookup, { chunkStartSec = 0.0, openNotes = [] } = {}) {
  const completed = [];
  const active = new Map();
  for (const note of openNotes) {
    active.set(`${note.program}:${note.pitch}`, { ...note });
  }

  const eosIdx = tokenIds.indexOf(lookup.eosId);
  const validTokens = eosIdx >= 0 ? tokenIds.slice(0, eosIdx) : tokenIds;

  let curTime = 0.0;
  let curInst = 0;
  let curVel = 1;
  let inTieSection = validTokens.includes(lookup.eventToId.get("tie:0"));
  const tieDeclared = new Set();

  for (const tid of validTokens) {
    const ev = lookup.idToEvent[tid];
    if (!ev) continue;

    if (ev.kind === "tie") {
      inTieSection = false;
      for (const [key, note] of [...active.entries()]) {
        if (!tieDeclared.has(key)) {
          note.offset = Math.max(note.onset + 0.01, chunkStartSec);
          completed.push(note);
          active.delete(key);
        }
      }
      continue;
    }

    if (inTieSection) {
      if (ev.kind === "inst") curInst = ev.value;
      else if (ev.kind === "pitch") tieDeclared.add(`${curInst}:${ev.value}`);
      continue;
    }

    if (ev.kind === "time") {
      curTime = ev.value / FRAME_RATE_HZ;
    } else if (ev.kind === "inst") {
      curInst = ev.value;
    } else if (ev.kind === "vel") {
      curVel = ev.value;
    } else if (ev.kind === "drum") {
      const t = chunkStartSec + curTime;
      completed.push({ isDrum: true, program: DRUM_PROGRAM, pitch: ev.value, onset: t, offset: t + 0.05 });
    } else if (ev.kind === "pitch") {
      const t = chunkStartSec + curTime;
      const key = `${curInst}:${ev.value}`;
      if (curVel > 0) {
        if (active.has(key)) {
          const prev = active.get(key);
          prev.offset = Math.max(prev.onset + 0.01, t);
          completed.push(prev);
        }
        active.set(key, { isDrum: false, program: curInst, pitch: ev.value, onset: t, offset: t });
      } else if (active.has(key)) {
        const note = active.get(key);
        note.offset = Math.max(note.onset + 0.01, t);
        completed.push(note);
        active.delete(key);
      }
    }
  }

  return { completed, nextOpenNotes: [...active.values()], nextChunkStartSec: chunkStartSec + CHUNK_SEC };
}

export function finalizeOpenNotes(openNotes, endSec) {
  return openNotes.map((n) => ({ ...n, offset: Math.max(n.onset + 0.01, endSec) }));
}

function writeVarLen(val) {
  const bytes = [val & 0x7f];
  let v = val >> 7;
  while (v > 0) {
    bytes.unshift((v & 0x7f) | 0x80);
    v >>= 7;
  }
  return bytes;
}

export function encodeNotesToMidi(notes, { bpm = 120, ppq = 480 } = {}) {
  const secToTick = (sec) => Math.max(0, Math.round((sec * bpm * ppq) / 60.0));
  const tracksEvents = new Map();

  for (const n of notes) {
    const key = n.isDrum ? "drum" : `p:${n.program}`;
    if (!tracksEvents.has(key)) tracksEvents.set(key, { isDrum: n.isDrum, program: n.program, events: [] });
    const tOn = secToTick(n.onset);
    const tOff = Math.max(tOn + 1, secToTick(n.offset));
    const list = tracksEvents.get(key).events;
    list.push({ tick: tOn, type: 0x90, pitch: n.pitch, vel: 90 });
    list.push({ tick: tOff, type: 0x80, pitch: n.pitch, vel: 0 });
  }

  const encodedTracks = [];
  const tempoUsec = Math.round(60_000_000 / bpm);
  encodedTracks.push(
    Uint8Array.from([
      0x00, 0xff, 0x51, 0x03,
      (tempoUsec >> 16) & 0xff, (tempoUsec >> 8) & 0xff, tempoUsec & 0xff,
      0x00, 0xff, 0x2f, 0x00,
    ])
  );

  let nextChannel = 0;
  for (const track of tracksEvents.values()) {
    const ch = track.isDrum ? 9 : nextChannel === 9 ? (nextChannel = 10, nextChannel++) : nextChannel++;
    if (ch > 15) {
      throw new RangeError("MIDI export supports at most 15 melodic instruments plus percussion; no tracks were exported.");
    }
    const raw = [];
    if (!track.isDrum) {
      raw.push(0x00, 0xc0 | ch, track.program & 0x7f);
    }
    track.events.sort((a, b) => a.tick - b.tick || a.type - b.type);
    let lastTick = 0;
    for (const ev of track.events) {
      raw.push(...writeVarLen(ev.tick - lastTick), ev.type | ch, ev.pitch & 0x7f, ev.vel & 0x7f);
      lastTick = ev.tick;
    }
    raw.push(0x00, 0xff, 0x2f, 0x00);
    encodedTracks.push(Uint8Array.from(raw));
  }

  const chunks = [];
  const header = new Uint8Array(14);
  const hv = new DataView(header.buffer);
  header.set([0x4d, 0x54, 0x68, 0x64], 0);
  hv.setUint32(4, 6);
  hv.setUint16(8, 1);
  hv.setUint16(10, encodedTracks.length);
  hv.setUint16(12, ppq);
  chunks.push(header);

  for (const body of encodedTracks) {
    const trkHead = new Uint8Array(8);
    const tv = new DataView(trkHead.buffer);
    trkHead.set([0x4d, 0x54, 0x72, 0x6b], 0);
    tv.setUint32(4, body.length);
    chunks.push(trkHead, body);
  }

  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(totalLen);
  let pos = 0;
  for (const c of chunks) {
    out.set(c, pos);
    pos += c.length;
  }
  return out;
}
