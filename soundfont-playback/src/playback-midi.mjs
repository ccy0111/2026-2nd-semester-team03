import {MIDIBuilder} from 'spessasynth_core';
const melodicChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
const ticks = seconds => Math.round(seconds * 960);

// Merging simultaneous notes routed to the same channel/pitch avoids a
// second track's note-off prematurely silencing the first.
function intervals(notes, drums) {
  const ordered = notes.map(n => ({start: ticks(n.onset), end: Math.max(ticks(n.onset)+1, ticks(n.offset)), pitch:n.pitch}))
    .sort((a,b) => a.pitch - b.pitch || a.start - b.start || a.end - b.end);
  const out = [];
  for (const n of ordered) {
    const last = out.at(-1);
    if (!drums && last?.pitch === n.pitch && n.start < last.end) last.end = Math.max(last.end, n.end);
    else out.push(n);
  }
  return out;
}

export function buildPlaybackMidi(tracks) {
  const midi = new MIDIBuilder({timeDivision:480, format:1});
  midi.setTempo(0, 120);
  let melodic = 0;
  tracks.forEach((t,i) => {
    const drum = t.id === 36;
    const port = drum ? 0 : Math.floor(melodic / 15);
    const channel = drum ? 9 : melodicChannels[melodic++ % 15];
    midi.addTrack(t.label, port);
    const track = i + 1;
    midi.programChange(0, track, channel, drum ? 0 : t.program);
    const events = intervals(t.notes, drum).flatMap(n => [
      {tick:n.start, on:true, pitch:n.pitch}, {tick:n.end,on:false,pitch:n.pitch},
    ]).sort((a,b) => a.tick - b.tick || Number(a.on) - Number(b.on));
    for (const e of events) {
      if (e.on) midi.noteOn(e.tick, track, channel, e.pitch, 90);
      else midi.noteOff(e.tick, track, channel, e.pitch);
    }
  });
  midi.flush();
  return midi;
}
