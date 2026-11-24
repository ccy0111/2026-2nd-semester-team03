import {test} from 'node:test';
import assert from 'node:assert/strict';
import {INSTRUMENT_GROUPS, groupPlaybackNotes} from '../src/grouping.mjs';
import {buildPlaybackMidi} from '../src/playback-midi.mjs';
import {BasicMIDI} from 'spessasynth_core';
const note = (program,onset=0,offset=1) => Object.freeze({program,pitch:60,onset,offset});
test('official group ids and original notes survive; excluded and unknown programs are reported',() => {
  const notes=[note(55),note(25),note(24),note(30),note(129),note(0),note(128)];
  const g=groupPlaybackNotes(notes);
  assert.deepEqual(g.tracks.map(x=>[x.id,x.program,x.notes.length]),[[0,0,1],[4,24,2],[6,29,1],[36,128,1]]);
  assert.deepEqual(g.unsupportedPrograms,[55,129]);
  assert.deepEqual(g.banks,['piano','guitars','drums']);
  assert.equal(g.tracks[1].notes[0],notes[1]);
  assert.equal(INSTRUMENT_GROUPS.length,36);
});
test('playback MIDI uses representative program, coalesces overlapping notes, uses drum channel',() => {
  const result=groupPlaybackNotes([note(24,0,2),note(25,0.5,3),note(128)]);
  const midi=buildPlaybackMidi(result.tracks);
  assert.equal(midi.timeDivision,480);
  assert.equal(midi.duration,3);
  const [_,guitar,drum]=midi.tracks;
  assert.ok(guitar.events.some(e=>e.statusByte===0xc0 && e.data[0]===24));
  assert.equal(guitar.events.filter(e=>e.statusByte===0x90).length,1);
  assert.ok(drum.events.some(e=>e.statusByte===0x99));
});
test('more than 15 melodic groups use multiple MIDI ports',() => {
  const result=groupPlaybackNotes(INSTRUMENT_GROUPS.map(x=>note(x.program)));
  const midi=buildPlaybackMidi(result.tracks);
  assert.equal(midi.tracks.length,37);
  const ports=midi.tracks.flatMap(t=>t.events.filter(e=>e.statusByte===0x21).map(e=>e.data[0]));
  assert.ok(ports.includes(2));
  assert.ok(midi.writeMIDI().byteLength>1000);
});
