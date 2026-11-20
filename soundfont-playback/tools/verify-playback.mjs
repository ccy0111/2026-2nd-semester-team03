import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {SoundBankLoader,SpessaSynthProcessor,SpessaSynthSequencer} from 'spessasynth_core';
import {INSTRUMENT_GROUPS,groupPlaybackNotes} from '../src/grouping.mjs';
import {buildPlaybackMidi} from '../src/playback-midi.mjs';

const base = process.argv[2];
if (!base) throw new Error('Usage: node tools/verify-playback.mjs <generated-banks-dir>');
const synth = new SpessaSynthProcessor(22050);
// Use isolated sample peak windows, without inter-note effect tails.
synth.setSystemParameter('reverbGain',0);
synth.setSystemParameter('chorusGain',0);
const groups = INSTRUMENT_GROUPS;
const notes = groups.map((g,i)=>({program:g.program,pitch:g.id===36?36:60,onset:i,offset:i+.25}));
const g = groupPlaybackNotes(notes);
for (const id of g.banks) {
  const raw = await readFile(resolve(base,`MuseScore_General_${id}.sf2`));
  synth.soundBankManager.addSoundBank(SoundBankLoader.fromArrayBuffer(
    raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)),id);
}
const seq = new SpessaSynthSequencer(synth);
seq.loopCount=0;
seq.loadNewSongList([buildPlaybackMidi(g.tracks)]);
seq.play();
const l=new Float32Array(128), r=new Float32Array(128);
const peaks = new Float32Array(groups.length);
for(let i=0;i<22050*groups.length;i+=128) {
  seq.processTick();
  synth.process(l,r);
  const s=i/22050;
  if (s%1>.02 && s%1<.23) {
    for (let z=0;z<l.length;z++) {
      assert.ok(Number.isFinite(l[z])&&Number.isFinite(r[z]));
      peaks[Math.floor(s)]=Math.max(peaks[Math.floor(s)],Math.abs(l[z]),Math.abs(r[z]));
    }
  }
}
for (const [i,peak] of peaks.entries()) assert.ok(peak>0.00001,`Silent: ${groups[i].label}`);
console.log(`SpessaSynth rendered all ${peaks.length} groups across multiple MIDI ports.`);
