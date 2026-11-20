import {Sequencer, WorkletSynthesizer} from 'spessasynth_lib';
import {groupPlaybackNotes} from './grouping.mjs';
import {buildPlaybackMidi} from './playback-midi.mjs';
import {createSoundfontCache} from './soundfont-cache.mjs';
export {groupPlaybackNotes, INSTRUMENT_GROUPS} from './grouping.mjs';

function loadSong(sequencer, binary) {
  const e = sequencer.eventHandler;
  return new Promise((resolve,reject) => {
    let timeout;
    const settle = (error) => {
      clearTimeout(timeout);
      e.removeEvent('midiError', 'load');
      e.removeEvent('songChange', 'load');
      error ? reject(error) : resolve();
    };
    e.addEvent('songChange','load', () => settle());
    e.addEvent('midiError','load', error => settle(error));
    timeout = setTimeout(() => settle(new Error('Playback MIDI load timed out')), 20000);
    try { sequencer.loadNewSongList([{binary}]); } catch(error) { settle(error); }
  });
}

/**
 * Call prepare() directly from a click (Web Audio user activation).
 * One player per active score. Dispose it before replacing that score.
 * The caller owns the AudioContext and output node.
 */
export function createGroupedPlayer({context, workletUrl, banks, cache = createSoundfontCache(),
  destination = context.destination, createSynth = c => new WorkletSynthesizer(c),
  createSequencer = synth => new Sequencer(synth, {skipToFirstNoteOn:false})}) {
  const abort = new AbortController();
  let synth, sequence, disposed = false, preparing, ready = false;
  const requireOpen = () => { if (disposed) throw new Error('Playback session is closed'); };
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    ready = false;
    abort.abort();
    sequence?.pause();
    synth?.destroy();
  };
  return {
    prepare(notes) {
      requireOpen();
      if (preparing) return preparing;
      const grouped = groupPlaybackNotes(notes);
      if (!grouped.tracks.length) return Promise.resolve(grouped);
      preparing = (async () => {
        // Resume before awaiting any network or AudioWorklet operation.
        await context.resume();
        requireOpen();
        await context.audioWorklet.addModule(workletUrl);
        requireOpen();
        synth = createSynth(context);
        synth.connect(destination);
        await synth.isReady;
        for (const id of grouped.banks) {
          requireOpen();
          if (!banks?.[id]) throw new Error(`SoundFont bank is not configured: ${id}`);
          const buffer = await cache.load({...banks[id], signal:abort.signal});
          requireOpen();
          await synth.soundBankManager.addSoundBank(buffer,id);
        }
        requireOpen();
        sequence = createSequencer(synth);
        sequence.loopCount = 0;
        await loadSong(sequence, buildPlaybackMidi(grouped.tracks).writeMIDI());
        requireOpen();
        ready = true;
        return grouped;
      })().catch(error => { dispose(); throw error; });
      return preparing;
    },
    async play() {
      requireOpen();
      if (!ready) return;
      await context.resume();
      requireOpen();
      sequence.play();
    },
    pause() { if (ready) sequence.pause(); },
    seek(seconds) {
      if (ready && Number.isFinite(seconds)) {
        sequence.currentTime = Math.max(0,Math.min(sequence.duration,seconds));
      }
    },
    setMuted(muted) {
      // The caller supplies a GainNode when it wants master mute.
      if (destination.gain) destination.gain.value = muted ? 0 : 1;
    },
    get currentTime() { return ready ? sequence.currentHighResolutionTime : 0; },
    get duration() { return ready ? sequence.duration : 0; },
    get paused() { return !ready || sequence.paused; },
    dispose,
  };
}
