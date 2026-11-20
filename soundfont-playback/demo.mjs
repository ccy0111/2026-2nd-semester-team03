import {createGroupedPlayer} from './dist/grouped-player.mjs';

const el = id => document.getElementById(id);
const response = await fetch('./soundfonts.json');
if (!response.ok) throw new Error(`SoundFont manifest HTTP ${response.status}`);
const {banks} = await response.json();
const context = new AudioContext();
const volume = context.createGain();
volume.connect(context.destination);
let player, selected;
const note = (program,pitch,onset,offset)=>({program,pitch,onset,offset});
const piano = [note(0,60,0,.5),note(0,64,.35,.9),note(0,67,.65,1.3)];
const ensemble = [note(25,55,0,.65),note(25,59,.55,1.2),note(128,36,0,.1),note(128,38,.55,.65)];
el('play').onclick = async () => {
  el('play').disabled = true;
  try {
    const mixed = el('ensemble').checked;
    if (!player || selected !== mixed) {
      selected = mixed;
      player?.dispose();
      player = createGroupedPlayer({context, destination: volume, banks,
        workletUrl: './dist/spessasynth_processor.min.js'});
      el('status').value = '사운드폰트 불러오는 중';
      const result = await player.prepare(mixed ? [...piano,...ensemble] : piano);
      el('status').value = result.tracks.map(x => x.label).join(', ');
    }
    await player.play();
  } catch(error) {
    player?.dispose();
    player = undefined;
    el('status').value = error.message;
  } finally { el('play').disabled = false; }
};
el('pause').onclick = () => player?.pause();
el('stop').onclick = () => { player?.pause(); player?.seek(0); };
el('mute').onchange = () => { volume.gain.value = el('mute').checked ? 0 : 1; };
