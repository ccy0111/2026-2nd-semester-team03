import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createGroupedPlayer} from '../src/player.mjs';
const note = program => ({program, pitch:60, onset:0, offset:1});
function fixture() {
  const calls=[];
  const listeners = new Map();
  const seq = {
    duration:1,currentHighResolutionTime:.24,
    eventHandler:{addEvent:(k,_,f)=>listeners.set(k,f), removeEvent:k=>listeners.delete(k)},
    play:()=>calls.push('play'),
    pause:()=>calls.push('pause'),
  };
  seq.loadNewSongList = ([{binary}]) => {
    assert.ok(binary.byteLength>20);
    calls.push('midi');
    listeners.get('songChange')();
  };
  const destination={gain:{value:1}};
  const synth={
    connect:()=>calls.push('connect'),
    isReady:Promise.resolve(),
    soundBankManager:{addSoundBank:(_,id)=>calls.push(id)},
    destroy:()=>calls.push('destroy')
  };
  const context={resume:()=>{calls.push('resume');return Promise.resolve();},
    audioWorklet:{addModule:()=>calls.push('worklet')}};
  const cache={load:({url})=>{calls.push(`fetch:${url}`);return new ArrayBuffer(8);}};
  const player=createGroupedPlayer({context,workletUrl:'/worklet',destination,cache,
    banks:{piano:{url:'/piano'},guitars:{url:'/guitars'}},createSynth:()=>synth,createSequencer:()=>seq});
  return {player,calls,seq,destination,listeners};
}
test('prepare requests only banks used by the score and keeps model notes intact',async()=>{
  const {player,calls,seq,destination,listeners}=fixture();
  const grouped=await player.prepare([note(0),note(25),note(55)]);
  assert.deepEqual(grouped.unsupportedPrograms,[55]);
  assert.deepEqual(calls,['resume','worklet','connect','fetch:/piano','piano','fetch:/guitars','guitars','midi']);
  assert.equal(listeners.size,0);
  assert.equal(player.currentTime,.24);
  await player.play();
  player.pause(); player.seek(8); player.setMuted(true);
  assert.equal(seq.currentTime,1);
  assert.equal(destination.gain.value,0);
  player.dispose();
  assert.deepEqual(calls.slice(-5),['resume','play','pause','pause','destroy']);
});
test('unsupported-only score avoids opening AudioWorklet',async()=>{
  const {player,calls}=fixture();
  assert.deepEqual((await player.prepare([note(55)])).unsupportedPrograms,[55]);
  assert.deepEqual(calls,[]);
});
test('missing bank fails explicitly and cleans up',async()=>{
  const {player,calls}=fixture();
  await assert.rejects(player.prepare([note(40)]),/bank is not configured: strings/);
  assert.equal(calls.at(-1),'destroy');
});
