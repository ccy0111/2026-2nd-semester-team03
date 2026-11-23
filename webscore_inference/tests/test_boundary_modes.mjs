import test from 'node:test';
import assert from 'node:assert/strict';
import { WebScoreClient, TokenLimitError } from '../web/model-client.mjs';
import { buildTokenLookup, decodeChunkTokens } from '../web/midi-decoder.mjs';

const tokenizer = { pad_id:0, bos_id:1, eos_id:2, unk_id:3, id_to_token:[
  '<pad>','<bos>','<eos>','<unk>','tie','inst:0','inst:1','inst:2','inst:3',
  'pitch:60','pitch:62','time:0','time:100','vel:0','vel:1','drum:60',
] };
const lookup = buildTokenLookup(tokenizer);
const tokens = (...events) => [...events.map(e => lookup.eventToId.get(e === 'tie' ? 'tie:0' : e)), lookup.eosId];
const held = (program, pitch=60) => ({ isDrum:false, program, pitch, onset:4, offset:4 });
const pitchTie = (events, openNotes) => decodeChunkTokens(tokens(...events), lookup, {
  chunkStartSec:5, openNotes, boundaryMode:'pitch-tie',
});
const pitches = notes => notes.map(n => [n.program,n.pitch,n.onset,n.offset]);

test('pitch tie reserves exact matches globally before accepting aliases', () => {
  for (const declarations of [
    ['inst:2','pitch:60','inst:1','pitch:60'],
    ['inst:1','pitch:60','inst:2','pitch:60'],
  ]) {
    const result = pitchTie([...declarations,'tie','time:100','vel:0','inst:2','pitch:60','inst:1','pitch:60'],
      [held(0),held(1)]);
    assert.deepEqual(pitches(result.completed), [[0,60,4,6],[1,60,4,6]]);
    assert.deepEqual(result.nextOpenNotes, []);
  }
});

test('pitch tie rejects ambiguity on either side and ignores orphan declarations and offs', () => {
  for (const [declarations, previous] of [
    [['inst:2','pitch:60'],[held(0),held(1)]],
    [['inst:2','pitch:60','inst:3','pitch:60'],[held(0)]],
    [['inst:2','pitch:62'],[]],
  ]) {
    const result = pitchTie([...declarations,'tie','time:100','inst:2','vel:0','pitch:60','pitch:62'],previous);
    assert.deepEqual(pitches(result.completed), previous.map(n => [n.program,n.pitch,n.onset,5]));
    assert.deepEqual(result.nextOpenNotes, []);
  }
});

test('pitch alias closes the original on noteoff; original and alias retriggers start real body notes', () => {
  for (const program of [0, 2]) {
    const events = ['inst:2','pitch:60','tie','time:100',`inst:${program}`,'vel:1','pitch:60','inst:2','vel:0','pitch:60'];
    const result = pitchTie(events, [held(0)]);
    assert.deepEqual(pitches(result.completed).slice(0,1), [[0,60,4,6]]);
    if (program === 0) assert.deepEqual(pitches(result.nextOpenNotes), [[0,60,6,6]]);
    else assert.equal(result.nextOpenNotes.length, 0);
  }
  const off = pitchTie(['inst:2','pitch:60','tie','time:100','inst:0','vel:0','pitch:60'], [held(0)]);
  assert.deepEqual(pitches(off.completed), [[0,60,4,6]]);
});

test('pitch aliases last one chunk and missing/late tie does not consume body or preserve undeclared notes', () => {
  const first = pitchTie(['inst:2','pitch:60','tie'], [held(0)]);
  assert.deepEqual(pitches(first.nextOpenNotes), [[0,60,4,4]]);
  const second = decodeChunkTokens(tokens('inst:2','pitch:60','tie','time:100','vel:0','pitch:60'), lookup,
    { chunkStartSec:10, openNotes:first.nextOpenNotes, boundaryMode:'pitch-tie' });
  assert.deepEqual(pitches(second.completed), [[0,60,4,11]]);
  const late = pitchTie(['inst:1','pitch:62','time:0','vel:1','pitch:62','tie','time:100','inst:1','vel:0','pitch:62'], [held(0)]);
  assert.deepEqual(pitches(late.completed), [[0,60,4,5]]);
});

test('explicit modes carry prelude program but ignore pitches before program or velocity is set', () => {
  for (const boundaryMode of ['prelude-forcing','pitch-tie']) {
    const decode = (...events) => decodeChunkTokens(tokens(...events), lookup, {boundaryMode,openNotes:[held(1)],chunkStartSec:5});
    const good = decode('inst:1','pitch:60','tie','time:100','pitch:62','pitch:60','vel:0','pitch:60');
    assert.deepEqual(pitches(good.completed),[[1,60,4,6]]);
    assert.deepEqual(good.nextOpenNotes,[]);
    const emptyPrelude = decode('tie','time:100','vel:1','pitch:62','inst:2','pitch:60');
    assert.deepEqual(pitches(emptyPrelude.nextOpenNotes),[[2,60,6,6]]);
    const noTie = decode('time:0','inst:2','pitch:60','vel:1','pitch:62','time:100','vel:0','pitch:62');
    assert.deepEqual(pitches(noTie.completed),[[1,60,4,5],[2,62,5,6]]);
  }
  const strict = decodeChunkTokens(tokens('inst:2','pitch:60','tie'),lookup,
    { boundaryMode:'prelude-forcing',openNotes:[held(1)],chunkStartSec:5 });
  assert.deepEqual(pitches(strict.completed),[[1,60,4,5]]);
  assert.deepEqual(strict.nextOpenNotes,[]);
  const legacy = decodeChunkTokens(tokens('tie','pitch:62'),lookup);
  assert.deepEqual(pitches(legacy.nextOpenNotes),[[0,62,0,0]]);
});

function makeClient(run) {
  const c = new WebScoreClient({}, {tokenizer,arch:{}});
  c.transcribeChunk = run;
  return c;
}
test('boundary modes send mutually exclusive normal prompts and default to pitch tie', async () => {
  for (const [mode, prompts] of [
    [undefined, [[],[]]],
    ['legacy', [[],tokens('inst:0','pitch:60','tie').slice(0,-1)]],
    ['prelude-forcing',[tokens('tie').slice(0,-1),tokens('inst:0','pitch:60','tie').slice(0,-1)]],
    ['pitch-tie',[[],[]]],
  ]) {
    const seen = [];
    const c = makeClient(async (_, {promptTokens}) => {
      seen.push(promptTokens);
      if (seen.length === 1) return tokens('tie','time:0','inst:0','vel:1','pitch:60');
      return tokens('inst:0','pitch:60','tie','time:100','vel:0','pitch:60');
    });
    const out = await c.transcribePcm(new Float32Array(160000), {boundaryMode:mode});
    assert.deepEqual(seen, prompts);
    assert.deepEqual(pitches(out.notes), [[0,60,0,6]]);
    assert.deepEqual(out.chunkStatuses.map(s => s.status), ['ok','ok']);
  }
});

test('Pitch Tie uses the same typed EOS recovery and rejects unknown modes before inference', async () => {
  const seen = [];
  const c = makeClient(async (_, o) => {
    seen.push(o.promptTokens);
    if (seen.length === 1) throw new TokenLimitError();
    return tokens('tie');
  });
  await assert.rejects(c.transcribePcm(new Float32Array(80000),{boundaryMode:'both'}), RangeError);
  const out = await c.transcribePcm(new Float32Array(80000),{boundaryMode:'pitch-tie'});
  assert.deepEqual(seen,[[],tokens('tie').slice(0,-1)]);
  assert.equal(out.chunkStatuses[0].status,'recovered');
  assert.equal(out.chunkStatuses[0].boundaryReset,true);
});
