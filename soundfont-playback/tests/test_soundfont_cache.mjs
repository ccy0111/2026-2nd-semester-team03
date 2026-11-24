import test from 'node:test';
import assert from 'node:assert/strict';
import {webcrypto as crypto} from 'node:crypto';
import {createSoundfontCache} from '../src/soundfont-cache.mjs';
const data = new TextEncoder().encode('RIFF example sf2 bytes').buffer;
const hash = Buffer.from(await crypto.subtle.digest('SHA-256', data)).toString('hex');
const spec = {url:'./piano-v1.sf2',sha256:hash,bytes:data.byteLength};
function fakeIDB({quota=false,blocked=false}={}) {
  const records = new Map();
  return {records, open() {
    const req = {};
    queueMicrotask(() => {
      if (blocked) {req.onblocked?.(); return;}
      req.result = {close(){}, transaction() {
        const tx = {objectStore() {return {
          get(key){return op(() => records.has(key) ? records.get(key).slice(0) : undefined);},
          put(value,key){return op(() => {if(quota) throw Error('Quota'); records.set(key,value.slice(0));});},
          delete(key){return op(() => records.delete(key));},
        };}};
        function op(fn) {const r={};queueMicrotask(()=>{try {r.result=fn();r.onsuccess?.();tx.oncomplete?.();}catch {tx.onerror?.();}});return r;}
        return tx;
      }};
      req.onsuccess?.();
    });
    return req;
  }};
}
function setup(options={}) {
 let calls=0;
 const indexedDB=options.indexedDB ?? fakeIDB();
 const cache=createSoundfontCache({crypto, indexedDB, baseUrl:'https://example.test/fonts/', fetch:async()=>{calls++;return {ok:true,arrayBuffer:async()=>data.slice(0)};},...options});
 return {cache,indexedDB,calls:()=>calls};
}
test('verified immutable entry supports offline hit across factory instances',async()=>{
 const s=setup();assert.deepEqual(await s.cache.load(spec),data);assert.equal(s.calls(),1);
 const offline=setup({indexedDB:s.indexedDB,fetch:async()=>{throw Error('offline');}});
 assert.deepEqual(await offline.cache.load(spec),data);
});
test('dedupe returns independent buffers safe for worker transfer',async()=>{
 const s=setup();const [a,b]=await Promise.all([s.cache.load(spec),s.cache.load(spec)]);
 assert.equal(s.calls(),1);assert.notEqual(a,b);new Uint8Array(a)[0]=0;assert.deepEqual(b,data);
});
test('changed hash or URL does not reuse old entry; corruption is refetched',async()=>{
 const s=setup();await s.cache.load(spec);const key=[...s.indexedDB.records.keys()][0];
 s.indexedDB.records.set(key,new ArrayBuffer(data.byteLength));await s.cache.load(spec);assert.equal(s.calls(),2);
 await s.cache.load({...spec,url:'./piano-v2.sf2'});assert.equal(s.calls(),3);
 await assert.rejects(s.cache.load({...spec,sha256:'0'.repeat(64)}),/integrity/);assert.equal(s.calls(),4);
});
test('invalid size/hash never persists and retry works',async()=>{
 const s=setup();await assert.rejects(s.cache.load({...spec,bytes:1}),/integrity/);assert.equal(s.indexedDB.records.size,0);
 await s.cache.load(spec);assert.equal(s.calls(),2);
});
test('different expected lengths cannot piggyback on successful concurrent load',async()=>{
 const s=setup();const out=await Promise.allSettled([s.cache.load(spec),s.cache.load({...spec,bytes:1})]);
 assert.equal(out[0].status,'fulfilled');assert.equal(out[1].status,'rejected');
});
test('quota, blocked, missing, throwing and hung storage fall back to download',async()=>{
 for(const indexedDB of [fakeIDB({quota:true}),fakeIDB({blocked:true}),null,{open(){throw Error('private');}},{open(){return {};}}]){
 const s=setup({indexedDB,cacheTimeoutMs:5});assert.deepEqual(await s.cache.load(spec),data);assert.equal(s.calls(),1);
 }
});
test('aborted body is not cached; independent normal caller is not cancelled',async()=>{
 const idb=fakeIDB();const ctrl=new AbortController();let calls=0;
 const s=setup({indexedDB:idb,fetch:async(_url,{signal})=>{calls++;return {ok:true,arrayBuffer:async()=>{if(signal)ctrl.abort();return data.slice(0);}};}});
 const out=await Promise.allSettled([s.cache.load({...spec,signal:ctrl.signal}),s.cache.load(spec)]);
 assert.equal(out[0].status,'rejected');assert.equal(out[1].status,'fulfilled');assert.equal(calls,2);assert.equal(idb.records.size,1);
 const pre=new AbortController();pre.abort();await assert.rejects(s.cache.load({...spec,signal:pre.signal}));assert.equal(calls,2);
});
test('failed HTTP/body and invalid digest do not poison retry',async()=>{
 const idb=fakeIDB();let mode=0;
 const s=setup({indexedDB:idb,fetch:async()=>({ok:mode!==0,status:500,arrayBuffer:async()=>{if(mode===1)throw Error('interrupted');return data.slice(0);}})});
 await assert.rejects(s.cache.load(spec),/500/);mode=1;await assert.rejects(s.cache.load(spec),/interrupted/);assert.equal(idb.records.size,0);mode=2;assert.deepEqual(await s.cache.load(spec),data);
});
test('manifest arguments and crypto fail closed before fetching',async()=>{
 const s=setup();for(const change of [{sha256:undefined},{sha256:'bad'},{bytes:0},{bytes:1.5},{url:'data:text/plain,hi'},{url:'file:///tmp/a'},{url:'https://a:b@example.test/f'}]) await assert.rejects(s.cache.load({...spec,...change}));
 assert.equal(s.calls(),0);await assert.rejects(setup({crypto:null}).cache.load(spec),/SHA-256 is unavailable/);
});
