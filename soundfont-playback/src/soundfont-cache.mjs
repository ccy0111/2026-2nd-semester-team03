/** Immutable, integrity-checked SF2 download cache. No credentials or signed URLs belong in manifests. */
export function createSoundfontCache({
  fetch: fetchImpl = (...args) => globalThis.fetch(...args),
  indexedDB = globalThis.indexedDB,
  crypto = globalThis.crypto,
  baseUrl = globalThis.location?.href,
  databaseName = 'webscore-soundfonts-v1',
  cacheTimeoutMs = 2000,
} = {}) {
  const pending = new Map();
  if (!Number.isFinite(cacheTimeoutMs) || cacheTimeoutMs <= 0) throw new TypeError('Invalid cache timeout');
  const aborted = signal => { if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError'); };
  // Storage must never prevent playback (private mode, quota, blocked upgrade, broken IDB).
  function storage(mode, key, value) {
    if (!indexedDB) return Promise.resolve(undefined);
    return new Promise(resolve => {
      let db, done = false;
      const finish = result => { if (done) return; done = true; clearTimeout(timer); db?.close(); resolve(result); };
      const timer = setTimeout(() => finish(undefined), cacheTimeoutMs);
      try {
        const open = indexedDB.open(databaseName, 1);
        open.onupgradeneeded = () => { if (!open.result.objectStoreNames.contains('files')) open.result.createObjectStore('files'); };
        open.onerror = open.onblocked = () => finish(undefined);
        open.onsuccess = () => {
          db = open.result;
          if (done) { db.close(); return; }
          try {
            const tx = db.transaction('files', mode === 'get' ? 'readonly' : 'readwrite');
            const store = tx.objectStore('files');
            const request = mode === 'get' ? store.get(key) : mode === 'put' ? store.put(value, key) : store.delete(key);
            let result;
            request.onsuccess = () => { result = request.result; };
            tx.oncomplete = () => finish(result);
            tx.onerror = tx.onabort = () => finish(undefined);
          } catch { finish(undefined); }
        };
      } catch { finish(undefined); }
    });
  }
  async function valid(buffer, sha256, bytes) {
    if (!(buffer instanceof ArrayBuffer) || !buffer.byteLength || (bytes !== undefined && buffer.byteLength !== bytes)) return false;
    const digest = await crypto.subtle.digest('SHA-256', buffer);
    return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('') === sha256;
  }
  async function retrieve(url, sha256, bytes, signal, key) {
    aborted(signal);
    const hit = await storage('get', key);
    aborted(signal);
    if (hit !== undefined) {
      if (await valid(hit, sha256, bytes)) { aborted(signal); return hit; }
      await storage('delete', key);
    }
    aborted(signal);
    const response = await fetchImpl(url, {signal, credentials: 'omit'});
    if (!response.ok) throw new Error(`SoundFont download failed (${response.status})`);
    const buffer = await response.arrayBuffer();
    aborted(signal);
    if (!await valid(buffer, sha256, bytes)) throw new Error('SoundFont integrity check failed');
    aborted(signal);
    // Only complete, verified responses reach storage. An abort during storage can
    // leave a verified immutable entry, never a partial or unverified download.
    await storage('put', key, buffer);
    aborted(signal);
    return buffer;
  }
  return {
    async load({url, sha256, bytes, signal} = {}) {
      if (typeof url !== 'string' || !url.trim()) throw new TypeError('SoundFont URL is required');
      let resolved;
      try { resolved = new URL(url, baseUrl); } catch { throw new TypeError('Invalid SoundFont URL'); }
      if (!['https:', 'http:'].includes(resolved.protocol) || resolved.username || resolved.password) throw new TypeError('SoundFont URL must be HTTP(S) without credentials');
      resolved.hash = '';
      if (typeof sha256 !== 'string' || !/^[a-f0-9]{64}$/i.test(sha256)) throw new TypeError('SoundFont SHA-256 is required');
      if (bytes !== undefined && (!Number.isSafeInteger(bytes) || bytes <= 0)) throw new TypeError('Invalid SoundFont byte count');
      if (!crypto?.subtle?.digest) throw new Error('SHA-256 is unavailable; use a secure browser context');
      sha256 = sha256.toLowerCase();
      const key = JSON.stringify([resolved.href, sha256]);
      // Include requested length in the in-flight key so mismatched manifests
      // cannot inherit another caller's successful validation.
      const flightKey = JSON.stringify([key, bytes ?? null]);
      if (signal) return (await retrieve(resolved.href, sha256, bytes, signal, key)).slice(0);
      if (!pending.has(flightKey)) {
        const promise = retrieve(resolved.href, sha256, bytes, undefined, key);
        pending.set(flightKey, promise);
        promise.then(() => pending.delete(flightKey), () => pending.delete(flightKey));
      }
      // Each consumer owns a copy, safe for transfer to a synth worker.
      return (await pending.get(flightKey)).slice(0);
    },
  };
}
