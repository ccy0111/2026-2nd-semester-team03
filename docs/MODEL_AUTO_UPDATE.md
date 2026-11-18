# Model-only version updates

`webscore_inference/web/model-version-client.mjs` adds a model wrapper, not a UI,
login API or live deployment. The existing direct `WebScoreClient` remains usable.

```js
import * as ort from 'onnxruntime-web'; // WASM example; WebGPU needs /webgpu build
import { VersionedWebScoreClient } from './web/model-version-client.mjs';
const model = await VersionedWebScoreClient.create(ort, {
  manifestUrl: '/api/model-manifest',
  clientOptions: { executionProviders: ['wasm'] },
  pollIntervalMs: 60000,
  checkBeforeInference: true,
  onUpdate: ({ version }) => { /* optional model-version display */ },
  onUpdateError: () => { /* optional retry indicator; no raw URLs */ },
});
const result = await model.transcribeAudio(audioFile);
// On application teardown:
await model.dispose();
```

## Manifest and publication contract

See `webscore_inference/examples/model-manifest.json`. Required format is
`webscore-onnx-v1`, `schemaVersion: 1`, immutable version ID, precision `int8` or
`fp32`, and a complete set of file URLs. FP32 uses `decoder_kv.onnx`. Relative URLs
resolve against the manifest endpoint; private deployments should return explicit
HTTPS signed URLs for all five files. The same version may renew its signed URLs
without forcing an already loaded model to download again. Never put a secret
access key or account API token in browser code or the manifest.

Publisher steps:
1. Export and validate the entire model set into a **new immutable version path**.
2. Upload every required file. Check object sizes/hashes and run compatibility and
   inference validation. Do not overwrite a published version's files.
3. Only then replace the small current-version manifest (or its authenticated API
   response) in one operation. Its `version` must change for any artifact change.
4. Serve manifest responses with `Cache-Control: no-store`; configure CDN/cache
   rules not to cache that endpoint. The client also requests `cache: no-store`,
   which does not itself override a misconfigured CDN. Model objects may be cached
   long-term at version-specific URLs, subject to the deployment's access policy.
5. Retain the preceding version for rollback, in-flight clients and expiring URLs.
   To roll back, publish its previous manifest as current. Avoid deleting old
   artifacts while existing users can still need them.

The publisher must supply one compatible set; the client checks manifest format,
metadata (16kHz, FFT2048, mel512, frame rate100, GQA dimensions, vocabulary and special
token IDs), session IO names and load success. This is **not** a cryptographic
verification or a proof of transcription quality. Only trust a controlled manifest
endpoint. Private manifest APIs must authenticate the user, sign URLs for the
selected version, ensure adequate URL lifetime, and configure object CORS for the
web app. That server integration is outside this model-only change.

## Update behavior

- Checks once during creation, every 60 seconds by default, and before inference.
  Set polling to `0` to disable background checks; manual `checkForUpdate()` is also
  available. This is polling, not instant server push.
- Inference and checks are serialized. An entire song finishes with one model,
  including all chunks, before a pending replacement is loaded and activated.
- All new sessions must load and pass interface checks before an atomic client
  swap; old sessions are released afterward. Updates may temporarily require
  memory for **both** versions. If replacement cannot load, the old model remains.
- Malformed/incompatible manifest, expired links or network/load failure preserve
  the old version; initial load failure rejects because no old version exists.
- `dispose()` stops polling, waits for work already queued, and releases sessions.
- Errors passed to callbacks omit URLs/provider causes. Browser developer tools
  and the runtime itself can still expose signed URLs to the authorized browser.

## Validation limits

The updater tests use controlled clients to cover version changes, failed loads,
metadata mismatch, session cleanup, active-inference ordering, polling, rollback,
and signed-URL redaction. They do not establish real-browser WebGPU compatibility,
live R2 CORS, authentication or live automatic rollout. R2 currently contains the
raw checkpoint; uploading browser ONNX files and configuring the manifest endpoint
are separate deployment steps. No R2 public-access setting is changed here.

Run: `node --test webscore_inference/tests/test_model_versions.mjs`
