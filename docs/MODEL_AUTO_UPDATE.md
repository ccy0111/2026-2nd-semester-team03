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
web app. An optional server-side signing adapter is provided below; wiring the host's authentication and deployment remains a host integration task.

## Update behavior

- Manifest requests (including JSON parsing) have a 15-second timeout by default;
  set `manifestTimeoutMs` to change it. Timed-out fetches are aborted and cannot
  block the queue indefinitely. This does not cancel ORT session loading.
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
live R2 CORS, authentication or live automatic rollout. Uploading browser ONNX files and configuring the manifest endpoint
are separate deployment steps; a successful code test does not prove a live rollout. No R2 public-access setting is changed here.

Run: `node --test webscore_inference/tests/test_model_versions.mjs`

## Optional private-R2 server adapter (disabled by default)

`backend/model_manifest.py` supplies an **opt-in** FastAPI router. The default app
still mounts no `/api/model-manifest` endpoint. It does not create a login bypass.
Install `requirements-model-manifest.txt` on the server if using this adapter.

Configure these **server-only** environment variables:

- `R2_MODEL_ACCOUNT_ID`, `R2_MODEL_BUCKET`
- `R2_MODEL_ACCESS_KEY_ID`, `R2_MODEL_SECRET_ACCESS_KEY` (bucket-scoped read only)
- `R2_MODEL_CATALOG_KEY` (default `web-student-kd/current.json`)
- `R2_MODEL_RELEASE_PREFIX` (default `web-student-kd/releases`)
- `R2_MODEL_URL_TTL_SECONDS` (default 3600, allowed 60–3600 seconds)

Never commit actual values, embed them in JavaScript, or log them. Do not enable
botocore wire/debug logging or response-body capture on this endpoint. Signed URLs
are temporary bearer access and must not appear in application logs/analytics.

Integration in the host's existing app factory, **after wiring its real auth**:

```python
from backend.model_manifest import (
    ManifestSettings, ModelManifestService, make_r2_client,
    create_model_manifest_router,
)

async def authorize_model_download(request):
    # Replace with the team's VERIFIED login/session + permission check.
    # Do not trust a client header, query parameter, or an unverified user ID.
    return False  # Fail closed until the host's auth integration exists.

settings = ManifestSettings.from_env()
service = ModelManifestService(settings, make_r2_client(settings))
app.include_router(create_model_manifest_router(service, authorize_model_download))
```

The hook must explicitly return boolean `True`; missing/false/error is denied.
The route reads one fixed catalog key from one configured bucket, bounds it to
64KiB, validates all five immutable release keys, and signs only GET requests.
Request parameters cannot choose a different bucket/key/version. All responses,
including denial and error, carry `Cache-Control: no-store, private`.

### Private catalog versus runtime manifest

The private R2 catalog stores **keys, sizes and SHA256**, not expiring URLs:

```json
{
  "schemaVersion": 1,
  "format": "webscore-onnx-v1",
  "version": "step-00036000-int8-v1",
  "precision": "int8",
  "files": {
    "config.json": {
      "key": "web-student-kd/releases/step-00036000-int8-v1/config.json",
      "size": 343,
      "sha256": "REPLACE_WITH_ACTUAL_64_HEX_SHA256"
    }
  }
}
```

The example is abbreviated: all five entries are mandatory. For each file,
`key` must exactly match `{release_prefix}/{version}/{filename}`. Publishing tools
must calculate real sizes/hashes and verify uploaded objects before publishing
`current.json`. The adapter validates metadata syntax, **not file bytes**; the
publisher's integrity verification remains required.

The server reads this private catalog directly via S3, then returns the existing
runtime manifest contract with five freshly signed URLs. Thus there is no stale
long-lived URL in the published catalog. The frontend keeps one stable,
same-origin `/api/model-manifest` URL and uses the wrapper described above.

Remaining host integration: mount the router with real authorization, provision
read-only server credentials, configure CORS for the actual website's origin, and
exclude the manifest endpoint from intermediary caching. No UI change or public
bucket is required. This PR does not perform those live environment changes.
