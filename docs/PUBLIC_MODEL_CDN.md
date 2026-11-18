# Public model CDN configuration

The public model entry point is `webscore_inference/web/deployed-model.mjs`.
This is model-only configuration; it does not change the team's UI.

```js
import * as ort from 'onnxruntime-web';
import { createDeployedModel } from './webscore_inference/web/deployed-model.mjs';
const model = await createDeployedModel(ort);
const { notes, midiBytes } = await model.transcribeAudio(audioFile);
// When the app no longer needs the model:
await model.dispose();
```

- Current-version manifest: https://models.midistudio.app/web-student-kd/current.json
- Initial release: https://models.midistudio.app/web-student-kd/releases/step-00036000-int8-v1/
- Precision: INT8 decoder, existing float32 input/output contract. No FP16 rollout.
- Default execution provider: WASM. Real-browser WebGPU remains unverified.
- The manifest controls later version selection. The initial release constant is
  informational, not a pin that disables automatic updates.
- Manifest fetches use `no-store`; the wrapper checks before inference and polls
  every 60 seconds. A failed replacement preserves the previous loaded model.
- The models are now intentionally public at this CDN domain. No user login,
  R2 key or signed URL is required by this public path. Do not add credentials
  to browser code. The separate private-R2 adapter remains opt-in and disabled
  in the default FastAPI app; this public entry point does not use it.

## Deployment verification and limits

The deployed manifest returned HTTP 200, `Cache-Control: no-store`, dynamic cache
status and wildcard CORS. The five release files were checked against their
published hashes; the large decoder also returned a CDN cache HIT. These checks
cover serving bytes, not transcription quality. Browser CORS headers are present,
but full real-browser WebAudio/WebGPU behavior is not established by them.

The existing real-model Node WASM tests use synthesized audio. Wrong instrument
classification and a missing note around a chunk boundary remain known model
limitations. The default factory regression checks the actual wrapper/configuration
with mocked downloads and ORT sessions. It is not itself a live-CDN inference test.

Publish a complete new immutable release before replacing `current.json`; never
cache the current-version manifest or overwrite an already published release.
See [MODEL_AUTO_UPDATE.md](MODEL_AUTO_UPDATE.md) for update/rollback semantics.
