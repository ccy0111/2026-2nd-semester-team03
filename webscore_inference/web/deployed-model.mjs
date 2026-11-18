import { VersionedWebScoreClient } from './model-version-client.mjs';

// Public model metadata only. Never put R2 credentials or signed URLs here.
export const MODEL_MANIFEST_URL = 'https://models.midistudio.app/web-student-kd/current.json';
export const INITIAL_MODEL_VERSION = 'step-00036000-int8-v1';
export const INITIAL_MODEL_BASE_URL = 'https://models.midistudio.app/web-student-kd/releases/step-00036000-int8-v1/';

/** Production CDN model entry point, independent of the application UI. */
export function createDeployedModel(ort, options = {}) {
  return VersionedWebScoreClient.create(ort, {
    ...options,
    // The manifest selects the complete immutable release, not this initial URL.
    manifestUrl: MODEL_MANIFEST_URL,
    clientOptions: { executionProviders: ['wasm'], ...options.clientOptions },
  });
}
