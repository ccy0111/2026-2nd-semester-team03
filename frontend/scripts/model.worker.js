// ==========================================================
// model.worker.js (Web Worker)
// - 무거운 모델 계산을 화면(메인 스레드)과 분리된 별도 스레드에서 실행
// - 화면이 멈추지 않아서 로딩바가 계속 움직이고 "응답 없음"이 뜨지 않음
//
// 메인 → 워커 메시지
//   { type: "load" }
//   { type: "transcribe", pcm: Float32Array, maxGenLen, boundaryMode, allowedPrograms, normalizeLoudness }
// 워커 → 메인 메시지
//   { type: "loaded" }
//   { type: "progress", progress: { completedChunks, totalChunks, chunkStatus } }
//   { type: "result", notes, chunkStatuses }
//   { type: "error", message }
// ==========================================================

// Keep the JS loader and WASM assets pinned together. 1.31.1 fixes V2 encoder NaNs in 1.20.1.
const ORT_VERSION = "1.31.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_MODULE_URL = "/inference/model-version-client.mjs";

let model = null;
let loadPromise = null;
let modelRuntime = null;

self.addEventListener("message", async (event) => {
	const { type } = event.data;
	try {
		if (type === "load") {
			await loadModel();
			self.postMessage({ type: "loaded", runtime: modelRuntime });
		}
		if (type === "transcribe") {
			const result = await transcribe(event.data.pcm, event.data.maxGenLen, event.data.boundaryMode, event.data.allowedPrograms, event.data.normalizeLoudness);
			self.postMessage({ type: "result", ...result });
		}
	} catch (error) {
		console.error(error);
		self.postMessage({ type: "error", message: error.message });
	}
});

async function createForRuntime(VersionedWebScoreClient, provider, adapter) {
	// Keep the runtime and artifact paired. Never try to run the FP16 graph on WASM.
	importScripts(`${ORT_BASE}${provider === "webgpu" ? "ort.webgpu.min.js" : "ort.min.js"}`);
	const runtime = globalThis.ort;
	runtime.env.wasm.wasmPaths = ORT_BASE;
	if (provider === "wasm") {
		const { cpuWasmThreadCount } = await import("/inference/cpu-wasm-policy.mjs");
		runtime.env.wasm.numThreads = cpuWasmThreadCount(globalThis);
	} else {
		runtime.env.wasm.numThreads = 1;
		runtime.env.webgpu.adapter = adapter;
	}
	const create = (options = {}) => VersionedWebScoreClient.create(runtime, {
		manifestVariant: provider,
		clientOptions: { executionProviders: [provider], ...options },
	});
	if (provider === "webgpu") {
		// The B4+K4 path was validated on this architecture, and the client
		// additionally requires the exact graph and pitch-tie mode.
		const wide = adapter?.info?.architecture === "rdna-3";
		try {
			return await create({ gpuKvCache: true, cacheEmbeddings: true, gpuTokenSelection: true,
				gpuBatchPairs: wide, gpuBatchWidth: wide ? 4 : 2, gpuFeedbackBlock: wide ? 4 : 0 });
		} catch {
			console.warn("[WebScore] GPU token selection unavailable; trying standard selection");
		}
		try {
			return await create({ gpuKvCache: true, cacheEmbeddings: true });
		} catch {
			console.warn("[WebScore] GPU KV cache unavailable; trying standard WebGPU");
		}
	}
	return create();
}

function loadModel() {
	if (!loadPromise) loadPromise = (async () => {
		const { VersionedWebScoreClient } = await import(MODEL_MODULE_URL);
		let fallbackReason = "WebGPU unavailable";
		try {
			const adapter = await globalThis.navigator?.gpu?.requestAdapter({ powerPreference: "high-performance" });
			if (adapter?.info?.isFallbackAdapter) {
				fallbackReason = "Software WebGPU adapter";
			} else if (adapter && !adapter.features.has("shader-f16")) {
				fallbackReason = "Missing shader-f16";
			} else if (adapter) {
				fallbackReason = "WebGPU FP16 initialization failed";
				model = await createForRuntime(VersionedWebScoreClient, "webgpu", adapter);
				modelRuntime = { provider: "webgpu", precision: "fp16",
					gpuKvCache: model.client?.gpuKvCache === true,
					gpuTokenSelection: Boolean(model.client?.tokenSelector),
					gpuBatchWidth: model.client?.gpuBatchPairs ? model.client.gpuBatchWidth : 1 };
				return;
			}
		} catch {
			// Provider/network exceptions may contain a signed URL; keep logs generic.
			console.warn("[WebScore] WebGPU initialization did not complete; using WASM INT8");
		}
		model = await createForRuntime(VersionedWebScoreClient, "wasm", null);
		modelRuntime = { provider: "wasm", precision: "int8", fallbackReason, numThreads: model.ort.env.wasm.numThreads };
	})().catch(error => { loadPromise = null; throw error; });
	return loadPromise;
}

async function transcribe(pcm, maxGenLen, boundaryMode, allowedPrograms, normalizeLoudness) {
	if (!model) {
		throw new Error("모델이 아직 준비되지 않았습니다.");
	}
	const options = {
		onProgress: (progress) => {
			self.postMessage({ type: "progress", progress });
		},
	};
	// 값이 있을 때만 넘김 (없으면 모델 기본값 사용)
	if (maxGenLen) {
		options.maxGenLen = maxGenLen;
	}
	if (boundaryMode != null) { // undefined·null이면 모델 기본값(pitch-tie)
		options.boundaryMode = boundaryMode;
	}
	if (allowedPrograms !== undefined) options.allowedPrograms = allowedPrograms;
	if (normalizeLoudness === true) {
		const { normalizeInferenceLoudness } = await import("./inference-loudness.mjs");
		pcm = normalizeInferenceLoudness(pcm);
	}
	const result = await model.transcribePcm(pcm, options);
	return { notes: result.notes, chunkStatuses: result.chunkStatuses };
}
