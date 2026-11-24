// ==========================================================
// model.worker.js (Web Worker)
// - 무거운 모델 계산을 화면(메인 스레드)과 분리된 별도 스레드에서 실행
// - 화면이 멈추지 않아서 로딩바가 계속 움직이고 "응답 없음"이 뜨지 않음
//
// 메인 → 워커 메시지
//   { type: "load" }
//   { type: "transcribe", pcm: Float32Array, maxGenLen, boundaryMode }
// 워커 → 메인 메시지
//   { type: "loaded" }
//   { type: "progress", progress: { completedChunks, totalChunks } }
//   { type: "result", notes, chunkStatuses }
//   { type: "error", message }
// ==========================================================

// Keep the JS loader and WASM assets pinned together. 1.31.1 fixes V2 encoder NaNs in 1.20.1.
const ORT_VERSION = "1.31.1";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_MODULE_URL = "/inference/model-version-client.mjs";

// 워커 안에서는 <script> 대신 importScripts로 라이브러리를 불러옴 → 전역 ort 생성
importScripts(`${ORT_BASE}ort.min.js`);
ort.env.wasm.wasmPaths = ORT_BASE;

let model = null;

self.addEventListener("message", async (event) => {
	const { type } = event.data;
	try {
		if (type === "load") {
			await loadModel();
			self.postMessage({ type: "loaded" });
		}
		if (type === "transcribe") {
			const result = await transcribe(event.data.pcm, event.data.maxGenLen, event.data.boundaryMode);
			self.postMessage({ type: "result", ...result });
		}
	} catch (error) {
		console.error(error);
		self.postMessage({ type: "error", message: error.message });
	}
});

async function loadModel() {
	const { VersionedWebScoreClient } = await import(MODEL_MODULE_URL);
	model = await VersionedWebScoreClient.create(ort, {
		clientOptions: { executionProviders: ["wasm"] },
	});
}

async function transcribe(pcm, maxGenLen, boundaryMode) {
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
	if (boundaryMode !== undefined) {
		options.boundaryMode = boundaryMode;
	}
	const result = await model.transcribePcm(pcm, options);
	return { notes: result.notes, chunkStatuses: result.chunkStatuses };
}
