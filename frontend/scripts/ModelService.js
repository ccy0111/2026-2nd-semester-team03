// ==========================================================
// ModelService
// - 모델을 Web Worker(model.worker.js)에서 돌리고, 결과를 Promise로 돌려주는 class
// - App.js는 load() / transcribe() / dispose()만 알면 됨 (워커 존재를 몰라도 됨)
//
// 역할 분리
//   메인 스레드: MP3 → 16kHz 모노 음원으로 디코딩 (AudioContext는 워커에서 못 씀)
//   워커 스레드: 모델 다운로드, 전처리, 추론 (무거운 계산)
// ==========================================================

const WORKER_URL = "/static/scripts/model.worker.js";
const MEL_MODULE_URL = "/inference/mel-frontend.mjs";
const SAMPLE_RATE = 16000;

// ---------- 개발용 설정 (발표·배포 전에 null로 되돌리기) ----------
// 앞부분 몇 초만 채보 (예: 20). null이면 전체 곡
export const DEV_MAX_SECONDS = 20;
// 구간당 토큰 한도 실험 (예: 2048). null이면 모델 기본값(1024)
export const DEV_MAX_GEN_LEN = null;

export class ModelService {
	constructor() {
		this.worker = null;
		this.ready = false;
		this.pending = null; // 지금 기다리는 작업 { resolve, reject, onProgress }
	}

	get isReady() {
		return this.ready;
	}

	// 모델 준비 (화면이 열릴 때 한 번, 실패하면 다시 호출 가능)
	load() {
		this.terminate();
		this.worker = new Worker(WORKER_URL);
		this.worker.addEventListener("message", (event) => this.handleMessage(event.data));
		// 워커 파일 자체를 못 불러온 경우
		this.worker.addEventListener("error", (event) => {
			event.preventDefault();
			this.finish("reject", new Error("모델 워커를 시작하지 못했습니다."));
		});

		return this.request({ type: "load" }).then(() => {
			this.ready = true;
		});
	}

	// MP3 File → { notes }
	async transcribe(file, onProgress) {
		if (!this.isReady) {
			throw new Error("모델이 아직 준비되지 않았습니다.");
		}
		const { decodeAudioTo16kMono } = await import(MEL_MODULE_URL);
		let pcm = await decodeAudioTo16kMono(file);
		if (DEV_MAX_SECONDS) {
			pcm = pcm.slice(0, DEV_MAX_SECONDS * SAMPLE_RATE);
		}

		// pcm.buffer를 복사하지 않고 워커로 넘김(transfer) → 큰 음원도 빠르게 전달
		const notes = await this.request(
			{ type: "transcribe", pcm, maxGenLen: DEV_MAX_GEN_LEN },
			onProgress,
			[pcm.buffer],
		);
		return { notes };
	}

	// 화면을 떠날 때 워커와 모델 정리
	dispose() {
		this.terminate();
	}

	// ---------- 내부 동작 ----------
	terminate() {
		if (this.worker) {
			this.worker.terminate(); // 워커를 끝내면 안의 모델 세션도 함께 해제됨
			this.worker = null;
		}
		this.ready = false;
		this.finish("reject", new Error("모델 작업이 중단되었습니다."));
	}

	// 워커에 메시지를 보내고, 답이 올 때까지 기다리는 Promise
	request(message, onProgress = null, transfer = []) {
		return new Promise((resolve, reject) => {
			this.pending = { resolve, reject, onProgress };
			this.worker.postMessage(message, transfer);
		});
	}

	handleMessage(data) {
		if (data.type === "progress") {
			this.pending?.onProgress?.(data.progress);
		} else if (data.type === "loaded") {
			this.finish("resolve");
		} else if (data.type === "result") {
			this.finish("resolve", data.notes);
		} else if (data.type === "error") {
			this.finish("reject", new Error(data.message));
		}
	}

	finish(kind, value) {
		const pending = this.pending;
		this.pending = null;
		pending?.[kind](value);
	}
}
