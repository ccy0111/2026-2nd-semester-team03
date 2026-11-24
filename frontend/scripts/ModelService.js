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
export const DEV_MAX_SECONDS = null;
// 구간당 토큰 한도 실험 (예: 2048). null이면 모델 기본값(1024)
export const DEV_MAX_GEN_LEN = null;

export class ModelService {
	constructor() {
		this.worker = null;
		this.ready = false;
		this.pending = null; // 지금 기다리는 작업 { resolve, reject, onProgress }
		this.onCrash = null;  // 쉬는 중에 워커가 고장 났을 때 알릴 함수 (App에서 연결)
	}

	get isReady() {
		return this.ready;
	}

	// 모델 준비 (화면이 열릴 때 한 번, 실패하면 다시 호출 가능)
	load() {
		this.terminate();
		const worker = new Worker(WORKER_URL);
		this.worker = worker;

		// 예전 워커가 늦게 보낸 메시지·오류는 무시 (worker !== this.worker)
		worker.addEventListener("message", (event) => {
			if (worker !== this.worker) return;
			this.handleMessage(event.data);
		});
		// 워커 파일을 못 불러왔거나, 실행 중에 워커가 고장 난 경우
		worker.addEventListener("error", (event) => {
			event.preventDefault();
			if (worker !== this.worker) return;
			const wasReady = this.ready;
			const message = wasReady
				? "모델 실행 중 오류가 발생했습니다. 모델을 다시 불러와 주세요."
				: "모델 워커를 시작하지 못했습니다.";
			this.finish("reject", new Error(message));
			// 고장 난 워커는 버림 → isReady가 false가 되어, 다시 쓰려면 load()부터
			this.terminate();
			if (wasReady) this.onCrash?.(message);
		});
		// 주고받는 데이터를 복원하지 못한 경우 (드묾)
		worker.addEventListener("messageerror", () => {
			if (worker !== this.worker) return;
			this.finish("reject", new Error("모델과 데이터를 주고받지 못했습니다."));
		});

		return this.request({ type: "load" }).then(() => {
			this.ready = true;
		});
	}

	// MP3 File → { notes, chunkStatuses }
	async transcribe(file, onProgress, { boundaryMode, allowedPrograms } = {}) {
		if (!this.isReady) {
			throw new Error("모델이 아직 준비되지 않았습니다.");
		}
		const worker = this.worker; // 디코딩하는 동안 워커가 바뀌었는지 확인용
		const { decodeAudioTo16kMono } = await import(MEL_MODULE_URL);
		let pcm;
		try {
			pcm = await decodeAudioTo16kMono(file);
		} catch (error) {
			console.error(error);
			throw new Error("음원 파일을 읽지 못했습니다. 손상되었거나 지원하지 않는 형식일 수 있어요.");
		}
		if (DEV_MAX_SECONDS) {
			pcm = pcm.slice(0, DEV_MAX_SECONDS * SAMPLE_RATE);
		}
		// 디코딩 도중 취소·고장으로 워커가 바뀌었으면 보내지 않음
		// (새 워커가 모델을 준비하는 중이라 이 요청을 받으면 안 됨)
		if (worker !== this.worker || !this.ready) {
			throw new Error("채보가 중단되었습니다.");
		}

		// pcm.buffer를 복사하지 않고 워커로 넘김(transfer) → 큰 음원도 빠르게 전달
		return this.request(
			{ type: "transcribe", pcm, maxGenLen: DEV_MAX_GEN_LEN, boundaryMode, allowedPrograms },
			onProgress,
			[pcm.buffer],
		);
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
			// 워커가 이미 정리된 경우(페이지 이탈 등) 예외 대신 실패로 처리
			if (!this.worker) {
				reject(new Error("모델이 준비되지 않았습니다."));
				return;
			}
			// 이전 작업이 남아 있으면 먼저 실패 처리 (평소엔 일어나지 않음)
			if (this.pending) this.finish("reject", new Error("이전 작업이 중단되었습니다."));
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
			this.finish("resolve", { notes: data.notes, chunkStatuses: data.chunkStatuses });
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
