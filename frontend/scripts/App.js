// ==========================================================
// App
// - 화면 요소를 찾고, 버튼 이벤트를 연결하고, 모델과 렌더러를 이어 주는 class
// ==========================================================
import { ModelService } from "./ModelService.js";
import { ScoreRenderer } from "./ScoreRenderer.js";
import { ProgressBar } from "./ProgressBar.js";

export class App {
	constructor() {
		// 화면 요소
		this.audioInput = document.querySelector("#audio-input");
		this.startButton = document.querySelector("#start-button");
		this.statusText = document.querySelector("#status-text");
		this.scoreView = document.querySelector("#score-view");

		// 기능 담당 객체
		this.modelService = new ModelService();
		this.renderer = new ScoreRenderer(this.scoreView);
		this.progressBar = new ProgressBar(document.querySelector("#progress-bar"));

		// 상태
		this.isBusy = false;
	}

	async init() {
		this.bindEvents();
		await this.loadModel();
	}

	bindEvents() {
		this.audioInput.addEventListener("change", () => this.updateButton());
		this.startButton.addEventListener("click", () => this.handleStart());
		// 페이지를 떠날 때 모델 세션 해제
		window.addEventListener("pagehide", () => this.modelService.dispose());
	}

	async loadModel() {
		this.setStatus("모델 준비 중... (처음에는 모델을 내려받느라 시간이 걸려요)", "info");
		try {
			await this.modelService.load();
			this.setStatus("모델 준비 완료. MP3를 선택하세요.", "success");
		} catch (error) {
			console.error(error);
			this.setStatus(`모델 준비 실패: ${error.message}`, "error");
			// TODO(지윤): 모델 준비 재시도 버튼
		}
		this.updateButton();
	}

	async handleStart() {
		const file = this.audioInput.files[0];
		if (!file || this.isBusy) return;

		this.isBusy = true;
		this.updateButton();
		this.renderer.clear("채보 중입니다...");
		this.setStatus("채보 중에는 이 탭을 닫지 마세요.", "info");
		this.progressBar.start();

		try {
			const result = await this.modelService.transcribe(file, (progress) => {
				this.progressBar.update(progress);
			});
			this.renderer.render(result.notes);
			this.progressBar.done();
			this.setStatus(`채보 완료! 음표 ${result.notes.length}개`, "success");
			console.log("notes", result.notes); // 개발 중 확인용
		} catch (error) {
			console.error(error);
			this.renderer.clear("채보에 실패했습니다.");
			this.progressBar.fail();
			this.setStatus(`채보 실패: ${error.message}`, "error");
			// TODO(지윤): 파일 선택은 유지한 채 재시도할 수 있게 안내
		} finally {
			this.isBusy = false;
			this.updateButton();
		}

		// TODO(지윤): 원본 MP3 다운로드 (URL.createObjectURL), 다 쓰면 URL.revokeObjectURL
		// TODO(지윤): <audio>로 재생 + this.renderer.timeToX(currentTime)로 현재 위치 세로선 표시
	}

	// 모델이 준비됐고, 파일이 있고, 처리 중이 아닐 때만 버튼 활성화
	updateButton() {
		const hasFile = this.audioInput.files.length > 0;
		this.startButton.disabled = !this.modelService.isReady || !hasFile || this.isBusy;
		this.audioInput.disabled = this.isBusy;
	}

	// type: info | success | error  →  root.css의 글자색 class로 연결
	setStatus(message, type = "info") {
		const colorClass = {
			info: "text-sub",
			success: "text-success",
			error: "text-error",
		}[type];
		this.statusText.textContent = message;
		this.statusText.className = `status-text ${colorClass}`;
	}
}
