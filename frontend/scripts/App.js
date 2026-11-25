// ==========================================================
// App
// - 화면 요소를 찾고, 버튼 이벤트를 연결하고, 각 기능 class를 이어 주는 class
//
// 필수 기능 (docs/MVP_REQUIREMENTS.md 프론트엔드 1~5)
//   1. MP3 선택 + 채보 시작
//   2. 모델 준비·채보 진행·완료·실패 상태 표시, 처리 중 중복 실행 방지
//   3. 추론 모듈 호출 → 음표를 SVG로 렌더링 (+ 실패/복구 구간 표시)
//   4. 원본 MP3 Blob URL 다운로드, 다 쓴 URL과 모델 세션 해제
//   5. 실패하면 파일 선택을 유지한 채 재시도
// + 채보 결과 재생 (docs/MVP_REQUIREMENTS.md "채보 결과 재생 연동"): 재생·위치·음소거
// + 내보내기: MIDI 다운로드(모델이 만든 midiBytes), 악보 PDF 저장(SheetView)
// ==========================================================
import { ModelService } from "./ModelService.js";
import { ScoreRenderer } from "./ScoreRenderer.js";
import { ProgressBar } from "./ProgressBar.js";
import { AudioPlayer } from "./AudioPlayer.js";
import { toUserMessage } from "./errorMessages.js";
import { InstrumentLegend } from "./InstrumentLegend.js";
import { SheetView } from "./SheetView.js";
import { NotePlayer } from "./NotePlayer.js";

export class App {
	constructor() {
		// 화면 요소
		this.audioInput = document.querySelector("#audio-input");
		this.startButton = document.querySelector("#start-button");
		this.cancelButton = document.querySelector("#cancel-button");
		this.retryButton = document.querySelector("#retry-button");
		this.normalizeLoudness = document.querySelector("#normalize-loudness");
		this.forcePrelude = document.querySelector("#force-prelude");
		this.statusText = document.querySelector("#status-text");
		this.playerArea = document.querySelector("#player-area");
		this.scoreLegend = document.querySelector("#score-legend");
		this.scoreSummary = document.querySelector("#score-summary");
		this.scoreView = document.querySelector("#score-view");
		this.viewTabs = document.querySelector("#view-tabs");
		this.rollArea = document.querySelector("#roll-area");
		this.sheetPanel = document.querySelector("#sheet-panel");
		this.notePlayerArea = document.querySelector("#note-player");
		this.notePlayButton = document.querySelector("#note-play");
		this.noteMuteButton = document.querySelector("#note-mute");
		this.noteTimeText = document.querySelector("#note-time");
		this.noteMessage = document.querySelector("#note-message");
		this.midiButton = document.querySelector("#midi-button");

		// 기능 담당 객체
		this.modelService = new ModelService();
		this.renderer = new ScoreRenderer(this.scoreView);
		this.progressBar = new ProgressBar(document.querySelector("#progress-bar"));
		this.instrumentLegend = new InstrumentLegend(
			document.querySelector("#instrument-legend"),
			(key, visible) => this.renderer.setInstrumentVisible(key, visible),
			(key, muted) => this.notePlayer.setInstrumentMuted(key, muted),
		);
		this.sheetView = new SheetView({
			instrumentSelect: document.querySelector("#sheet-instrument"),
			guitarModeSelect: document.querySelector("#sheet-guitar-mode"),
			guitarModeField: document.querySelector("#guitar-mode-field"),
			bpmInput: document.querySelector("#sheet-bpm"),
			meterSelect: document.querySelector("#sheet-meter"),
			audioArea: document.querySelector("#sheet-audio"),
			sheetArea: document.querySelector("#sheet-area"),
			messageText: document.querySelector("#sheet-message"),
			pdfButton: document.querySelector("#pdf-button"),
		});
		this.player = new AudioPlayer({
			audio: document.querySelector("#audio-player"),
			downloadLink: document.querySelector("#download-link"),
			onTimeUpdate: (seconds, follow) => {
				// 악기 소리 재생 중에는 커서를 그쪽이 움직임
				if (this.lastSource === "original") this.renderer.setCursor(seconds, follow);
			},
		});
		// 채보 결과를 악기 소리로 재생 (팀 재생 모듈)
		this.notePlayer = new NotePlayer({
			onTimeUpdate: (seconds, follow) => {
				if (this.lastSource === "notes") this.renderer.setCursor(seconds, follow);
				this.updateNoteTime();
			},
			onStateChange: () => this.updateNotePlayer(),
			onMessage: (text, type) => {
				this.noteMessage.textContent = text;
				this.noteMessage.className = `note-message ${type === "error" ? "text-error" : "text-sub"}`;
			},
		});

		// 피아노롤 클릭 → 마지막으로 쓴 재생기(원본 음원 / 악기 소리)로 그 위치부터 재생
		this.renderer.onSeek = (seconds) => {
			if (this.lastSource === "notes") {
				this.notePlayer.seek(seconds, true);
				return;
			}
			this.player.seek(seconds);
			// 브라우저가 재생을 막는 경우(파일 오류 등)에도 화면이 멈추지 않게
			this.player.audio.play().catch((error) => console.warn(error));
		};

		// 쉬고 있을 때 워커가 고장 나면 화면에 알리고 모델 재시도 안내
		// (채보 중 고장은 handleStart의 catch가 처리)
		this.modelService.onCrash = (message) => {
			if (this.isBusy) return;
			this.setStatus(message, "error");
			this.showModelRetry();
			this.updateButton();
		};

		// 상태
		this.isBusy = false;
		this.cancelRequested = false; // 사용자가 채보 취소를 눌렀는지
		this.retryAction = null; // 재시도 버튼이 실행할 함수 (실패 종류에 따라 다름)
		this.retryKind = null;   // "model"(모델 준비 실패) | "transcribe"(채보 실패)
		this.lastSource = "original"; // 피아노롤 커서·클릭이 따를 재생기: "original" | "notes"
		this.midiBytes = null;  // 마지막 채보 결과의 MIDI 파일 내용
		this.songTitle = "";    // MP3 파일 이름(확장자 제외) — 악보 제목, 저장 파일 이름
	}

	async init() {
		this.bindEvents();
		await this.loadModel();
	}

	bindEvents() {
		this.audioInput.addEventListener("change", () => this.handleFileChange());
		this.startButton.addEventListener("click", () => this.handleStart());
		this.retryButton.addEventListener("click", () => this.retryAction?.());
		this.cancelButton.addEventListener("click", () => this.handleCancel());
		// 악기 소리로 듣기: 원본 음원은 멈추고 같은 화면 커서를 이어받음
		this.notePlayButton.addEventListener("click", () => {
			if (!this.notePlayer.isPlaying) {
				this.player.audio.pause();
				this.useSource("notes");
			}
			this.notePlayer.toggle();
		});
		this.midiButton.addEventListener("click", () => this.downloadMidi());
		this.noteMuteButton.addEventListener("click", () => {
			this.notePlayer.setMuted(!this.notePlayer.isMuted);
		});
		// 원본 음원을 재생하면 악기 소리는 멈춤 (두 소리가 겹치지 않게)
		this.player.audio.addEventListener("play", () => {
			this.notePlayer.pause();
			this.useSource("original");
		});
		// 피아노롤 / 악보 탭 전환
		this.viewTabs.querySelectorAll(".view-tab").forEach((tab) => {
			tab.addEventListener("click", () => this.setView(tab.dataset.view));
		});
		// 페이지를 떠날 때 Object URL과 모델 세션 해제
		window.addEventListener("pagehide", () => {
			this.player.release();
			this.notePlayer.dispose();
			this.modelService.dispose();
		});
		// 뒤로/앞으로 가기로 돌아왔을 때(bfcache) 이미 해제된 상태이므로 새로 시작
		window.addEventListener("pageshow", (event) => {
			if (event.persisted) window.location.reload();
		});
	}

	// ---------- 모델 준비 ----------
	async loadModel() {
		this.hideRetry();
		this.setStatus("모델 준비 중... (처음에는 모델을 내려받느라 시간이 걸려요)", "info");
		this.updateButton();
		try {
			await this.modelService.load();
			const hasFile = this.audioInput.files.length > 0;
			this.setStatus(hasFile ? "모델 준비 완료. 채보 시작 버튼을 누르세요." : "모델 준비 완료. MP3를 선택하세요.", "success");
		} catch (error) {
			console.error(error);
			this.setStatus(`모델 준비 실패: ${toUserMessage(error)}`, "error");
			this.showModelRetry();
		}
		this.updateButton();
	}

	// ---------- 파일 선택 ----------
	handleFileChange() {
		const file = this.audioInput.files[0];
		// 채보 재시도 버튼은 이전 파일용이라 숨김 (모델 재시도 버튼은 그대로 둠)
		if (this.retryKind === "transcribe") this.hideRetry();
		this.resetResult();

		if (!file) {
			// 파일 선택 창에서 취소하면 선택이 비워짐 → 이전 결과 문구도 정리
			this.player.release();
			this.playerArea.classList.add("is-hidden");
			if (this.modelService.isReady) {
				this.setStatus("MP3를 선택하세요.", "info");
			}
		} else {
			// 원본 MP3 재생·다운로드는 채보 전에도 바로 사용 가능
			this.player.setFile(file);
			this.playerArea.classList.remove("is-hidden");
			if (this.modelService.isReady) {
				this.setStatus("채보 시작 버튼을 누르세요.", "info");
			}
		}
		this.updateButton();
	}

	// ---------- 채보 ----------
	async handleStart() {
		const file = this.audioInput.files[0];
		if (!file || this.isBusy || !this.modelService.isReady) return;

		this.isBusy = true;
		this.cancelRequested = false;
		this.startedAt = performance.now();
		this.hideRetry();
		this.resetResult();
		this.updateButton();
		this.cancelButton.classList.remove("is-hidden");
		this.renderer.clear("채보 중입니다...");
		this.setStatus("채보 중에는 이 탭을 닫지 마세요.", "info");
		this.progressBar.start("모델 버전 확인·음원 분석 준비 중...");

		let failedSoFar = 0;
		try {
			const result = await this.modelService.transcribe(file, (progress) => {
				if (progress.chunkStatus?.status === "failed") failedSoFar += 1;
				this.progressBar.update(progress, failedSoFar ? `(실패 구간 ${failedSoFar}개)` : "");
			}, {
				boundaryMode: this.forcePrelude?.checked ? "prelude-forcing" : "pitch-tie",
				normalizeLoudness: this.normalizeLoudness.checked,
			});
			this.showResult(result);
		} catch (error) {
			console.error(error);
			if (this.cancelRequested) {
				// 사용자가 취소한 경우: 실패로 표시하지 않음 (모델은 handleCancel에서 다시 준비 중)
				this.renderer.clear("채보를 취소했습니다.");
				this.progressBar.fail("채보 취소됨");
				return;
			}
			this.renderer.clear("채보에 실패했습니다.");
			this.progressBar.fail();
			this.setStatus(`채보 실패: ${toUserMessage(error)}`, "error");
			if (this.modelService.isReady) {
				// 파일 선택은 그대로 두고 같은 파일로 다시 시도
				this.showRetry("transcribe", "다시 채보하기", () => this.handleStart());
			} else {
				// 워커가 고장 나서 모델이 내려간 경우 → 모델부터 다시 준비
				this.showModelRetry();
			}
		} finally {
			this.isBusy = false;
			this.cancelButton.classList.add("is-hidden");
			this.updateButton();
		}
	}

	// 채보 취소: 모델 계산은 중간에 멈출 수 없어서, 워커를 끄고 모델을 새로 준비함
	handleCancel() {
		if (!this.isBusy || this.cancelRequested) return;
		// 채보 시작 직후 실수로 눌린 클릭은 무시 (더블클릭 등)
		if (performance.now() - this.startedAt < 800) return;
		this.cancelRequested = true;
		this.cancelButton.classList.add("is-hidden");
		this.loadModel(); // 내부의 terminate()가 진행 중인 채보를 중단시킴
	}

	showResult({ notes, chunkStatuses = [], midiBytes = null }) {
		const instruments = this.renderer.render(notes, chunkStatuses);
		this.instrumentLegend.render(instruments);
		// 노래 제목 = MP3 파일 이름에서 확장자(.mp3)를 뺀 것 (악보 제목에 사용)
		this.songTitle = (this.audioInput.files[0]?.name ?? "").replace(/\.[^.]+$/, "");
		this.sheetView.setResult(notes, instruments, this.songTitle);

		// MIDI 다운로드: 모델이 MIDI를 돌려줬을 때만 버튼 표시
		this.midiBytes = midiBytes instanceof Uint8Array && midiBytes.length > 0 ? midiBytes : null;
		this.midiButton.classList.toggle("is-hidden", !this.midiBytes);
		this.viewTabs.classList.remove("is-hidden");
		this.renderer.setCursor(this.player.audio.currentTime, false);

		// 악기 소리 재생: 피아노롤과 같은 규칙으로 거른 음표 사용
		this.notePlayer.setNotes(notes.filter((note) => ScoreRenderer.isValidNote(note)));
		this.noteMessage.textContent = "";
		this.notePlayerArea.classList.toggle("is-hidden", instruments.length === 0);
		this.updateNotePlayer();

		// 음높이 있는 음표와 드럼을 나눠서 셈 (드럼은 맨 아래 드럼 줄에 표시)
		const pitched = notes.filter((note) => !note.isDrum).length;
		const drums = notes.length - pitched;
		const failed = chunkStatuses.filter((chunk) => chunk.status === "failed").length;
		const recovered = chunkStatuses.filter((chunk) => chunk.status === "recovered").length;

		// 요약 문구
		const parts = [`음표 ${pitched}개`];
		if (drums) parts.push(`드럼·타악기 ${drums}개`);
		if (recovered) parts.push(`복구 구간 ${recovered}개`);
		if (failed) parts.push(`실패 구간 ${failed}개`);
		this.scoreSummary.textContent = parts.join(" · ");
		this.scoreLegend.classList.remove("is-hidden");

		// 일부 구간이 실패해도 결과는 보여 주되, 완전한 성공처럼 표시하지 않음
		if (pitched === 0 && drums > 0 && !failed) {
			// 타악기만 인식됨 → 피아노롤에 그릴 음표가 없음을 알려 줌
			this.progressBar.done();
			this.setStatus(`채보 완료! 드럼·타악기만 ${drums}개 인식됐어요. 맨 아래 드럼 줄을 확인하세요.`, "success");
		} else if (pitched === 0 && !failed) {
			// 끝까지 돌았지만 음표가 하나도 없음 (조용한 음원, 모델이 못 알아들음 등)
			this.progressBar.done("채보 완료 (음표 없음)");
			this.setStatus("채보는 끝났지만 인식된 음표가 없어요. 다른 음원으로 시도해 보세요.", "info");
		} else if (failed) {
			this.progressBar.done(`채보 완료 (실패 구간 ${failed}개)`);
			this.setStatus(`채보는 끝났지만 ${failed}개 구간을 인식하지 못했어요. 회색 구간을 확인하세요.`, "error");
			this.showRetry("transcribe", "다시 채보하기", () => this.handleStart());
		} else {
			this.progressBar.done();
			this.setStatus(`채보 완료! ${parts.join(" · ")}`, "success");
		}
	}

	// 이전 결과 지우기 (새 파일 선택 / 새 채보 시작)
	resetResult() {
		this.renderer.clear();
		this.instrumentLegend.clear();
		this.sheetView.clear();
		this.notePlayer.clear();
		this.midiBytes = null;
		this.useSource("original");
		this.noteMessage.textContent = "";
		this.notePlayerArea.classList.add("is-hidden");
		this.setView("roll");
		this.viewTabs.classList.add("is-hidden");
		this.progressBar.hide();
		this.scoreLegend.classList.add("is-hidden");
		this.scoreSummary.textContent = "";
	}

	// ---------- 보기 전환 ----------
	// view: "roll"(피아노롤) | "sheet"(악보)
	setView(view) {
		this.viewTabs.querySelectorAll(".view-tab").forEach((tab) => {
			const active = tab.dataset.view === view;
			tab.classList.toggle("is-active", active);
			tab.setAttribute("aria-selected", String(active));
		});
		this.rollArea.classList.toggle("is-hidden", view !== "roll");
		this.sheetPanel.classList.toggle("is-hidden", view !== "sheet");
		if (view === "sheet") {
			this.notePlayer.pause(); // 악보 탭은 자기 재생기를 씀
			this.sheetView.show();
		} else {
			this.sheetView.hide();
		}
	}

	// 채보 페이지를 떠날 때: 소리 모두 멈춤 (결과는 그대로)
	pauseAll() {
		this.player.audio.pause();
		this.notePlayer.pause();
		this.sheetView.stopSynth();
	}

	// ---------- 내보내기 ----------
	// 모델이 만든 MIDI를 파일로 저장 (Blob URL을 잠깐 만들었다가 해제)
	downloadMidi() {
		if (!this.midiBytes) return;
		const blob = new Blob([this.midiBytes], { type: "audio/midi" });
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = `${this.songTitle || "채보결과"}.mid`;
		document.body.appendChild(link);
		link.click();
		link.remove();
		// 다운로드가 시작될 시간을 준 뒤 해제
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}

	// ---------- 악기 소리 재생 표시 ----------
	// 피아노롤 커서가 따를 재생기를 바꾸고 커서를 그 위치로 옮김
	useSource(source) {
		if (this.lastSource === source) return;
		this.lastSource = source;
		const seconds = source === "notes" ? this.notePlayer.currentTime : this.player.audio.currentTime;
		this.renderer.setCursor(seconds, false);
	}

	updateNotePlayer() {
		const notePlayer = this.notePlayer;
		if (notePlayer.isPreparing) {
			this.notePlayButton.textContent = "소리 준비 중...";
		} else {
			this.notePlayButton.textContent = notePlayer.isPlaying ? "⏸ 일시정지" : "▶ 악기 소리로 듣기";
		}
		this.notePlayButton.disabled = notePlayer.isPreparing;
		this.noteMuteButton.textContent = notePlayer.isMuted ? "음소거 해제" : "음소거";
		this.noteMuteButton.setAttribute("aria-pressed", String(notePlayer.isMuted));
		this.updateNoteTime();
	}

	updateNoteTime() {
		const format = (seconds) => {
			const total = Math.max(0, Math.floor(seconds));
			return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
		};
		this.noteTimeText.textContent = `${format(this.notePlayer.currentTime)} / ${format(this.notePlayer.duration)}`;
	}

	// ---------- 버튼·상태 표시 ----------
	// 모델이 준비됐고, 파일이 있고, 처리 중이 아닐 때만 버튼 활성화
	updateButton() {
		const hasFile = this.audioInput.files.length > 0;
		this.startButton.disabled = !this.modelService.isReady || !hasFile || this.isBusy;
		this.audioInput.disabled = this.isBusy;
		this.normalizeLoudness.disabled = this.isBusy;
		if (this.forcePrelude) this.forcePrelude.disabled = this.isBusy;
	}

	showRetry(kind, label, action) {
		this.retryKind = kind;
		this.retryAction = action;
		this.retryButton.textContent = label;
		this.retryButton.classList.remove("is-hidden");
	}

	showModelRetry() {
		this.showRetry("model", "모델 다시 불러오기", () => this.loadModel());
	}

	hideRetry() {
		this.retryKind = null;
		this.retryAction = null;
		this.retryButton.classList.add("is-hidden");
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
