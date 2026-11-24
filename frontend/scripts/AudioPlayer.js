// ==========================================================
// AudioPlayer
// - 사용자가 고른 원본 MP3의 재생·다운로드를 담당하는 class
// - Blob(Object) URL을 하나 만들어 <audio>와 다운로드 링크에 같이 사용
// - 재생 중에는 매 프레임 현재 시각을 onTimeUpdate로 알려 줌 (악보의 재생 위치 표시용)
// ==========================================================

export class AudioPlayer {
	constructor({ audio, downloadLink, onTimeUpdate }) {
		this.audio = audio;
		this.downloadLink = downloadLink;
		this.onTimeUpdate = onTimeUpdate;
		this.url = null;      // 지금 사용 중인 Object URL
		this.frameId = null;  // requestAnimationFrame 번호

		this.bindEvents();
	}

	get hasFile() {
		return this.url !== null;
	}

	// 새 파일을 고를 때마다 호출: 이전 URL은 해제하고 새 URL 생성
	setFile(file) {
		this.release();
		this.url = URL.createObjectURL(file);
		this.audio.src = this.url;
		this.downloadLink.href = this.url;
		this.downloadLink.download = file.name; // 다운로드 파일 이름 = 원래 이름
	}

	// 지정한 시각(초)으로 이동
	seek(seconds) {
		if (!this.hasFile) return;
		const duration = Number.isFinite(this.audio.duration) ? this.audio.duration : seconds;
		this.audio.currentTime = Math.min(Math.max(0, seconds), duration);
		this.notify();
	}

	// 파일 선택 해제·페이지 이탈 시: 재생 멈추고 URL 해제
	release() {
		this.stopLoop();
		this.audio.pause();
		this.audio.removeAttribute("src");
		this.audio.load(); // src를 비운 상태를 반영
		this.downloadLink.removeAttribute("href");
		if (this.url) {
			URL.revokeObjectURL(this.url);
			this.url = null;
		}
	}

	// ---------- 내부 동작 ----------
	bindEvents() {
		this.audio.addEventListener("play", () => this.startLoop());
		this.audio.addEventListener("pause", () => this.stopLoop());
		this.audio.addEventListener("ended", () => this.stopLoop());
		// 멈춘 상태에서 진행바를 옮겼을 때도 위치 갱신
		this.audio.addEventListener("seeked", () => this.notify());
	}

	// 재생 중에는 화면이 그려질 때마다(약 60번/초) 현재 시각 전달
	startLoop() {
		this.stopLoop();
		const tick = () => {
			this.notify();
			this.frameId = requestAnimationFrame(tick);
		};
		this.frameId = requestAnimationFrame(tick);
	}

	stopLoop() {
		if (this.frameId !== null) {
			cancelAnimationFrame(this.frameId);
			this.frameId = null;
		}
		this.notify(false); // 멈출 때는 위치만 갱신하고 스크롤은 그대로
	}

	// 두 번째 값: 재생 중이거나 사용자가 위치를 옮긴 경우에만 화면이 따라가도록
	notify(follow = true) {
		this.onTimeUpdate?.(this.audio.currentTime, follow);
	}
}
