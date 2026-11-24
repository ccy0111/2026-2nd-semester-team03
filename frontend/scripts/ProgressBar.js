// ==========================================================
// ProgressBar
// - 채보 진행률을 로딩바로 보여 주는 class
// - 구간(chunk) 하나가 끝날 때마다 update()로 채워짐
// ==========================================================

export class ProgressBar {
	constructor(container) {
		this.container = container;
		this.fill = container.querySelector(".progress-fill");
		this.label = container.querySelector(".progress-label");
		this.percent = container.querySelector(".progress-percent");
	}

	// 처음 시작할 때: 0%로 보여 주기
	start(message = "음원 분석 준비 중...") {
		this.container.classList.remove("is-hidden", "is-done", "is-error");
		this.container.classList.add("is-running");
		// 이전 100%에서 0%로 미끄러지는 애니메이션 없이 바로 0%로
		this.fill.style.transition = "none";
		this.setValue(0);
		void this.fill.offsetWidth; // 브라우저에 지금 상태를 먼저 반영시킴
		this.fill.style.transition = "";
		this.label.textContent = message;
	}

	// 모델의 onProgress 값 그대로 받기: { completedChunks, totalChunks }
	// extraText: 뒤에 덧붙일 문구 (예: "(실패 구간 1개)")
	update({ completedChunks, totalChunks }, extraText = "") {
		const ratio = totalChunks > 0 ? completedChunks / totalChunks : 0;
		this.setValue(ratio);
		this.label.textContent = `채보 중... ${completedChunks} / ${totalChunks} 구간 ${extraText}`.trim();
	}

	// 성공
	done(message = "채보 완료!") {
		this.container.classList.remove("is-running");
		this.container.classList.add("is-done");
		this.setValue(1);
		this.label.textContent = message;
	}

	// 실패: 멈춘 위치 그대로 두고 빨간색으로
	fail(message = "채보 실패") {
		this.container.classList.remove("is-running");
		this.container.classList.add("is-error");
		this.label.textContent = message;
	}

	hide() {
		this.container.classList.add("is-hidden");
	}

	// 0 ~ 1 사이 값을 막대 너비와 퍼센트 글자로
	setValue(ratio) {
		const percent = Math.round(Math.min(1, Math.max(0, ratio)) * 100);
		this.fill.style.width = `${percent}%`;
		this.percent.textContent = `${percent}%`;
		this.container.setAttribute("aria-valuenow", percent);
	}
}
