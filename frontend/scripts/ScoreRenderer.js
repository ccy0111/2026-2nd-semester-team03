// ==========================================================
// ScoreRenderer
// - 모델이 돌려준 notes 배열을 피아노롤(SVG)로 그리는 class
// - 가로축: 시간(초) → x좌표, 세로축: 음높이(MIDI pitch) → y좌표
//
// note 형식 (docs/API.md)
//   { program, pitch, onset, offset, isDrum }
//   pitch: 0~127 (60 = 가운데 도, C4)
//   onset, offset: 원본 음원 기준 시작/끝 시각(초)
// ==========================================================

const SVG_NS = "http://www.w3.org/2000/svg";
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK_KEYS = [1, 3, 6, 8, 10]; // 한 옥타브 안에서 검은 건반 위치

export class ScoreRenderer {
	constructor(container, options = {}) {
		this.container = container;
		this.pxPerSecond = options.pxPerSecond ?? 80; // 1초 = 80px
		this.rowHeight = options.rowHeight ?? 10;     // 반음 하나 = 10px
		this.labelWidth = options.labelWidth ?? 40;   // 왼쪽 음이름 칸
		this.svg = null;
	}

	// 결과 지우고 안내 문구 표시
	clear(message = "채보가 끝나면 여기에 음표가 표시됩니다.") {
		this.container.innerHTML = "";
		const p = document.createElement("p");
		p.className = "text-sub";
		p.textContent = message;
		this.container.appendChild(p);
		this.svg = null;
	}

	render(notes) {
		// 지금은 타악기를 제외하고 음높이가 있는 음표만 그림
		const melodic = notes.filter((note) => !note.isDrum);
		if (melodic.length === 0) {
			this.clear("인식된 음표가 없습니다.");
			return;
		}

		const { minPitch, maxPitch, duration } = this.getRange(melodic);
		const rows = maxPitch - minPitch + 1;
		const width = this.labelWidth + Math.ceil(duration) * this.pxPerSecond;
		const height = rows * this.rowHeight;

		// 위치 계산 함수 (나중에 재생 위치 표시에도 같은 계산을 사용)
		this.minPitch = minPitch;
		this.maxPitch = maxPitch;

		this.svg = this.createElement("svg", {
			class: "score-svg",
			width,
			height,
			viewBox: `0 0 ${width} ${height}`,
		});

		this.drawRows(minPitch, maxPitch, width);
		this.drawTimeGrid(duration, height);
		this.drawNotes(melodic);

		this.container.innerHTML = "";
		this.container.appendChild(this.svg);
	}

	// ---------- 좌표 변환 ----------
	timeToX(seconds) {
		return this.labelWidth + seconds * this.pxPerSecond;
	}

	pitchToY(pitch) {
		return (this.maxPitch - pitch) * this.rowHeight; // 높은 음일수록 위쪽
	}

	// ---------- 그리기 ----------
	// 건반 줄무늬 배경 + C음 이름
	drawRows(minPitch, maxPitch, width) {
		for (let pitch = minPitch; pitch <= maxPitch; pitch++) {
			const y = this.pitchToY(pitch);
			const isBlack = BLACK_KEYS.includes(pitch % 12);
			this.svg.appendChild(this.createElement("rect", {
				class: isBlack ? "score-row-black" : "score-row-white",
				x: 0,
				y,
				width,
				height: this.rowHeight,
			}));

			if (pitch % 12 === 0) {
				const label = this.createElement("text", {
					class: "score-label",
					x: 4,
					y: y + this.rowHeight - 1,
				});
				label.textContent = this.getNoteName(pitch);
				this.svg.appendChild(label);
			}
		}
	}

	// 1초마다 세로선 + 시간 표시
	drawTimeGrid(duration, height) {
		for (let sec = 0; sec <= Math.ceil(duration); sec++) {
			const x = this.timeToX(sec);
			this.svg.appendChild(this.createElement("line", {
				class: "score-grid-line",
				x1: x,
				y1: 0,
				x2: x,
				y2: height,
			}));
		}
	}

	// 음표 막대
	drawNotes(notes) {
		const minWidth = 2; // 너무 짧은 음도 보이게
		notes.forEach((note) => {
			const x = this.timeToX(note.onset);
			const w = Math.max(minWidth, (note.offset - note.onset) * this.pxPerSecond);
			const rect = this.createElement("rect", {
				class: "score-note",
				x,
				y: this.pitchToY(note.pitch) + 1,
				width: w,
				height: this.rowHeight - 2,
				rx: 2,
			});
			// 나중에 재생 위치와 비교할 때 쓰도록 시간 정보를 저장
			rect.dataset.onset = note.onset;
			rect.dataset.offset = note.offset;

			const title = this.createElement("title");
			title.textContent = `${this.getNoteName(note.pitch)} | ${note.onset.toFixed(2)}s ~ ${note.offset.toFixed(2)}s`;
			rect.appendChild(title);

			this.svg.appendChild(rect);
		});
	}

	// ---------- 도우미 ----------
	getRange(notes) {
		let minPitch = 127;
		let maxPitch = 0;
		let duration = 0;
		notes.forEach((note) => {
			minPitch = Math.min(minPitch, note.pitch);
			maxPitch = Math.max(maxPitch, note.pitch);
			duration = Math.max(duration, note.offset);
		});
		// 위아래로 2반음씩 여유
		return {
			minPitch: Math.max(0, minPitch - 2),
			maxPitch: Math.min(127, maxPitch + 2),
			duration,
		};
	}

	getNoteName(pitch) {
		const octave = Math.floor(pitch / 12) - 1;
		return `${NOTE_NAMES[pitch % 12]}${octave}`;
	}

	createElement(tag, attrs = {}) {
		const el = document.createElementNS(SVG_NS, tag);
		Object.entries(attrs).forEach(([key, value]) => {
			el.setAttribute(key, value);
		});
		return el;
	}
}
