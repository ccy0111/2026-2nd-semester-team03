// ==========================================================
// ScoreRenderer
// - 모델이 돌려준 notes 배열을 피아노롤(SVG)로 그리는 class
// - 가로축: 시간(초) → x좌표, 세로축: 음높이(MIDI pitch) → y좌표
//
// note 형식 (docs/API.md)
//   { program, pitch, onset, offset, isDrum }
//   pitch: 0~127 (60 = 가운데 도, C4)
//   onset, offset: 원본 음원 기준 시작/끝 시각(초)
//
// 악기: program(0~127, GM 악기 번호) → instruments.js의 계열 이름·색으로 칠함
//        드럼(isDrum)은 음높이 대신 맨 아래 "드럼 줄"에 짧은 막대로 표시
//
// chunkStatus 형식 (docs/EOS_CHUNK_RECOVERY.md)
//   { index, startSec, endSec, status: "ok" | "recovered" | "failed", attempts, ... }
// ==========================================================

import { getInstrument, instrumentOrder } from "./instruments.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const BLACK_KEYS = [1, 3, 6, 8, 10]; // 한 옥타브 안에서 검은 건반 위치
const MIN_ROWS = 24;                 // 음표가 적어도 최소 2옥타브 높이로 그림
const DRUM_LANE_HEIGHT = 20;         // 맨 아래 드럼 줄 높이

export class ScoreRenderer {
	constructor(container, options = {}) {
		this.container = container;
		this.pxPerSecond = options.pxPerSecond ?? 80; // 1초 = 80px
		this.rowHeight = options.rowHeight ?? 10;     // 반음 하나 = 10px
		this.labelWidth = options.labelWidth ?? 40;   // 왼쪽 음이름 칸
		this.svg = null;
		this.cursor = null;
		this.instrumentGroups = new Map(); // 악기 key → 그 악기 음표를 담은 <g>
		this.onSeek = null; // 악보 클릭 시 호출할 함수 (App에서 연결)
		this.userScrolledAt = 0; // 사용자가 직접 스크롤한 시각

		// 사용자가 직접 스크롤하면 잠깐 동안 자동 따라가기를 멈춤
		const markUserScroll = () => { this.userScrolledAt = performance.now(); };
		this.container.addEventListener("wheel", (event) => {
			// 가로 스크롤(또는 Shift+휠)일 때만 (페이지를 위아래로 내리는 건 제외)
			if (Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey) markUserScroll();
		}, { passive: true });
		this.container.addEventListener("touchstart", markUserScroll, { passive: true });
		this.container.addEventListener("pointerdown", (event) => {
			// 스크롤바를 잡은 경우 (악보 클릭은 제외)
			if (!this.svg || !this.svg.contains(event.target)) markUserScroll();
		});

		// 악보 클릭 → 그 시각으로 이동
		this.container.addEventListener("click", (event) => this.handleClick(event));
	}

	// 결과 지우고 안내 문구 표시
	clear(message = "채보가 끝나면 여기에 음표가 표시됩니다.") {
		this.container.innerHTML = "";
		const p = document.createElement("p");
		p.className = "text-sub";
		p.textContent = message;
		this.container.appendChild(p);
		this.svg = null;
		this.cursor = null;
		this.instrumentGroups.clear();
	}

	// 반환값: 그려진 악기 목록 [{ key, name, color, count }] (범례용)
	render(notes, chunkStatuses = []) {
		// 값이 이상한 음표·구간은 그리지 않음 (그림 전체가 깨지는 것 방지)
		const valid = notes.filter((note) => ScoreRenderer.isValidNote(note));
		const melodic = valid.filter((note) => !note.isDrum);
		const drums = valid.filter((note) => note.isDrum);
		chunkStatuses = chunkStatuses.filter((chunk) => this.isValidChunk(chunk));
		if (valid.length === 0 && chunkStatuses.length === 0) {
			this.clear("인식된 음표가 없습니다.");
			return [];
		}

		const { minPitch, maxPitch, duration } = this.getRange(melodic, chunkStatuses, drums);
		const width = this.labelWidth + Math.ceil(duration) * this.pxPerSecond;
		const pitchHeight = (maxPitch - minPitch + 1) * this.rowHeight;
		const laneHeight = drums.length > 0 ? DRUM_LANE_HEIGHT : 0; // 드럼이 있을 때만 드럼 줄
		const height = pitchHeight + laneHeight;

		// 좌표 계산(pitchToY 등)에 쓰도록 저장
		this.minPitch = minPitch;
		this.maxPitch = maxPitch;
		this.pitchHeight = pitchHeight;
		this.height = height;

		this.svg = this.createElement("svg", {
			class: "score-svg",
			width,
			height,
			viewBox: `0 0 ${width} ${height}`,
		});

		// 그리는 순서 = 쌓이는 순서 (나중에 그린 것이 위)
		this.drawRows(minPitch, maxPitch, width);
		if (laneHeight) this.drawDrumLane(width, pitchHeight, laneHeight);
		this.drawTimeGrid(duration, height);
		this.drawChunkStatuses(chunkStatuses, height);
		this.drawTimeLabels(duration); // 실패 구간 박스에 가려지지 않게 박스 다음에
		const instruments = this.drawNotes(melodic, drums, laneHeight);
		this.drawCursor(height);

		this.container.innerHTML = "";
		this.container.appendChild(this.svg);
		// 스크롤 계산에 쓸 여백 (매 프레임 계산하지 않도록 한 번만)
		this.paddingLeft = parseFloat(getComputedStyle(this.container).paddingLeft) || 0;
		return instruments;
	}

	// 범례 체크박스에서 호출: 악기별 음표 보이기/숨기기
	setInstrumentVisible(key, visible) {
		this.instrumentGroups.get(key)?.classList.toggle("is-hidden", !visible);
	}

	// ---------- 좌표 변환 ----------
	timeToX(seconds) {
		return this.labelWidth + seconds * this.pxPerSecond;
	}

	xToTime(x) {
		return Math.max(0, (x - this.labelWidth) / this.pxPerSecond);
	}

	pitchToY(pitch) {
		return (this.maxPitch - pitch) * this.rowHeight; // 높은 음일수록 위쪽
	}

	// ---------- 재생 위치 ----------
	// AudioPlayer가 호출 (재생 중에는 매 프레임)
	// follow: true면 재생 위치가 화면 밖으로 나갈 때 스크롤을 따라감
	setCursor(seconds, follow = true) {
		if (!this.cursor) return;
		const x = this.timeToX(seconds);
		this.cursor.setAttribute("x1", x);
		this.cursor.setAttribute("x2", x);
		this.cursor.classList.toggle("is-hidden", seconds <= 0);
		if (follow) this.keepVisible(x);
	}

	// 재생 위치가 화면 밖으로 나가면 가로 스크롤을 따라 이동
	// (사용자가 2초 안에 직접 스크롤했다면 방해하지 않음)
	keepVisible(x) {
		if (performance.now() - this.userScrolledAt < 2000) return;
		const view = this.container;
		const viewX = x + (this.paddingLeft ?? 0); // 스크롤 영역 안에서의 실제 위치
		const left = view.scrollLeft;
		const right = left + view.clientWidth;
		if (viewX < left + 20 || viewX > right - 40) {
			view.scrollLeft = Math.max(0, viewX - view.clientWidth * 0.3);
		}
	}

	handleClick(event) {
		if (!this.svg || !this.onSeek) return;
		// 악보(SVG) 위를 클릭했을 때만 (가로 스크롤바·여백 클릭은 무시)
		if (!this.svg.contains(event.target)) return;
		const rect = this.svg.getBoundingClientRect();
		const x = event.clientX - rect.left;
		if (x < this.labelWidth) return; // 음이름 칸 클릭은 무시
		this.onSeek(this.xToTime(x));
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

	// 1초마다 세로선 (5초마다 진하게)
	drawTimeGrid(duration, height) {
		for (let sec = 0; sec <= Math.ceil(duration); sec++) {
			const x = this.timeToX(sec);
			this.svg.appendChild(this.createElement("line", {
				class: sec % 5 === 0 ? "score-grid-line is-strong" : "score-grid-line",
				x1: x,
				y1: 0,
				x2: x,
				y2: height,
			}));
		}
	}

	// 5초마다 시간 글자
	drawTimeLabels(duration) {
		for (let sec = 0; sec <= Math.ceil(duration); sec += 5) {
			const label = this.createElement("text", {
				class: "score-label score-time-label",
				x: this.timeToX(sec) + 3,
				y: 11,
			});
			label.textContent = this.formatTime(sec);
			this.svg.appendChild(label);
		}
	}

	// 실패 구간: 회색 박스 + 글자 / 복구 구간: 옅은 노란 박스
	drawChunkStatuses(chunkStatuses, height) {
		chunkStatuses.forEach((chunk) => {
			if (chunk.status !== "failed" && chunk.status !== "recovered") return;

			const x = this.timeToX(chunk.startSec);
			const width = Math.max(0, (chunk.endSec - chunk.startSec) * this.pxPerSecond);
			const box = this.createElement("rect", {
				class: chunk.status === "failed" ? "score-chunk-failed" : "score-chunk-recovered",
				x,
				y: 0,
				width,
				height,
			});
			const title = this.createElement("title");
			title.textContent = chunk.status === "failed"
				? `채보 실패 구간 (${this.formatTime(chunk.startSec)} ~ ${this.formatTime(chunk.endSec)})`
				: `재시도로 복구된 구간 (${this.formatTime(chunk.startSec)} ~ ${this.formatTime(chunk.endSec)})`;
			box.appendChild(title);
			this.svg.appendChild(box);

			if (chunk.status === "failed") {
				const label = this.createElement("text", {
					class: "score-chunk-label",
					x: x + width / 2,
					y: height / 2,
				});
				label.textContent = "채보 실패 구간";
				this.svg.appendChild(label);
			}
		});
	}

	// 맨 아래 드럼 줄 배경 + 이름
	drawDrumLane(width, top, laneHeight) {
		this.svg.appendChild(this.createElement("rect", {
			class: "score-drum-lane",
			x: 0,
			y: top,
			width,
			height: laneHeight,
		}));
		const label = this.createElement("text", {
			class: "score-label",
			x: 4,
			y: top + laneHeight / 2 + 4,
		});
		label.textContent = "드럼";
		this.svg.appendChild(label);
	}

	// 음표 막대: 악기별 <g>로 묶어서 색칠 (숨기기·보이기를 묶음 단위로 하려고)
	// 반환값: [{ key, name, color, count }]
	drawNotes(melodic, drums, laneHeight) {
		const minWidth = 2; // 너무 짧은 음도 보이게
		const counts = new Map(); // key → { key, name, color, count }

		const groupFor = (instrument) => {
			if (!this.instrumentGroups.has(instrument.key)) {
				const group = this.createElement("g", { class: "score-instrument" });
				group.dataset.instrument = instrument.key;
				this.instrumentGroups.set(instrument.key, group);
				counts.set(instrument.key, { ...instrument, count: 0 });
			}
			counts.get(instrument.key).count += 1;
			return this.instrumentGroups.get(instrument.key);
		};

		// 음높이가 있는 음표
		melodic.forEach((note) => {
			const instrument = getInstrument(note);
			const rect = this.createElement("rect", {
				class: "score-note",
				x: this.timeToX(note.onset),
				y: this.pitchToY(note.pitch) + 1,
				width: Math.max(minWidth, (note.offset - note.onset) * this.pxPerSecond),
				height: this.rowHeight - 2,
				rx: 2,
				style: `fill: ${instrument.color}`, // CSS보다 우선하도록 style로 지정
			});
			const title = this.createElement("title");
			title.textContent = `${instrument.name} · ${this.getNoteName(note.pitch)} | ${note.onset.toFixed(2)}s ~ ${note.offset.toFixed(2)}s`;
			rect.appendChild(title);
			groupFor(instrument).appendChild(rect);
		});

		// 드럼: 맨 아래 드럼 줄에 짧은 세로 막대 (언제 쳤는지만 표시)
		drums.forEach((note) => {
			const instrument = getInstrument(note);
			const rect = this.createElement("rect", {
				class: "score-drum-hit",
				x: this.timeToX(note.onset) - 1,
				y: this.pitchHeight + 3,
				width: 3,
				height: laneHeight - 6,
				style: `fill: ${instrument.color}`,
			});
			const title = this.createElement("title");
			title.textContent = `${instrument.name} (드럼 번호 ${note.pitch}) | ${note.onset.toFixed(2)}s`;
			rect.appendChild(title);
			groupFor(instrument).appendChild(rect);
		});

		// 악기 순서대로 <g>를 붙임 (피아노가 맨 아래, 드럼이 맨 위)
		const instruments = [...counts.values()].sort((a, b) => instrumentOrder(a.key) - instrumentOrder(b.key));
		instruments.forEach((instrument) => {
			this.svg.appendChild(this.instrumentGroups.get(instrument.key));
		});
		return instruments;
	}

	// 재생 위치 세로선 (처음엔 숨김)
	drawCursor(height) {
		this.cursor = this.createElement("line", {
			class: "score-cursor is-hidden",
			x1: this.labelWidth,
			y1: 0,
			x2: this.labelWidth,
			y2: height,
		});
		this.svg.appendChild(this.cursor);
	}

	// ---------- 도우미 ----------
	// static: 악보 탭(SheetView)도 같은 규칙으로 음표를 거르도록 class 이름으로 부를 수 있게 함
	static isValidNote(note) {
		return Number.isFinite(note.pitch) && note.pitch >= 0 && note.pitch <= 127
			&& Number.isFinite(note.onset) && Number.isFinite(note.offset)
			&& note.onset >= 0 && note.offset >= note.onset;
	}

	isValidChunk(chunk) {
		return Number.isFinite(chunk?.startSec) && Number.isFinite(chunk?.endSec)
			&& chunk.endSec >= chunk.startSec;
	}

	getRange(notes, chunkStatuses, drums = []) {
		let minPitch = 127;
		let maxPitch = 0;
		let duration = 0;
		notes.forEach((note) => {
			minPitch = Math.min(minPitch, note.pitch);
			maxPitch = Math.max(maxPitch, note.pitch);
			duration = Math.max(duration, note.offset);
		});
		// 드럼만 곡 끝에 있어도 그 시각까지 그리도록
		drums.forEach((note) => {
			duration = Math.max(duration, note.offset);
		});
		// 실패 구간이 곡 끝에 있어도 그 구간까지 그리도록
		chunkStatuses.forEach((chunk) => {
			duration = Math.max(duration, chunk.endSec);
		});

		// 음표가 하나도 없으면 가운데 도 주변으로
		if (notes.length === 0) {
			minPitch = 60;
			maxPitch = 60;
		}

		// 위아래로 2반음씩 여유 + 최소 높이 보장
		minPitch -= 2;
		maxPitch += 2;
		while (maxPitch - minPitch + 1 < MIN_ROWS) {
			minPitch -= 1;
			maxPitch += 1;
		}
		return {
			minPitch: Math.max(0, minPitch),
			maxPitch: Math.min(127, maxPitch),
			duration,
		};
	}

	getNoteName(pitch) {
		const p = Math.round(pitch);
		const octave = Math.floor(p / 12) - 1;
		return `${NOTE_NAMES[((p % 12) + 12) % 12]}${octave}`;
	}

	formatTime(seconds) {
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${m}:${String(s).padStart(2, "0")}`;
	}

	createElement(tag, attrs = {}) {
		const el = document.createElementNS(SVG_NS, tag);
		Object.entries(attrs).forEach(([key, value]) => {
			el.setAttribute(key, value);
		});
		return el;
	}
}
