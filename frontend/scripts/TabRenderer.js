// ==========================================================
// TabRenderer
// - 기타 음표를 "시간 기준 타브"로 그리는 class (A안: 리듬 표시 없음)
// - 6개 줄 = 기타 6줄 (위: 1번 줄 높은 미, 아래: 6번 줄 낮은 미)
// - 숫자 = 몇 번째 프렛을 누르는지 (0 = 개방현)
// - 가로축은 피아노롤과 같은 시간(초) 기준이라 원본 재생 위치 표시를 그대로 쓸 수 있음
// ==========================================================

const SVG_NS = "http://www.w3.org/2000/svg";

// 기타 표준 조율: 6번 줄(E2) → 1번 줄(E4)의 MIDI 음높이
const STANDARD_TUNING = [40, 45, 50, 55, 59, 64];
const STRING_NAMES = ["E", "A", "D", "G", "B", "e"];
const MAX_FRET = 24; // abcjs 타브와 같은 최대 프렛
const SAME_TIME = 0.03; // 0.03초 안에 시작하면 동시에 치는 음(화음)으로 봄

export class TabRenderer {
	constructor(container) {
		this.container = container;
		this.pxPerSecond = 80; // 1초 = 80px (피아노롤과 같은 비율)
		this.labelWidth = 40;  // 왼쪽 줄 이름 칸
		this.lineGap = 18;     // 줄 간격
		this.top = 24;         // 위쪽 여백 (시간 글자 자리)
		this.svg = null;
		this.cursor = null;
		this.userScrolledAt = -Infinity; // 사용자가 직접 스크롤한 시각 (아직 없음)

		// 사용자가 직접 스크롤하면 잠깐 동안 자동 따라가기를 멈춤 (ScoreRenderer와 같은 규칙)
		const markUserScroll = () => { this.userScrolledAt = performance.now(); };
		this.container.addEventListener("wheel", (event) => {
			// 가로 스크롤(또는 Shift+휠)일 때만 (페이지를 위아래로 내리는 건 제외)
			if (Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey) markUserScroll();
		}, { passive: true });
		this.container.addEventListener("touchstart", markUserScroll, { passive: true });
		this.container.addEventListener("pointerdown", markUserScroll);
	}

	clear() {
		this.container.innerHTML = "";
		this.svg = null;
		this.cursor = null;
	}

	// 줄 번호 0(6번 줄, 낮은 미) ~ 5(1번 줄, 높은 미) → y좌표 (높은 줄이 위)
	stringToY(string) {
		return this.top + (5 - string) * this.lineGap;
	}

	timeToX(seconds) {
		return this.labelWidth + seconds * this.pxPerSecond;
	}

	render(notes) {
		const positions = TabRenderer.assignFrets(notes);
		// 곡 길이 (음표가 아주 많아도 되도록 Math.max(...배열) 대신 반복문)
		let duration = 1;
		notes.forEach((note) => { duration = Math.max(duration, note.offset); });
		const width = this.labelWidth + Math.ceil(duration) * this.pxPerSecond;
		const height = this.top + 5 * this.lineGap + 16;

		this.svg = this.createElement("svg", {
			class: "tab-svg",
			width,
			height,
			viewBox: `0 0 ${width} ${height}`,
		});

		// 6개 줄 + 줄 이름
		for (let string = 0; string < 6; string++) {
			const y = this.stringToY(string);
			this.svg.appendChild(this.createElement("line", {
				class: "tab-string",
				x1: this.labelWidth - 6,
				y1: y,
				x2: width,
				y2: y,
			}));
			const label = this.createElement("text", { class: "tab-string-name", x: 8, y: y + 4 });
			label.textContent = STRING_NAMES[string];
			this.svg.appendChild(label);
		}

		// 5초마다 시간 글자
		for (let sec = 0; sec <= Math.ceil(duration); sec += 5) {
			const label = this.createElement("text", { class: "score-label", x: this.timeToX(sec) + 3, y: 12 });
			label.textContent = `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
			this.svg.appendChild(label);
		}

		// 프렛 번호 (흰 바탕 위에 숫자 → 줄이 숫자를 가리지 않게)
		positions.forEach(({ note, string, fret, shifted }) => {
			// 빈 줄이 없어 다른 숫자와 같은 자리에 놓인 "?"는 오른쪽으로 비켜서 그림
			const x = this.timeToX(note.onset) + (shifted ? 9 : 0);
			const y = this.stringToY(string);
			const text = fret === null ? "?" : String(fret);
			const group = this.createElement("g", { class: fret === null ? "tab-fret is-unplayable" : "tab-fret" });
			group.appendChild(this.createElement("rect", {
				class: "tab-fret-bg",
				x: x - 1,
				y: y - 7,
				width: text.length * 7 + 2,
				height: 14,
			}));
			const label = this.createElement("text", { class: "tab-fret-text", x, y: y + 4 });
			label.textContent = text;
			const title = this.createElement("title");
			title.textContent = fret === null
				? `기타 음역 밖의 음 | ${note.onset.toFixed(2)}s`
				: `${6 - string}번 줄 ${fret}프렛 | ${note.onset.toFixed(2)}s`;
			group.append(label, title);
			this.svg.appendChild(group);
		});

		// 원본 재생 위치
		this.cursor = this.createElement("line", {
			class: "score-cursor is-hidden",
			x1: this.labelWidth,
			y1: 0,
			x2: this.labelWidth,
			y2: height,
		});
		this.svg.appendChild(this.cursor);

		this.container.innerHTML = "";
		this.container.appendChild(this.svg);
	}

	// follow: true면 재생 위치가 화면 밖으로 나갈 때 가로 스크롤을 따라감
	setCursor(seconds, follow = true) {
		if (!this.cursor) return;
		const x = this.timeToX(seconds);
		this.cursor.setAttribute("x1", x);
		this.cursor.setAttribute("x2", x);
		this.cursor.classList.toggle("is-hidden", seconds <= 0);
		if (follow && seconds > 0) this.keepVisible(x);
	}

	// 재생 위치가 화면 밖으로 나가면 가로 스크롤을 따라 이동
	// (사용자가 2초 안에 직접 스크롤했다면 방해하지 않음)
	keepVisible(x) {
		if (performance.now() - this.userScrolledAt < 2000) return;
		const view = this.container;
		const left = view.scrollLeft;
		const right = left + view.clientWidth;
		if (x < left + 20 || x > right - 40) {
			view.scrollLeft = Math.max(0, x - view.clientWidth * 0.3);
		}
	}

	// ---------- 줄·프렛 정하기 ----------
	// 같은 음도 여러 줄에서 칠 수 있음 (예: 미(E4)는 1번 줄 0프렛, 2번 줄 5프렛 ...)
	// 규칙: 동시에 치는 음은 서로 다른 줄, 손 위치(직전 프렛들)와 가까운 프렛을 우선
	static assignFrets(notes, tuning = STANDARD_TUNING) {
		const sorted = [...notes].sort((a, b) => a.onset - b.onset || b.pitch - a.pitch);
		const result = [];
		let hand = 2; // 손 위치 (처음엔 낮은 프렛 근처)

		let i = 0;
		while (i < sorted.length) {
			// 같은 시각에 시작하는 음 묶기
			const chord = [sorted[i]];
			while (i + chord.length < sorted.length && sorted[i + chord.length].onset - sorted[i].onset < SAME_TIME) {
				chord.push(sorted[i + chord.length]);
			}
			i += chord.length;

			const used = new Set();
			const frets = [];
			const unplayable = []; // 칠 수 없는 음 (줄을 다 정한 뒤 빈 줄에 "?" 표시)
			// 높은 음부터 높은 줄에 배치
			chord.sort((a, b) => b.pitch - a.pitch).forEach((note) => {
				let best = null;
				tuning.forEach((open, string) => {
					const fret = Math.round(note.pitch) - open;
					if (fret < 0 || fret > MAX_FRET || used.has(string)) return;
					// 손 위치와 가까울수록, 낮은 프렛일수록, 개방현일수록 좋음 (손이 위로만 올라가는 것 방지)
					const cost = Math.abs(fret - hand) * 0.5 + fret * 0.5 - (fret === 0 ? 1 : 0);
					if (!best || cost < best.cost) best = { string, fret, cost };
				});
				if (best) {
					used.add(best.string);
					if (best.fret > 0) frets.push(best.fret);
					result.push({ note, string: best.string, fret: best.fret });
				} else {
					unplayable.push(note);
				}
			});
			// 칠 수 없는 음 (음역 밖이거나 줄이 모자람) → 가까운 쪽 끝 줄부터 비어 있는 줄에 "?"
			// 빈 줄이 없으면 끝 줄에 두고 shifted 표시 (그릴 때 옆으로 비킴 → 숫자와 겹치지 않게)
			unplayable.forEach((note) => {
				const order = Math.round(note.pitch) < 52 ? [0, 1, 2, 3, 4, 5] : [5, 4, 3, 2, 1, 0];
				const free = order.find((string) => !used.has(string));
				if (free === undefined) {
					result.push({ note, string: order[0], fret: null, shifted: true });
				} else {
					used.add(free);
					result.push({ note, string: free, fret: null, shifted: false });
				}
			});
			// 손 위치는 천천히 따라감 (한 음 때문에 손이 크게 움직이지 않게)
			if (frets.length) hand = hand * 0.5 + (frets.reduce((a, b) => a + b, 0) / frets.length) * 0.5;
		}
		return result;
	}

	createElement(tag, attrs = {}) {
		const el = document.createElementNS(SVG_NS, tag);
		Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
		return el;
	}
}
