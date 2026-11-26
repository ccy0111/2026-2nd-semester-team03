// ==========================================================
// SheetView
// - "악보" 탭 화면을 담당하는 class
//   악기 선택 → 그 악기 음표만 골라 → 오선보(abcjs)로 그림
// - 템포(BPM)·박자 입력, 기타 표시 방식(오선보 / 오선보+타브) 선택
//   (타브만 따로 그리던 방식은 한 줄로 길게 이어지다 잘려서 뺌 → 타브는 abcjs가 오선보 아래에 줄바꿈해 그림)
// - abcjs의 재생 기능으로 악보를 소리로 듣고, 재생 중인 음표를 표시
// - 음표를 클릭하면 그 위치부터 재생
//
// abcjs: 오선 악보 라이브러리 (MIT). index.html에서 <script>로 불러와 window.ABCJS로 사용
// ==========================================================
import { AbcBuilder, SHEET_KEYS } from "./AbcBuilder.js";
import { ScoreRenderer } from "./ScoreRenderer.js";
import { getInstrument, instrumentOrder } from "./instruments.js";

// 오선보 한 줄의 너비(px). 화면 크기에 맞춰 줄이지 않고 고정 → 좁은 화면에서는 가로 스크롤
// (휴대폰에서 악보 전체를 줄이면 글자·음표가 너무 작아져 읽을 수 없음)
const STAFF_WIDTH = 940;
const DEFAULT_BPM = 120;

// abcjs 재생 버튼 안내 문구 (abcjs 6.4.4의 SynthController는 문구 설정을 넘겨주지 않아 직접 바꿈)
const SYNTH_TITLES = {
	".abcjs-midi-reset": "처음으로",
	".abcjs-midi-start": "재생 / 일시정지",
	".abcjs-midi-progress-background": "누른 위치부터 재생",
	".abcjs-midi-tempo": "재생 속도(%)",
};

export class SheetView {
	constructor(elements) {
		this.instrumentSelect = elements.instrumentSelect;
		this.guitarModeSelect = elements.guitarModeSelect;
		this.guitarModeField = elements.guitarModeField;
		this.bpmInput = elements.bpmInput;
		this.meterSelect = elements.meterSelect;
		this.audioArea = elements.audioArea;   // abcjs 재생 버튼 자리
		this.sheetArea = elements.sheetArea;   // 악보가 그려질 자리
		this.messageText = elements.messageText;
		this.pdfButton = elements.pdfButton;   // 악보 PDF 저장(인쇄) 버튼

		this.notes = [];
		this.songTitle = "";
		this.startSec = 0;
		this.isVisible = false;
		this.needsRender = false;
		this.synthControl = null;
		this.visualObj = null;   // 지금 그려진 abcjs 악보
		this.lastMoment = 0;     // 악보 마지막 음의 시각(ms) — 클릭 위치를 재생 비율로 바꿀 때 사용
		this.noteTimes = new Map(); // 악보 SVG 요소 → 그 음이 나오는 시각(ms)
		this.bpm = DEFAULT_BPM;  // 마지막으로 쓴 올바른 템포
		this.soundFailed = false; // 악보 소리 파일 받기에 실패했는지

		// 설정을 바꾸면 다시 그림
		[this.instrumentSelect, this.guitarModeSelect, this.meterSelect].forEach((el) => {
			el.addEventListener("change", () => this.render());
		});
		this.bpmInput.addEventListener("change", () => this.render());
		// 음표 클릭 → 그 위치부터 재생
		this.sheetArea.addEventListener("click", (event) => this.handleSheetClick(event));
		this.pdfButton.addEventListener("click", () => this.printPdf());
	}

	// 채보 결과 받기 (instruments: InstrumentLegend와 같은 목록)
	// songTitle: 악보 맨 위 큰 제목 (MP3 파일 이름)
	setResult(notes, instruments, songTitle = "") {
		this.songTitle = songTitle;
		// 피아노롤(ScoreRenderer)과 같은 규칙으로 이상한 음표를 거름 (악기 목록 개수와 맞도록)
		this.notes = notes.filter((note) => ScoreRenderer.isValidNote(note));
		// 곡 첫 음 시각 (음표가 아주 많아도 되도록 Math.min(...배열) 대신 반복문)
		this.startSec = this.notes.length ? Infinity : 0;
		this.notes.forEach((note) => { this.startSec = Math.min(this.startSec, note.onset); });

		// 템포는 드럼을 포함한 전체 음의 시작 시각으로 추정 (모든 악기가 같은 템포)
		this.bpm = AbcBuilder.estimateBpm(this.notes);
		this.bpmInput.value = this.bpm;

		// 악기 목록 채우기
		this.instrumentSelect.innerHTML = "";
		[...instruments]
			.sort((a, b) => instrumentOrder(a.key) - instrumentOrder(b.key))
			.forEach((instrument) => {
				const option = document.createElement("option");
				option.value = instrument.key;
				option.textContent = `${instrument.name} (${instrument.count})`;
				option.dataset.name = instrument.name;
				this.instrumentSelect.appendChild(option);
			});

		this.needsRender = true;
		if (this.isVisible) this.render();
	}

	// 탭이 보일 때만 그림 (긴 곡도 피아노롤만 볼 때는 시간을 쓰지 않도록)
	show() {
		this.isVisible = true;
		if (this.needsRender) this.render();
	}

	hide() {
		this.isVisible = false;
		this.stopSynth();
	}

	clear() {
		this.stopSynth();
		this.notes = [];
		this.instrumentSelect.innerHTML = "";
		this.sheetArea.innerHTML = "";
		this.setMessage("");
		this.visualObj = null;
		this.noteTimes = new Map();
		this.needsRender = false;
	}

	// ---------- 그리기 ----------
	render() {
		this.needsRender = false;
		this.stopSynth();

		const key = this.instrumentSelect.value;
		const isGuitar = key === SHEET_KEYS.GUITAR_KEY;
		this.guitarModeField.classList.toggle("is-hidden", !isGuitar);
		if (!key) {
			this.sheetArea.innerHTML = "";
			this.setMessage("악보로 만들 음표가 없어요.");
			return;
		}

		const notes = this.notes.filter((note) => getInstrument(note).key === key);
		const guitarMode = isGuitar ? this.guitarModeSelect.value : "staff";

		// 오선보 (abcjs)
		this.visualObj = null;
		const ABCJS = window.ABCJS;
		if (!ABCJS) {
			this.sheetArea.innerHTML = "";
			this.setMessage("악보 라이브러리(abcjs)를 불러오지 못했어요. 인터넷 연결을 확인하고 새로고침해 주세요.");
			return;
		}

		const bpm = this.readBpm();
		const name = this.instrumentSelect.selectedOptions[0]?.dataset.name ?? "";
		let visualObj;
		try {
			const { abc, tablature } = AbcBuilder.build({
				notes,
				instrumentKey: key,
				instrumentName: name,
				songTitle: this.songTitle,
				bpm,
				beatsPerBar: Number(this.meterSelect.value),
				startSec: this.startSec,
				withTab: guitarMode === "both",
			});

			this.sheetArea.innerHTML = "";
			const tunes = ABCJS.renderAbc(this.sheetArea, abc, {
				staffwidth: STAFF_WIDTH,
				add_classes: true,
				tablature,
				paddingleft: 0,
				paddingright: 0,
			});
			visualObj = tunes[0];
		} catch (error) {
			// 예상하지 못한 음표 때문에 그리기에 실패해도 화면 전체가 멈추지 않게
			console.warn(error);
			this.sheetArea.innerHTML = "";
			this.audioArea.classList.add("is-hidden");
			this.setMessage("악보를 그리지 못했어요.");
			return;
		}
		this.visualObj = visualObj;
		this.lastMoment = this.computeTiming(visualObj);
		if (visualObj?.warnings?.length) console.warn("abcjs warnings", visualObj.warnings);

		let message = `템포 ${bpm} BPM 기준으로 16분음표 단위로 정리한 악보예요. 템포가 맞지 않으면 직접 고쳐 보세요.`;
		if (key !== SHEET_KEYS.DRUM_KEY) message += " 같은 손에서 겹치는 음은 다음 음이 시작될 때 끊어서 표시해요.";
		this.setMessage(message);
		this.setupSynth(ABCJS, visualObj);
	}

	// abcjs에게 음마다 시각을 계산하게 해서 "SVG 요소 → 시각(ms)" 표를 만들고, 마지막 음의 시각을 돌려줌
	// timing.elements: 그 시각에 나오는 음표들의 SVG 요소 묶음 (add_classes: true일 때)
	computeTiming(visualObj) {
		this.noteTimes = new Map();
		try {
			const timings = visualObj?.setTiming?.() ?? [];
			let last = 0;
			timings.forEach((timing) => {
				if (!Number.isFinite(timing.milliseconds)) return;
				last = Math.max(last, timing.milliseconds);
				(timing.elements ?? []).forEach((group) => {
					(group ?? []).forEach((el) => {
						if (!this.noteTimes.has(el)) this.noteTimes.set(el, timing.milliseconds); // 붙임줄로 이어진 음은 처음 시각
					});
				});
			});
			return last;
		} catch (error) {
			console.warn(error);
			return 0;
		}
	}

	// 클릭한 곳에서 가장 가까운 음표를 찾아 그 시각부터 재생
	// (abcjs 자체 클릭 기능은 큰보표 아래 줄 음표를 놓치는 경우가 있어 직접 거리로 찾음)
	handleSheetClick(event) {
		const MAX_DISTANCE = 30; // 음표에서 이 거리(px) 안쪽을 눌렀을 때만
		let best = null;
		let bestDistance = MAX_DISTANCE * MAX_DISTANCE;
		this.noteTimes.forEach((ms, el) => {
			const rect = el.getBoundingClientRect();
			if (rect.width === 0 && rect.height === 0) return;
			const dx = Math.max(rect.left - event.clientX, 0, event.clientX - rect.right);
			const dy = Math.max(rect.top - event.clientY, 0, event.clientY - rect.bottom);
			const distance = dx * dx + dy * dy;
			if (distance <= bestDistance) {
				bestDistance = distance;
				best = ms;
			}
		});
		if (best !== null) this.playFrom(best);
	}

	// ms: 악보 시작부터의 시각
	// abcjs 재생기의 seek(비율)를 씀: 비율로 넘기면 재생 속도(%)를 바꿔도 같은 자리로 감
	async playFrom(ms) {
		const synth = this.synthControl;
		if (!synth || !Number.isFinite(ms) || this.lastMoment <= 0 || this.soundFailed) return;
		const ratio = Math.min(Math.max(ms / this.lastMoment, 0), 1);
		try {
			// abcjs의 play()는 재생 ↔ 일시정지 전환이라, 멈춰 있을 때만 누름
			if (!synth.isStarted) {
				const result = await synth.play();
				if (result?.status === "error" || !synth.isStarted) return; // 소리 준비 실패
			}
			synth.seek(ratio);
		} catch (error) {
			console.warn(error);
		}
	}

	// ---------- PDF 저장 ----------
	// 브라우저 인쇄 창을 열고, style.css의 @media print가 악보 종이만 인쇄되게 함
	// (인쇄 창에서 대상을 "PDF로 저장"으로 고르면 PDF 파일이 됨)
	printPdf() {
		const svgs = this.sheetArea.querySelectorAll("svg");
		if (!this.visualObj || svgs.length === 0) {
			this.setMessage("저장할 악보가 없어요.");
			return;
		}
		this.stopSynth();
		// viewBox를 넣어 두면 인쇄할 때 종이 너비에 맞춰 악보 전체가 같이 줄어듦 (화면 모양은 그대로)
		svgs.forEach((svg) => {
			if (svg.hasAttribute("viewBox")) return;
			const width = parseFloat(svg.getAttribute("width"));
			const height = parseFloat(svg.getAttribute("height"));
			if (width > 0 && height > 0) svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
		});
		// 저장 창의 기본 파일 이름 = 문서 제목 → "노래 제목 - 악기" 로 잠깐 바꿨다가 되돌림
		const originalTitle = document.title;
		const name = this.instrumentSelect.selectedOptions[0]?.dataset.name ?? "";
		document.title = [this.songTitle || "악보", name].filter(Boolean).join(" - ");
		window.addEventListener("afterprint", () => { document.title = originalTitle; }, { once: true });
		window.print();
	}

	// 템포 입력 읽기: 40~240으로 맞추고, 비었거나 숫자가 아니면 마지막 값 유지 (40으로 바뀌지 않게)
	readBpm() {
		const text = this.bpmInput.value.trim();
		const value = Number(text);
		if (text !== "" && Number.isFinite(value)) {
			this.bpm = Math.min(240, Math.max(40, Math.round(value)));
		}
		this.bpmInput.value = this.bpm;
		return this.bpm;
	}

	setMessage(text) {
		this.messageText.textContent = text;
	}

	// ---------- 악보 재생 (abcjs) ----------
	setupSynth(ABCJS, visualObj) {
		if (!visualObj || !ABCJS.synth?.supportsAudio?.()) {
			this.audioArea.classList.add("is-hidden");
			return;
		}
		this.audioArea.classList.remove("is-hidden");
		if (!this.synthControl) this.synthControl = this.createSynthControl(ABCJS);

		// 새 악보 연결. 소리 파일(사운드폰트)은 여기서 받지 않고, 재생 버튼을 누를 때 받음
		// isLoaded = false: 다음 재생 때 새 악보(바뀐 템포·악기)로 소리를 다시 준비하게 함
		// (abcjs는 setTune만으로는 이전 악보의 재생 준비를 그대로 씀)
		this.synthControl.isLoaded = false;
		this.synthControl.setTune(visualObj, false).catch((error) => {
			console.warn(error);
			this.setMessage("악보 재생을 준비하지 못했어요. (악보 보기는 그대로 사용할 수 있어요)");
		});
	}

	// abcjs 재생기 만들기 + 소리 파일을 받지 못했을 때의 처리
	createSynthControl(ABCJS) {
		const synth = new ABCJS.synth.SynthController();

		// 주의: 아래 처리는 abcjs 6.4.4 내부 값(go, play, randomAccess, onWarp,
		//       isLoaded, isLoading, isStarted)에 기대고 있음 → abcjs 버전을 올리면 다시 확인할 것

		// 재생 준비(go): 재생·진행 막대·속도(%) 버튼이 모두 처음에 이 함수를 거침
		// → 소리 파일 실패 처리는 여기 한 곳에서만 함
		//   - 소리 파일을 하나도 받지 못했으면(인터넷 없음 등) 실패
		//   - 한 번 실패하면 다시 받지 않음: abcjs가 실패한 소리도 "받아 둔 것"으로 기억해서
		//     다시 받으면 소리 없이 재생되기 때문 (새로고침하면 다시 시도)
		const go = synth.go;
		synth.go = () => {
			if (this.soundFailed) {
				this.handleSynthFailure();
				return Promise.reject(new Error("악보 소리 파일을 받지 못함"));
			}
			return go()
				.then((result) => {
					const status = result?.notesStatus;
					const count = (status?.loaded?.length ?? 0) + (status?.cached?.length ?? 0);
					if (count === 0) throw new Error("악보 소리 파일을 받지 못함");
					return result;
				})
				.catch((error) => {
					this.soundFailed = true;
					this.handleSynthFailure();
					throw error; // 뒤에 이어지는 재생 동작은 멈춤
				});
		};

		// 버튼에 연결되는 함수들: 실패 처리는 go에서 이미 했으므로, 여기서는 오류가 밖으로 새지 않게만 막음
		// (load() 안에서 버튼에 연결하므로 load 전에 바꿔야 함)
		["play", "randomAccess", "onWarp"].forEach((name) => {
			const original = synth[name];
			synth[name] = (...args) => original(...args).catch((error) => {
				console.warn(error);
				return { status: "error" };
			});
		});

		synth.load(this.audioArea, this.createCursorControl(), {
			displayPlay: true,
			displayRestart: true,
			displayProgress: true,
			displayWarp: true, // 재생 속도 조절
		});

		// 버튼 안내 문구 한국어로
		Object.entries(SYNTH_TITLES).forEach(([selector, title]) => {
			const el = this.audioArea.querySelector(selector);
			if (!el) return;
			el.title = title;
			el.setAttribute("aria-label", title);
		});
		return synth;
	}

	handleSynthFailure() {
		// 재생 준비 상태 되돌리기 (abcjs는 실패해도 "준비 중"으로 남아 버튼이 멈춤)
		this.synthControl.isLoading = false;
		this.synthControl.isLoaded = false;
		this.synthControl.isStarted = false;
		this.audioArea.querySelectorAll(".abcjs-pushed, .abcjs-loading").forEach((el) => {
			el.classList.remove("abcjs-pushed", "abcjs-loading");
		});
		this.clearHighlight();
		this.setMessage("악보 소리를 불러오지 못했어요(인터넷 연결 필요). 새로고침하면 다시 시도해요. 악보 보기는 그대로 쓸 수 있어요.");
	}

	stopSynth() {
		try {
			this.synthControl?.pause();
		} catch (error) {
			console.warn(error);
		}
		this.clearHighlight();
	}

	// 재생 중인 음표 칠하기 + 세로선 (abcjs CursorControl 규칙)
	createCursorControl() {
		const view = this;
		return {
			beatSubdivisions: 4,
			onStart() {
				const svg = view.sheetArea.querySelector("svg");
				if (!svg || svg.querySelector(".abcjs-cursor")) return;
				const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
				line.setAttribute("class", "abcjs-cursor");
				svg.appendChild(line);
			},
			onEvent(event) {
				if (event.measureStart && event.left === null) return;
				view.clearHighlight();
				event.elements.forEach((group) => group.forEach((el) => el.classList.add("abcjs-highlight")));
				const cursor = view.sheetArea.querySelector(".abcjs-cursor");
				if (cursor) {
					cursor.setAttribute("x1", event.left - 2);
					cursor.setAttribute("x2", event.left - 2);
					cursor.setAttribute("y1", event.top);
					cursor.setAttribute("y2", event.top + event.height);
				}
			},
			onFinished() {
				view.clearHighlight();
			},
		};
	}

	clearHighlight() {
		this.sheetArea.querySelectorAll(".abcjs-highlight").forEach((el) => el.classList.remove("abcjs-highlight"));
		const cursor = this.sheetArea.querySelector(".abcjs-cursor");
		if (cursor) {
			["x1", "x2", "y1", "y2"].forEach((attr) => cursor.setAttribute(attr, 0));
		}
	}
}
