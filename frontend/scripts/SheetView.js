// ==========================================================
// SheetView
// - "악보" 탭 화면을 담당하는 class
//   악기 선택 → 그 악기 음표만 골라 → 오선보(abcjs) 또는 타브(TabRenderer)로 그림
// - 템포(BPM)·박자 입력, 기타 표시 방식(오선보/타브/오선보+타브) 선택
// - 오선보는 abcjs의 재생 기능으로 악보를 소리로 듣고, 재생 중인 음표를 표시
//
// abcjs: 오선 악보 라이브러리 (MIT). index.html에서 <script>로 불러와 window.ABCJS로 사용
// ==========================================================
import { AbcBuilder, SHEET_KEYS } from "./AbcBuilder.js";
import { TabRenderer } from "./TabRenderer.js";
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

		this.tabRenderer = new TabRenderer(this.sheetArea);
		this.notes = [];
		this.startSec = 0;
		this.isVisible = false;
		this.needsRender = false;
		this.synthControl = null;
		this.mode = null; // 지금 그려진 방식: "abc" | "tab"
		this.bpm = DEFAULT_BPM;  // 마지막으로 쓴 올바른 템포
		this.soundFailed = false; // 악보 소리 파일 받기에 실패했는지
		this.originalTime = 0;   // 원본 MP3의 마지막 재생 위치 (타브를 다시 그려도 커서 유지)

		// 설정을 바꾸면 다시 그림
		[this.instrumentSelect, this.guitarModeSelect, this.meterSelect].forEach((el) => {
			el.addEventListener("change", () => this.render());
		});
		this.bpmInput.addEventListener("change", () => this.render());
	}

	// 채보 결과 받기 (instruments: InstrumentLegend와 같은 목록)
	setResult(notes, instruments) {
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
		this.tabRenderer.clear();
		this.setMessage("");
		this.mode = null;
		this.needsRender = false;
		this.originalTime = 0;
	}

	// 원본 MP3 재생 위치 (타브 보기에서만 사용: 타브는 시간 기준이라 원본과 맞출 수 있음)
	// follow: 재생 위치를 따라 가로 스크롤할지 (AudioPlayer가 정해 줌)
	setOriginalTime(seconds, follow = true) {
		this.originalTime = seconds;
		if (this.mode === "tab") this.tabRenderer.setCursor(seconds, follow);
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

		// 기타 타브만 보기: 직접 그리는 시간 기준 타브
		if (guitarMode === "tab") {
			this.mode = "tab";
			this.audioArea.classList.add("is-hidden");
			this.tabRenderer.render(notes);
			this.tabRenderer.setCursor(this.originalTime, false); // 다시 그려도 재생 위치 유지
			this.setMessage("타브는 시간 기준으로 그려요(리듬 표시 없음). 위쪽 원본 재생기로 들으면 재생 위치가 표시돼요.");
			return;
		}

		// 오선보 (abcjs)
		const ABCJS = window.ABCJS;
		if (!ABCJS) {
			this.mode = null;
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
			this.mode = null;
			this.sheetArea.innerHTML = "";
			this.audioArea.classList.add("is-hidden");
			this.setMessage("악보를 그리지 못했어요.");
			return;
		}
		this.mode = "abc";
		if (visualObj?.warnings?.length) console.warn("abcjs warnings", visualObj.warnings);

		let message = `템포 ${bpm} BPM 기준으로 16분음표 단위로 정리한 악보예요. 템포가 맞지 않으면 직접 고쳐 보세요.`;
		if (key !== SHEET_KEYS.DRUM_KEY) message += " 같은 손에서 겹치는 음은 다음 음이 시작될 때 끊어서 표시해요.";
		this.setMessage(message);
		this.setupSynth(ABCJS, visualObj);
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
