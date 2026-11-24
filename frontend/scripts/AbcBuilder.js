// ==========================================================
// AbcBuilder
// - 모델의 notes(초 단위)를 ABC 악보 글자로 바꾸는 class
//   ABC: 글자로 적는 악보 형식. abcjs 라이브러리가 이 글자를 오선 악보로 그려 줌
//   예) "C2 D2 E4 |"  → 도(8분) 레(8분) 미(4분) 마디선
//
// 처리 순서
//   1. 거의 동시에(0.04초 안에) 시작한 음들은 시작 시각을 가장 이른 음에 맞춤
//   2. 템포(BPM)로 초 → 칸(tick, 16분음표 1칸) 변환 = "박자 맞추기(양자화)"
//   3. 같은 칸에서 시작하는 음은 화음 [CEG]으로 묶음
//   4. 빈 곳은 쉼표, 마디를 넘는 음은 둘로 나누고 붙임줄(-)로 연결
//   5. 한 박 안의 음표는 붙여 써서 꼬리를 빔(가로줄)으로 묶음
//   6. 악기에 맞는 음자리표 (피아노 큰보표, 베이스·기타 옥타브 표기, 드럼 보표)
// ==========================================================

const TICKS_PER_BEAT = 4;                       // 4분음표 1박 = 16분음표 4칸
const NOTE_LENGTHS = [16, 12, 8, 6, 4, 3, 2, 1]; // 악보에 쓸 수 있는 길이(칸): 온·점2·2·점4·4·점8·8·16분
const BARS_PER_LINE = 4;                        // 한 줄에 4마디
const SAME_ONSET = 0.04;                        // 무리의 첫 음에서 0.04초 안에 시작하면 같은 순간에 친 음으로 봄

// 템포 추정 설정
const MIN_BPM = 60;
const MAX_BPM = 180;
const PHASES = 8;            // 격자 시작점을 8가지로 밀어 보며 가장 잘 맞는 것을 씀
const FOLD_MIN = 70;         // 최종 템포는 70~140 안으로 (2배/절반으로 접어 넣음)
const FOLD_MAX = 140;

// 반음(0~11) → [음이름, 샵 여부]
const PITCH_CLASSES = [
	["C", false], ["C", true], ["D", false], ["D", true], ["E", false], ["F", false],
	["F", true], ["G", false], ["G", true], ["A", false], ["A", true], ["B", false],
];

// 드럼 번호(GM) → 드럼 보표 위치 (표준 드럼 악보 배치를 단순화)
const DRUM_POSITIONS = {
	35: "F", 36: "F",                     // 베이스 드럼(킥) → 아래 칸
	38: "c", 40: "c", 37: "c", 39: "c",   // 스네어·림샷·박수 → 셋째 칸
	41: "A", 43: "A", 45: "B", 47: "d", 48: "e", 50: "e", // 탐 (낮은 → 높은)
	42: "g", 44: "D", 46: "g",            // 하이햇 (닫힘·페달·열림)
	49: "a", 57: "a", 55: "a", 52: "a",   // 크래시·스플래시·차이나
	51: "f", 59: "f", 53: "f",            // 라이드
};
const DRUM_DEFAULT = "c";       // 모르는 드럼 번호는 스네어 자리에 표시
const DRUM_DEFAULT_SOUND = 38;  // 그 자리에 쓸 소리 (스네어)
// 음표 머리를 x 모양으로 그리는 드럼 (하이햇·심벌 = 악보 관례)
const DRUM_X_HEADS = new Set([42, 44, 46, 49, 51, 52, 53, 55, 57, 59]);

// 악기 계열 key → 악보 설정 (instruments.js의 key와 같음)
const GUITAR_KEY = "family-3";
const BASS_KEY = "family-4";
const PIANO_KEY = "family-0";
const DRUM_KEY = "drums";

export class AbcBuilder {
	// ---------- 템포 추정 ----------
	// 1) 60~180 BPM마다 "음 시작 시각이 16분음표 격자에 얼마나 잘 맞나"를 잼
	//    첫 음이 박자에서 조금 어긋나 있어도 되도록 격자 시작점을 8가지로 밀어 보고 가장 좋은 값을 씀
	// 2) 가장 잘 맞는 템포 b를 찾고, b의 절반·2배 중에서 100 BPM에 가장 가까운 것을 고름
	//    (120 BPM 곡은 60 BPM 격자에도 맞기 때문에, "같은 곡의 2배/절반 템포" 중 읽기 쉬운 쪽을 고르는 것)
	// 3) 박자가 조금 흔들리는 곡(잘 안 맞는 경우)도 120으로 버리지 않고, 찾은 값을 70~140 안으로 접어서 씀
	static estimateBpm(notes) {
		const onsets = [...new Set(notes.map((note) => Math.round(note.onset * 100) / 100))]
			.filter(Number.isFinite)
			.sort((a, b) => a - b)
			// 앞쪽 2,000개만 사용: 긴 곡도 템포를 재는 데는 충분하고, 결과가 나올 때 화면이 멈추지 않게
			.slice(0, 2000);
		if (onsets.length < 4) return 120;

		const errors = new Map(); // bpm → 어긋난 정도 (0 = 딱 맞음, 0.25 = 아무렇게나 친 것과 비슷)
		let best = MIN_BPM;
		for (let bpm = MIN_BPM; bpm <= MAX_BPM; bpm += 1) {
			errors.set(bpm, AbcBuilder.gridError(onsets, bpm));
			if (errors.get(bpm) < errors.get(best)) best = bpm; // 같으면 느린 쪽 유지
		}

		// 2배 템포는 격자가 더 촘촘할 뿐 항상 맞으므로 후보,
		// 절반 템포는 실제로 거의 똑같이 잘 맞을 때만 후보
		const candidates = [best, best * 2];
		const half = Math.round(best / 2);
		if (half >= MIN_BPM && errors.get(half) <= errors.get(best) * 1.25 + 0.01) candidates.push(half);

		// 100에 가까운 정도는 "비율"로 비교 (70과 140은 100에서 각각 1.43배, 1.4배 떨어짐)
		const distance = (bpm) => Math.max(bpm / 100, 100 / bpm);
		let chosen = candidates
			.filter((bpm) => bpm >= MIN_BPM && bpm <= MAX_BPM)
			.reduce((a, b) => (distance(b) < distance(a) ? b : a));

		// 70~140 안으로 접기 (같은 곡의 2배/절반 템포라 악보 모양만 달라짐)
		while (chosen > FOLD_MAX) chosen /= 2;
		while (chosen < FOLD_MIN) chosen *= 2;
		return Math.round(chosen);
	}

	// 시작 시각들이 이 템포의 16분음표 격자에서 평균적으로 얼마나 벗어났나 (격자 한 칸 = 1)
	static gridError(onsets, bpm) {
		const grid = 60 / bpm / TICKS_PER_BEAT; // 16분음표 길이(초)
		const t0 = onsets[0];
		let best = Infinity;
		for (let k = 0; k < PHASES; k++) {
			const shift = k / PHASES; // 격자 시작점 밀기 (칸 단위)
			let sum = 0;
			for (const t of onsets) {
				const x = (t - t0) / grid - shift; // 격자 몇 칸째인지 (소수)
				const r = x - Math.floor(x);       // 0~1: 칸 안에서의 위치
				sum += Math.min(r, 1 - r);         // 가까운 격자선까지 거리
			}
			best = Math.min(best, sum / onsets.length);
		}
		return best;
	}

	// ---------- ABC 만들기 ----------
	// options: { notes, instrumentKey, instrumentName, bpm, beatsPerBar, startSec }
	// 반환: { abc, tablature } (tablature는 기타 오선보+타브일 때 abcjs에 넘길 설정)
	static build({ notes, instrumentKey, instrumentName = "", bpm, beatsPerBar = 4, startSec = 0, withTab = false }) {
		const secPerTick = 60 / bpm / TICKS_PER_BEAT;
		const barTicks = beatsPerBar * TICKS_PER_BEAT;

		// 초 → 칸 (곡 첫 음을 0칸으로)
		const toTick = (sec) => Math.max(0, Math.round((sec - startSec) / secPerTick));
		const events = AbcBuilder.clusterOnsets(notes).map(({ note, onset }) => {
			const start = toTick(onset);
			const end = Math.max(start + 1, toTick(note.offset));
			return { start, end, pitch: note.pitch };
		});

		// 마지막 칸 (음표가 아주 많아도 되도록 Math.max(...배열) 대신 반복문)
		let lastTick = barTicks;
		events.forEach((e) => { lastTick = Math.max(lastTick, e.end); });
		const totalBars = Math.ceil(lastTick / barTicks);

		// 악기별 성부(voice) 나누기
		let voices;
		if (instrumentKey === PIANO_KEY) {
			voices = [
				{ id: "RH", clef: "treble", events: events.filter((e) => e.pitch >= 60) },
				{ id: "LH", clef: "bass", events: events.filter((e) => e.pitch < 60) },
			];
		} else if (instrumentKey === DRUM_KEY) {
			voices = [{ id: "DR", clef: "perc", events, drums: true }];
		} else if (instrumentKey === GUITAR_KEY) {
			// 기타 악보는 실제 소리보다 한 옥타브 높게 적는 것이 표준 (abcjs 타브도 이 기준)
			// transpose=-12: 적힌 음보다 한 옥타브 낮게 재생 → 실제 기타 소리 높이로 들림
			voices = [{ id: "GT", clef: "treble", transpose: -12, events: AbcBuilder.octaveUp(events) }];
		} else if (instrumentKey === BASS_KEY) {
			// 베이스도 기타처럼 한 옥타브 높게 적고(덧줄이 줄어듦), 재생은 한 옥타브 낮게
			voices = [{ id: "BS", clef: "bass", transpose: -12, events: AbcBuilder.octaveUp(events) }];
		} else {
			// 그 외: 음들의 가운데 값이 가운데 도(60) 이상이면 높은음자리표
			const pitches = events.map((e) => e.pitch).sort((a, b) => a - b);
			const median = pitches[Math.floor(pitches.length / 2)] ?? 60;
			voices = [{ id: "V1", clef: median >= 57 ? "treble" : "bass", events }];
		}

		// 머리말
		const lines = [
			"X:1",
			`T:${AbcBuilder.escapeText(instrumentName)}`,
			`M:${beatsPerBar}/4`,
			"L:1/16",
			`Q:1/4=${Math.round(bpm)}`,
		];
		if (voices.length === 2) lines.push("%%score {RH | LH}");
		// 드럼: 보표 위치마다 어떤 드럼 소리를 낼지 (%%percmap은 K: 줄보다 앞에 있어야 함)
		if (instrumentKey === DRUM_KEY) lines.push(...AbcBuilder.percmapLines(events));
		lines.push("K:C");

		// 성부마다 마디 목록 만들기
		const voiceBars = voices.map((voice) => AbcBuilder.buildBars(voice, barTicks, totalBars));

		// 4마디씩 줄을 나눠서, 줄마다 성부를 번갈아 적음 (큰보표가 위아래로 맞도록)
		for (let bar = 0; bar < totalBars; bar += BARS_PER_LINE) {
			voices.forEach((voice, i) => {
				let header = `V:${voice.id}`;
				if (bar === 0) {
					header += ` clef=${voice.clef}`;
					if (voice.drums) header += " stafflines=5";
					if (voice.transpose) header += ` transpose=${voice.transpose}`;
				}
				lines.push(header);
				if (bar === 0) {
					// 재생할 때 쓸 악기 소리
					lines.push(voice.drums ? "%%MIDI channel 10" : `%%MIDI program ${AbcBuilder.programFor(instrumentKey)}`);
				}
				const chunk = voiceBars[i].slice(bar, bar + BARS_PER_LINE);
				const isLast = bar + BARS_PER_LINE >= totalBars;
				lines.push(chunk.join(" | ") + (isLast ? " |]" : " |"));
			});
		}

		return {
			abc: lines.join("\n"),
			tablature: withTab && instrumentKey === GUITAR_KEY
				? [{ instrument: "guitar", label: "Guitar" }]
				: undefined,
		};
	}

	// 거의 동시에 시작한 음들의 시작 시각을 하나로 맞추기
	// 모델의 시작 시각은 10~30ms씩 흔들려서, 그대로 칸에 맞추면 한 화음이 두 칸으로 갈라질 수 있음
	// → 시간순으로 보며 무리의 "첫 음"과 0.04초 안이면 같은 무리, 무리의 가장 이른 시각을 함께 씀
	//   - 첫 음과만 비교: 빠른 트릴처럼 계속 이어지는 음이 하나의 큰 화음으로 뭉치지 않게
	//   - 가장 이른 시각: 아르페지오처럼 굴려 친 화음도 박 위(첫 음 위치)에서 시작하게
	// 반환: [{ note, onset }] (onset = 무리의 첫 음 시작 시각)
	static clusterOnsets(notes) {
		const sorted = [...notes].sort((a, b) => a.onset - b.onset);
		const result = [];
		let cluster = [];
		const flush = () => {
			const first = cluster[0].onset; // 정렬되어 있으므로 첫 음이 가장 이름
			cluster.forEach((note) => result.push({ note, onset: first }));
			cluster = [];
		};
		sorted.forEach((note) => {
			if (cluster.length && note.onset - cluster[0].onset > SAME_ONSET) flush();
			cluster.push(note);
		});
		if (cluster.length) flush();
		return result;
	}

	// 한 옥타브(12반음) 높게 적기
	static octaveUp(events) {
		return events.map((e) => ({ ...e, pitch: e.pitch + 12 }));
	}

	// 드럼 보표 위치 → 소리(GM 드럼 번호) 연결 줄 만들기
	// 예) "%%percmap g 42 x" = g 자리의 음표는 닫힌 하이햇 소리, 머리는 x 모양
	// 한 자리에는 소리를 하나만 정할 수 있어서, 그 자리에 가장 많이 나온 드럼 번호를 씀
	// (모르는 드럼 번호는 스네어 자리에 그리고 스네어 소리로 재생)
	static percmapLines(events) {
		const counts = new Map(); // 자리 → Map(드럼 번호 → 개수)
		events.forEach((e) => {
			const known = DRUM_POSITIONS[e.pitch] !== undefined;
			const position = known ? DRUM_POSITIONS[e.pitch] : DRUM_DEFAULT;
			if (!counts.has(position)) counts.set(position, new Map());
			if (known) counts.get(position).set(e.pitch, (counts.get(position).get(e.pitch) ?? 0) + 1);
		});
		const lines = [];
		counts.forEach((keyCounts, position) => {
			let sound = DRUM_DEFAULT_SOUND;
			let most = 0;
			keyCounts.forEach((count, key) => {
				if (count > most) {
					most = count;
					sound = key;
				}
			});
			lines.push(`%%percmap ${position} ${sound}${DRUM_X_HEADS.has(sound) ? " x" : ""}`);
		});
		return lines;
	}

	// 한 성부의 사건(음/쉼표)을 마디별 ABC 글자 배열로
	static buildBars(voice, barTicks, totalBars) {
		// 1) 같은 칸에서 시작하는 음을 화음으로 묶기
		const groups = new Map(); // start → { pitches:Set, end }
		voice.events.forEach((e) => {
			const group = groups.get(e.start) ?? { pitches: new Set(), end: e.end };
			group.pitches.add(e.pitch);
			group.end = Math.max(group.end, e.end);
			groups.set(e.start, group);
		});
		const starts = [...groups.keys()].sort((a, b) => a - b);

		// 2) 시간 순서대로 [쉼표 / 음] 조각 만들기
		const totalTicks = totalBars * barTicks;
		const pieces = []; // { pitches: number[] | null(쉼표), length }
		let cursor = 0;
		starts.forEach((start, i) => {
			if (start >= totalTicks) return;
			if (start > cursor) pieces.push({ pitches: null, length: start - cursor });
			const next = starts[i + 1] ?? totalTicks;
			const group = groups.get(start);
			// 다음 음이 시작되면 이전 음은 끊음 (한 성부에서는 겹치지 않게)
			let end = Math.min(group.end, next);
			if (voice.drums) end = Math.min(next, start + TICKS_PER_BEAT); // 드럼은 다음 타격까지(최대 1박)
			pieces.push({ pitches: [...group.pitches].sort((a, b) => a - b), length: Math.max(1, end - start) });
			cursor = start + Math.max(1, end - start);
		});
		if (cursor < totalTicks) pieces.push({ pitches: null, length: totalTicks - cursor });

		// 3) 마디선에 맞춰 자르고, 쓸 수 있는 길이로 쪼개기
		//    한 박 안의 음표는 띄어쓰기 없이 붙여 씀 → abcjs가 꼬리를 빔(가로줄)으로 묶어 줌
		//    박이 시작되는 곳에서만 띄어 써서 빔을 박 단위로 끊음 (예: "C2D2 E2F2 | ...")
		const bars = Array.from({ length: totalBars }, () => []);
		let barIndex = 0;
		let posInBar = 0;
		let accidentals = new Map(); // 이 마디에서 이미 붙인 임시표 (ABC는 마디 끝까지 유지됨)

		pieces.forEach((piece) => {
			let remaining = piece.length;
			while (remaining > 0 && barIndex < totalBars) {
				const take = Math.min(remaining, barTicks - posInBar);
				const lengths = AbcBuilder.splitLength(take, posInBar);
				let pos = posInBar; // 마디 안에서 이 조각이 시작하는 칸
				let tiedMarks = null; // 이 마디 안 "이어진 뒷부분"들이 함께 쓰는 임시표 기록 복사본
				lengths.forEach((length, k) => {
					const isLastPiece = remaining - take === 0 && k === lengths.length - 1;
					const isFirstPiece = remaining === piece.length && k === 0;
					let token;
					if (!piece.pitches || (voice.drums && !isFirstPiece)) {
						// 쉼표, 또는 드럼의 남은 길이 (드럼은 붙임줄 없이 한 번만 침)
						token = `z${AbcBuilder.lengthText(length)}`;
					} else {
						// 붙임줄로 이어진 뒷부분은 임시표 기록을 바꾸면 안 됨 → 복사본을 넘김
						// (abcjs는 붙임줄 뒤 음의 ^를 마디 기록에 넣지 않아서, 같은 마디의 다음 C#이 C로 재생되던 문제)
						// 같은 마디의 뒷부분들은 복사본 하나를 같이 써서 임시표를 반복해 붙이지 않음 (예: C3- C8- C2)
						if (!isFirstPiece && !tiedMarks) tiedMarks = new Map(accidentals);
						const marks = isFirstPiece ? accidentals : tiedMarks;
						token = AbcBuilder.chordToken(piece.pitches, length, marks, voice.drums)
							+ (isLastPiece || voice.drums ? "" : "-"); // 이어지는 음은 붙임줄
						if (!isFirstPiece && !voice.drums && pos === 0) {
							// 마디선을 넘어 새 마디 첫머리에서 이어진 음이름은 이 마디에서 "모름(null)"으로 기록
							// → 다음에 같은 줄·칸의 음이 오면 항상 ^ 또는 =를 붙여서, 읽는 사람도 헷갈리지 않게
							piece.pitches.forEach((pitch) => accidentals.set(AbcBuilder.noteName(pitch).name, null));
						}
					}
					const atBeat = pos % TICKS_PER_BEAT === 0;
					bars[barIndex].push(atBeat && bars[barIndex].length > 0 ? ` ${token}` : token);
					pos += length;
				});
				remaining -= take;
				posInBar += take;
				if (posInBar >= barTicks) {
					barIndex += 1;
					posInBar = 0;
					accidentals = new Map(); // 새 마디: 임시표 초기화
				}
			}
		});

		return bars.map((tokens) => tokens.join("") || `z${AbcBuilder.lengthText(barTicks)}`);
	}

	// 길이(칸)를 악보에 쓸 수 있는 길이들의 합으로 (예: 5 → 4 + 1)
	// 박 위치(posInBar)가 박 중간이면 박의 경계까지 먼저 채워서 읽기 쉽게
	static splitLength(length, posInBar) {
		const result = [];
		let remaining = length;
		let pos = posInBar;
		while (remaining > 0) {
			const toBeat = TICKS_PER_BEAT - (pos % TICKS_PER_BEAT);
			let piece;
			if (pos % TICKS_PER_BEAT !== 0 && remaining > toBeat) {
				piece = toBeat; // 박 중간에서 시작하면 우선 박까지
			} else {
				piece = NOTE_LENGTHS.find((n) => n <= remaining);
			}
			result.push(piece);
			remaining -= piece;
			pos += piece;
		}
		return result;
	}

	// 음(들) → ABC 글자 (예: [CEG]4, ^F2)
	static chordToken(pitches, length, accidentals, drums) {
		const names = pitches.map((pitch) => (drums
			? (DRUM_POSITIONS[pitch] ?? DRUM_DEFAULT)
			: AbcBuilder.pitchName(pitch, accidentals)));
		const unique = [...new Set(names)];
		const body = unique.length > 1 ? `[${unique.join("")}]` : unique[0];
		return body + AbcBuilder.lengthText(length);
	}

	// MIDI 음높이 → 임시표 없는 ABC 음이름 + 샵 여부 (C4=C, C5=c, C6=c', C3=C,)
	static noteName(pitch) {
		const p = Math.round(pitch);
		const [letter, sharp] = PITCH_CLASSES[((p % 12) + 12) % 12];
		const octave = Math.floor(p / 12) - 1;

		let name;
		if (octave >= 5) name = letter.toLowerCase() + "'".repeat(octave - 5);
		else name = letter + ",".repeat(Math.max(0, 4 - octave));
		return { name, sharp };
	}

	// MIDI 음높이 → 임시표를 붙인 ABC 음이름 (예: ^C, =C, C)
	// ABC에서 임시표는 그 마디 끝까지 유지되므로, 상태가 바뀔 때만 ^(샵) / =(제자리표)를 붙임
	static pitchName(pitch, accidentals) {
		const { name, sharp } = AbcBuilder.noteName(pitch);

		// 같은 줄·칸(name)마다 지금 샵 상태인지 기록 (true 샵 / false 제자리 / null 모름)
		const current = accidentals.has(name) ? accidentals.get(name) : false; // 기본은 제자리
		let prefix = "";
		if (current === null) prefix = sharp ? "^" : "="; // 모르면 항상 적어 줌
		else if (sharp && !current) prefix = "^";
		else if (!sharp && current) prefix = "=";
		accidentals.set(name, sharp);
		return prefix + name;
	}

	static lengthText(length) {
		return length === 1 ? "" : String(length);
	}

	// 재생용 악기 소리: 계열의 첫 번째 GM 악기 (예: 현악기 계열 → 40 바이올린)
	static programFor(instrumentKey) {
		if (instrumentKey?.startsWith("family-")) return Number(instrumentKey.replace("family-", "")) * 8;
		return 0;
	}

	// 제목 줄에 들어갈 글자 정리 (줄바꿈·% 제거)
	static escapeText(text) {
		return String(text).replace(/[\r\n%]/g, " ");
	}
}

export const SHEET_KEYS = { GUITAR_KEY, BASS_KEY, PIANO_KEY, DRUM_KEY };
