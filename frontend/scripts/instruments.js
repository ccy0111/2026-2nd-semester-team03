// ==========================================================
// instruments
// - 음표의 악기 번호(program)를 "악기 계열" 이름과 색으로 바꿔 주는 도우미
//
// program은 General MIDI(GM) 표준 악기 번호 0~127
//   8개씩 한 계열: 0~7 피아노, 8~15 크로매틱 타악기, 16~23 오르간 ...
//   그래서 Math.floor(program / 8)로 계열을 알 수 있음
// 드럼은 program 대신 isDrum: true 로 옴 (팀 모델 코드에서는 program 128)
// ==========================================================

// [계열 이름, 색] — 순서가 곧 계열 번호(0~15)
const FAMILIES = [
	["피아노", "#4c6ef5"],
	["크로매틱 타악기", "#15aabf"],   // 실로폰, 비브라폰, 오르골 등
	["오르간", "#7950f2"],
	["기타(Guitar)", "#f76707"],
	["베이스", "#2b8a3e"],
	["현악기", "#e64980"],            // 바이올린, 첼로 등
	["앙상블", "#d6336c"],            // 현악 합주, 합창 등
	["금관악기", "#f59f00"],          // 트럼펫, 트롬본 등
	["리드 목관악기", "#82c91e"],     // 색소폰, 클라리넷 등
	["파이프 목관악기", "#20c997"],   // 플루트, 리코더 등
	["신스 리드", "#be4bdb"],
	["신스 패드", "#9775fa"],
	["신스 효과", "#748ffc"],
	["민속 악기", "#a9844d"],
	["타악기류", "#868e96"],          // 팀파니, 스틸드럼 등 음높이 있는 타악기
	["효과음", "#495057"],
];

const DRUMS = { key: "drums", name: "드럼·타악기", color: "#212529" };
const OTHER = { key: "other", name: "그 밖의 악기", color: "#adb5bd" };

// note → { key, name, color }
export function getInstrument(note) {
	if (note.isDrum) return DRUMS;
	const program = note.program;
	if (!Number.isInteger(program) || program < 0 || program > 127) return OTHER;
	const index = Math.floor(program / 8);
	const [name, color] = FAMILIES[index];
	return { key: `family-${index}`, name, color };
}

// 목록을 보여 줄 순서 (피아노 → … → 효과음 → 드럼 → 그 밖)
export function instrumentOrder(key) {
	if (key === DRUMS.key) return 100;
	if (key === OTHER.key) return 101;
	return Number(key.replace("family-", ""));
}
