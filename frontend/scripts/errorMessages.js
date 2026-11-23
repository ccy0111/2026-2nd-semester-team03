// ==========================================================
// errorMessages
// - 팀 모델 코드가 던지는 영어 오류를 사용자용 한국어 안내로 바꿈
// - 원래 오류는 개발자 도구(Console)에 그대로 남음 (App.js의 console.error)
// ==========================================================

// [오류 문구에 포함된 글자, 한국어 안내]
const MESSAGE_MAP = [
	["Unable to read a compatible model manifest", "모델 정보를 불러오지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요."],
	["Manifest request timed out", "모델 정보 요청이 너무 오래 걸려요. 잠시 후 다시 시도해 주세요."],
	["Manifest request failed", "모델 정보를 불러오지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요."],
	["Unable to load the new model", "모델 파일을 내려받지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요."],
	["Failed to fetch", "모델 파일을 내려받지 못했어요. 인터넷 연결을 확인하고 다시 시도해 주세요."],
	["Failed to load", "모델 파일을 불러오지 못했어요. 잠시 후 다시 시도해 주세요."],
	["importScripts", "모델 실행 도구(onnxruntime-web)를 불러오지 못했어요. 인터넷 연결을 확인해 주세요."],
	["dynamically imported module", "모델 코드를 불러오지 못했어요. 서버가 켜져 있는지 확인해 주세요."],
	["A transcription is already running", "이미 채보가 진행 중이에요. 끝날 때까지 기다려 주세요."],
	["No model is loaded", "모델이 아직 준비되지 않았어요. 모델을 다시 불러와 주세요."],
	["Web AudioContext unavailable", "이 브라우저는 음원 분석을 지원하지 않아요. 최신 크롬에서 열어 주세요."],
];

export function toUserMessage(error) {
	const original = error?.message ?? String(error);
	const found = MESSAGE_MAP.find(([key]) => original.includes(key));
	return found ? found[1] : original; // 모르는 오류는 원문 그대로
}
