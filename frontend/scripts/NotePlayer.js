// ==========================================================
// NotePlayer
// - 채보 결과(notes)를 실제 악기 소리로 들려주는 class (피아노롤 보기에서 사용)
// - 팀 재생 모듈(soundfont-playback, docs/API.md "채보 결과 재생")을 감싸서 사용
//   frontend/vendor/soundfont/dist/grouped-player.mjs  : 재생기 (빌드된 파일)
//   frontend/vendor/soundfont/soundfonts.json           : 악기 소리 파일(SF2) 목록 (CDN 주소)
// - 악기별 음소거: 재생 모듈에는 악기별 음소거가 없어서,
//   소리를 끈 악기를 뺀 음표로 재생기를 새로 만들고 같은 위치부터 이어서 재생함
//   (소리 파일은 브라우저에 저장돼 있어서 두 번째부터는 다시 받지 않음)
// ==========================================================
import { getInstrument } from "./instruments.js";

// 이 파일(scripts/) 기준으로 vendor/soundfont/ 위치 계산 → 서버 경로가 바뀌어도 동작
const SOUNDFONT_BASE = new URL("../vendor/soundfont/", import.meta.url);

export class NotePlayer {
	// onTimeUpdate(seconds, follow): 재생 위치가 바뀔 때 (피아노롤 커서 이동용)
	// onStateChange(): 재생/정지/준비 상태가 바뀔 때 (버튼 글자 바꾸기용)
	// onMessage(text, type): 안내 문구 ("info" | "error")
	constructor({ onTimeUpdate, onStateChange, onMessage }) {
		this.onTimeUpdate = onTimeUpdate;
		this.onStateChange = onStateChange;
		this.onMessage = onMessage;

		this.notes = [];
		this.mutedKeys = new Set(); // 소리를 끈 악기 key (instruments.js의 key)
		this.context = null;        // AudioContext (처음 재생 버튼을 누를 때 만듦)
		this.output = null;         // 전체 음소거용 GainNode
		this.isMuted = false;
		this.player = null;
		this.isPreparing = false;
		this.position = 0;          // 재생기가 없을 때 기억해 둘 재생 위치(초)
		this.session = 0;           // 재생기를 새로 만들 때마다 +1 (이전 준비 작업 무시용)
		this.frameId = null;
		this.modulePromise = null;
		this.banksPromise = null;
	}

	get isPlaying() {
		return Boolean(this.player && !this.player.paused);
	}

	// 곡 길이(초): 재생기가 준비되기 전에도 음표로 계산
	get duration() {
		if (this.player?.duration) return this.player.duration;
		let end = 0;
		this.notes.forEach((note) => { end = Math.max(end, note.offset); });
		return end;
	}

	get currentTime() {
		return this.player ? this.player.currentTime : this.position;
	}

	// 새 채보 결과: 이전 재생기 정리
	setNotes(notes) {
		this.disposePlayer();
		this.notes = notes;
		this.mutedKeys.clear();
		this.position = 0;
		this.onStateChange?.();
	}

	clear() {
		this.setNotes([]);
	}

	// 재생 버튼: 재생 ↔ 일시정지
	async toggle() {
		if (this.isPlaying) {
			this.pause();
			return;
		}
		await this.play();
	}

	async play() {
		if (this.isPreparing) return;
		if (!this.player) {
			const ready = await this.prepare();
			if (!ready) return;
		}
		// 끝까지 재생한 뒤 다시 누르면 처음부터
		if (this.player.currentTime >= this.player.duration - 0.05) this.player.seek(0);
		await this.player.play();
		this.startLoop();
		this.onStateChange?.();
	}

	pause() {
		if (!this.player) return;
		this.player.pause();
		this.stopLoop();
		this.onStateChange?.();
	}

	// 지정한 위치(초)로 이동. play=true면 그 위치부터 재생
	async seek(seconds, play = false) {
		const time = Math.min(Math.max(0, seconds), this.duration);
		if (this.player) {
			this.player.seek(time);
		} else {
			this.position = time;
		}
		this.onTimeUpdate?.(time, true);
		if (play && !this.isPlaying) await this.play();
	}

	// 전체 음소거 (재생 위치는 그대로)
	setMuted(muted) {
		this.isMuted = muted;
		if (this.output) this.output.gain.value = muted ? 0 : 1;
		this.onStateChange?.();
	}

	// 악기 하나 소리 켜기/끄기
	async setInstrumentMuted(key, muted) {
		if (muted) this.mutedKeys.add(key);
		else this.mutedKeys.delete(key);
		if (!this.player && !this.isPreparing) return; // 아직 재생 전이면 다음 재생 때 반영

		// 지금 위치·재생 여부를 기억하고 재생기를 새로 만듦
		const wasPlaying = this.isPlaying || this.isPreparing;
		const time = this.currentTime;
		this.disposePlayer();
		this.position = time;
		if (wasPlaying) await this.play();
		else this.onStateChange?.();
	}

	// 페이지를 떠날 때
	dispose() {
		this.disposePlayer();
		this.context?.close().catch(() => {});
		this.context = null;
	}

	// ---------- 내부 동작 ----------
	// 재생기 만들기 + 필요한 악기 소리 파일 받기. 성공하면 true
	async prepare() {
		const notes = this.notes.filter((note) => !this.mutedKeys.has(getInstrument(note).key));
		if (notes.length === 0) {
			this.onMessage?.("소리를 켠 악기가 없어요. 악기 목록에서 🔇 버튼을 눌러 소리를 켜 주세요.", "info");
			return false;
		}

		const session = ++this.session;
		this.isPreparing = true;
		this.onStateChange?.();
		this.onMessage?.("악기 소리를 준비하고 있어요... (처음 한 번은 소리 파일을 받느라 오래 걸려요. 피아노는 약 100MB)", "info");

		try {
			// AudioContext는 버튼 클릭 안에서 만들어야 브라우저가 소리를 막지 않음 → await 전에 만듦
			this.createContext();
			const [module, banks] = await Promise.all([this.loadModule(), this.loadBanks()]);
			if (session !== this.session) return false; // 그 사이에 새 결과·음소거 변경이 있었음

			const player = module.createGroupedPlayer({
				context: this.context,
				workletUrl: new URL("dist/spessasynth_processor.min.js", SOUNDFONT_BASE).href,
				banks,
				destination: this.output,
			});
			this.player = player;
			const { unsupportedPrograms = [] } = await player.prepare(notes);
			if (session !== this.session) return false;

			player.seek(this.position);
			this.onMessage?.(
				unsupportedPrograms.length
					? "일부 악기(효과음 등)는 재생할 소리가 없어서 빠졌어요."
					: "",
				"info",
			);
			return true;
		} catch (error) {
			if (session !== this.session) return false;
			console.error(error);
			this.disposePlayer();
			this.onStateChange?.();
			this.onMessage?.("악기 소리를 불러오지 못했어요. 인터넷 연결을 확인하고 재생 버튼을 다시 눌러 주세요. (채보 결과와 원본 음원은 그대로예요)", "error");
			return false;
		} finally {
			if (session === this.session) {
				this.isPreparing = false;
				this.onStateChange?.();
			}
		}
	}

	createContext() {
		if (this.context) return;
		this.context = new AudioContext();
		this.output = this.context.createGain();
		this.output.gain.value = this.isMuted ? 0 : 1;
		this.output.connect(this.context.destination);
	}

	// 재생기 파일은 처음 재생할 때 한 번만 불러옴 (채보만 하는 사람은 받지 않음)
	loadModule() {
		if (!this.modulePromise) {
			this.modulePromise = import(new URL("dist/grouped-player.mjs", SOUNDFONT_BASE).href)
				.catch((error) => {
					this.modulePromise = null; // 실패하면 다음에 다시 시도
					throw error;
				});
		}
		return this.modulePromise;
	}

	loadBanks() {
		if (!this.banksPromise) {
			this.banksPromise = fetch(new URL("soundfonts.json", SOUNDFONT_BASE))
				.then((response) => {
					if (!response.ok) throw new Error(`SoundFont manifest HTTP ${response.status}`);
					return response.json();
				})
				.then((data) => data.banks)
				.catch((error) => {
					this.banksPromise = null;
					throw error;
				});
		}
		return this.banksPromise;
	}

	disposePlayer() {
		this.session += 1; // 진행 중인 준비 작업이 있으면 결과를 버리게 함
		this.isPreparing = false;
		this.stopLoop();
		if (this.player) {
			// 준비가 끝난 재생기만 위치를 믿을 수 있음 (준비 전에는 0을 돌려줌)
			if (this.player.duration > 0) this.position = this.player.currentTime;
			try {
				this.player.dispose();
			} catch (error) {
				console.warn(error);
			}
			this.player = null;
		}
	}

	// 재생 중에는 매 프레임 위치 알림. 끝까지 가면 멈춤 상태로 바꿈
	startLoop() {
		this.stopLoop();
		const tick = () => {
			if (!this.player) return;
			this.onTimeUpdate?.(this.player.currentTime, true);
			if (this.player.paused) {
				this.frameId = null;
				this.onStateChange?.();
				return;
			}
			this.frameId = requestAnimationFrame(tick);
		};
		this.frameId = requestAnimationFrame(tick);
	}

	stopLoop() {
		if (this.frameId !== null) {
			cancelAnimationFrame(this.frameId);
			this.frameId = null;
		}
	}
}
