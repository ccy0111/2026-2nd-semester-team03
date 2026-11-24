# 브라우저 채보 재생: MuseScore General SF2

기존 추론 결과 `notes`를 MuScriptor의 [MT3_FULL_PLUS](https://github.com/muscriptor/muscriptor/blob/68456b17da8eea651eee6084c104611089810231/muscriptor/tokenizer/mt3.py#L44-L82) 기준으로 묶어 오디오를 재생하는 독립 모듈. 모델·기존 프론트 PR·원본 MIDI는 바꾸지 않는다. 분류표의 존재가 현재 모델의 채보 정확도를 보장하지 않는다.

- 35개 멜로디 악기군과 표준 드럼 킷. 오케스트라 히트(GM 55, 그룹 18) 없음. 원본 GM 번호 및 그룹 ID는 재번호화하지 않는다.
- 같은 그룹은 대표 음색으로 재생한다. 예: GM 24·25 → 어쿠스틱 기타 GM 24. 드럼은 전용 MIDI 채널과 SF2 bank 128/program 0을 쓴다. 미분류 GM은 원본에서 제거하지 않고 `unsupportedPrograms`로 알려준다.
- 첫 재생 때 실제로 쓰는 SF2만 버전 고정 CDN에서 받는다. 크기와 SHA-256을 검증한 뒤 IndexedDB에 저장하며, 저장이 불가능해도 네트워크 재생은 진행한다. 피아노 최초 로드 약 100 MB; 전체 7뱅크 약 156 MB. 샘플은 재압축하지 않았다.
- `soundfonts.json`은 캐시 가능한 불변 릴리스다. 교체할 때 새 릴리스 경로 및 해시를 커밋한다. 기존 모델 manifest와 R2 키는 필요 없다.

## 실행 및 연결

```sh
cd soundfont-playback
npm ci
npm test
npm run build
python -m http.server 8080
```

`http://localhost:8080/demo.html`에서 테스트 음표로 재생한다. 서비스에는 `dist/grouped-player.mjs`, `dist/spessasynth_processor.min.js`와 이 manifest를 앱 자산으로 배포한다. Worklet은 앱과 같은 origin, 웹페이지는 HTTPS(개발 시 localhost), SF2는 CDN에서 읽는다.

```js
import {createGroupedPlayer} from './dist/grouped-player.mjs';
const {banks} = await (await fetch('./soundfonts.json')).json();
// 재생 버튼에서 최초 호출하며, notes는 기존 모델 응답의 배열을 그대로 전달한다.
const player = createGroupedPlayer({context: audioContext,
  workletUrl: './dist/spessasynth_processor.min.js', banks});
const {tracks, unsupportedPrograms} = await player.prepare(notes);
await player.play();
```

원본 MP3 재생은 `<audio>` 쪽에 별도로 두고 MIDI 합성과 혼동하지 않는다. 일시정지 `pause()`, 시킹 `seek(seconds)`, 커서 `currentTime`, 음소거 `setMuted`(출력으로 GainNode 전달)가 제공된다. 다음 채보 결과를 열기 전에 `dispose()`한 뒤 새 플레이어를 만든다. 원본 MIDI 내보내기는 기존 코드를 그대로 사용한다.

## SF2 재현 및 검증

공식 원본 URL과 체크섬은 [SOURCE.json](tools/SOURCE.json), 저작권 및 MIT 고지는 [라이선스](tools/MuseScore_General_License.md), 원본 샘플 저작자는 [샘플 출처](tools/MuseScore_General_Sample_Sources.csv) 참고.

```sh
python tools/subset_sf2.py /path/to/MuseScore_General.sf2 --out assets
python tools/verify_render.py /path/to/MuseScore_General.sf2 assets    # libfluidsynth 필요
node tools/verify-playback.mjs assets                                 # 실제 SpessaSynth PCM
```

FluidSynth 비교는 모든 36개 프리셋에서 대표 음 3개 × 벨로시티 2개를 각각 렌더해 원본과의 PCM을 비교한다. SpessaSynth 비교는 실제 7개 뱅크를 로드하고, 3개 MIDI 포트에서 36개 악기군의 소리가 나오는지 검증한다. IndexedDB 경로는 자동 단위 테스트에 포함되며, 브라우저 실제 음성 출력은 데모로 별도 확인할 수 있다.
