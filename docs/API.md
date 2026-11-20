# 채보 플랫폼 MVP API

브라우저가 CDN에서 모델을 직접 다운로드해 음원을 처리한다. 백엔드는 화면, JavaScript 추론 모듈, 모델 다운로드 정보를 제공한다. 이 문서는 [MVP 요구사항](MVP_REQUIREMENTS.md)에 따른 HTTP API와 프론트엔드·모델 간 JavaScript 인터페이스를 정의한다.

## 담당자와 구현 위치

| 담당 에이전트 | 구현 위치 | 담당 작업 |
| --- | --- | --- |
| 프론트엔드 | `frontend/` | 파일 선택, 모델 준비·채보 상태 표시, 추론 호출, 음표의 이미지 렌더링, 원본 MP3 다운로드를 구현한다. |
| 백엔드 | `backend/main.py`, `backend/model_manifest.py` | 화면·정적 파일·추론 모듈을 제공하고 모델 파일의 URL 목록을 반환한다. |
| 모델 | `webscore_inference/web/`, CDN 버전 경로 | 음원 전처리, 브라우저 추론, 토큰 해석과 음표 반환을 구현한다. 같은 export의 모델 파일을 CDN에 배포하고 버전 정보를 전달한다. |

## HTTP API

저장소 루트에서 `uvicorn backend.main:app --host 127.0.0.1 --port 8000`으로 실행한다. 브라우저 클라이언트는 기본 설정의 CDN `current.json`을 직접 조회한다. 서버에서 URL을 발급하는 배포에서는 클라이언트의 `manifestUrl`에 `/api/model-manifest`를 지정한다.

`FRONTEND_DIR`은 제공할 프론트엔드 디렉터리다. 기본값은 `frontend/`이며 상대 경로는 저장소 루트 기준이다. Vite를 사용하면 `FRONTEND_DIR=frontend/dist`, 빌드 `base=/static/`으로 설정한다.

| 메서드와 경로 | 요청 | 성공 응답 | 구현 담당과 사용 방식 |
| --- | --- | --- | --- |
| `GET /` | 본문 없음 | `200 OK`, `text/html`로 프론트엔드 디렉터리의 `index.html` | 백엔드가 경로를 제공하고 프론트엔드가 HTML을 작성하거나 빌드한다. |
| `GET /static/{path}` | JS·CSS 등 자산 경로 | `200 OK`, 해당 정적 파일 | 백엔드가 프론트엔드 디렉터리를 마운트한다. HTML은 `/static/` 아래 자산을 참조한다. |
| `GET /inference/{path}` | `model-version-client.mjs`, `model-config.mjs`, `model-client.mjs`, `mel-frontend.mjs`, `midi-decoder.mjs` | `200 OK`, 해당 JavaScript 모듈 | 백엔드가 `webscore_inference/web/`를 제공한다. 프론트엔드가 모듈을 import한다. |
| `GET {manifestUrl}` | 기본값은 CDN의 `current.json` | `200 OK`, 아래 형식의 버전과 파일 URL 목록 | `VersionedWebScoreClient`가 CDN의 최신 모델 목록을 조회한다. |
| `GET /api/model-manifest` | 본문·쿼리·인증 헤더 없음 | `200 OK`, `application/json`으로 버전과 파일 URL 목록 | 서버에서 URL을 제공할 때 `manifestUrl`을 이 경로로 지정한다. |
| `GET {manifest.files[파일명]}` | 백엔드가 반환한 파일별 URL | `200 OK`, JSON 메타데이터 또는 ONNX 바이너리 | CDN이 응답하고 브라우저 추론 클라이언트가 다운로드한다. CDN은 실제 웹페이지 origin의 GET을 허용하도록 CORS를 설정한다. |

MVP는 로그인 없이 모델 URL 정보를 제공한다. R2를 선택한 경우에도 이 엔드포인트가 서명 URL을 발급하므로, 배포 담당자가 웹 사용자에게 제공할 모델 버전을 설정한다.

### 모델 URL 응답

```json
{
  "schemaVersion": 1,
  "format": "webscore-onnx-v1",
  "version": "piano-v1",
  "precision": "int8",
  "files": {
    "config.json": "https://models.example.com/piano-v1/config.json",
    "tokenizer.json": "https://models.example.com/piano-v1/tokenizer.json",
    "mel_encoder.onnx": "https://models.example.com/piano-v1/mel_encoder.onnx",
    "token_embed.onnx": "https://models.example.com/piano-v1/token_embed.onnx",
    "decoder_kv.int8.onnx": "https://models.example.com/piano-v1/decoder_kv.int8.onnx"
  }
}
```

| 필드 | 타입 | 규칙 |
| --- | --- | --- |
| `schemaVersion` | 정수 | `1` |
| `format` | 문자열 | `webscore-onnx-v1` |
| `version` | 문자열 | 같은 export를 가리키는 고유 버전 ID. 파일이 바뀌면 ID도 변경한다. |
| `precision` | 문자열 | `int8` 또는 `fp32`. FP32의 디코더 파일명은 `decoder_kv.onnx`. |
| `files` | 파일명과 URL의 객체 | 메타데이터 두 개, ONNX 세 개를 모두 포함한다. URL은 절대 HTTPS 주소를 사용하며 로컬 개발에서는 localhost의 HTTP를 허용한다. 비공개 R2는 파일별 서명 URL을 반환한다. |

응답은 `Cache-Control: no-store, private`로 제공한다. 모델 업데이트와 비공개 R2 배포 설정은 [모델 배포 문서](MODEL_AUTO_UPDATE.md)를 따른다.

### 백엔드 환경 설정

| 환경 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `MODEL_SOURCE` | `manifest` | `manifest`는 공개 버전 목록 조회, `cdn`은 고정 CDN 버전 주소, `r2`는 비공개 R2 카탈로그와 서명 URL을 사용한다. |
| `MODEL_MANIFEST_URL` | `https://models.midistudio.app/web-student-kd/current.json` | `manifest` 모드에서 조회할 모델 버전 목록. 각 API 요청마다 5초 제한으로 조회하고 최대 64KiB의 JSON을 읽는다. |
| `MODEL_BASE_URL` | `cdn` 모드에서 설정 | 공개 CDN의 한 모델 버전 디렉터리 URL. 쿼리·프래그먼트 없이 지정한다. |
| `MODEL_VERSION` | `cdn` 모드에서 설정 | 고정 CDN 응답에 넣을 버전 ID. |
| `MODEL_PRECISION` | `int8` | `cdn` 모드의 디코더 정밀도. |
| `FRONTEND_DIR` | `frontend/` | 화면과 정적 파일을 제공할 디렉터리. |

기본 설정으로 실행하면 모델 담당자가 제공한 `current.json`에서 버전과 파일 URL을 조회한다. 다른 배포의 목록은 `MODEL_MANIFEST_URL`로 지정한다.

```sh
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## 프론트엔드와 모델 간 JavaScript 인터페이스

프론트엔드는 `onnxruntime-web`을 준비하고 `VersionedWebScoreClient`를 초기화한다. 모델 클라이언트는 모델 URL 조회·다운로드·음원 전처리·추론·토큰 해석을 맡는다.

| 인터페이스 | 입력 | 반환과 호출 규칙 |
| --- | --- | --- |
| `VersionedWebScoreClient.create(ort, options)` | `ort`는 ONNX Runtime Web 모듈. `options.manifestUrl`의 기본값은 `model-config.mjs`에 지정한 CDN `current.json`. `options.clientOptions.executionProviders`는 기본 MVP에서 `['wasm']`. | `Promise<VersionedWebScoreClient>`. 화면 준비 시 한 번 호출하고 파일 처리에 재사용한다. |
| `model.transcribeAudio(file, options)` | 사용자가 선택한 MP3 `File`. `options.onProgress`는 선택적인 진행 콜백. | `Promise<{ notes, midiBytes, tokensByChunk }>`. 아래 음표 형식으로 렌더링한다. `midiBytes`와 `tokensByChunk`는 모델 패키지의 추가 반환값이다. |
| `options.onProgress(progress)` | `{ completedChunks: number, totalChunks: number }` | 프론트엔드가 채보 진행 상태를 표시한다. |
| `model.checkForUpdate()` | 없음 | 새 버전 확인 결과 `{ updated: boolean, version: string }`. 로딩된 버전이 있으면 업데이트 실패 시 해당 버전을 유지한다. |
| `model.dispose()` | 없음 | 화면 종료 시 예약된 작업이 끝난 뒤 세션을 해제한다. |

`notes`는 다음 객체의 배열이다. 프론트엔드가 이 값을 SVG 등으로 렌더링한다.

| 음표 필드 | 타입 | 의미 |
| --- | --- | --- |
| `program` | 정수 | MIDI 악기 번호 `0..127`. 타악기는 `128`. |
| `pitch` | 정수 | MIDI 음높이 `0..127`. |
| `onset` | 숫자 | 원본 음원 기준 시작 시각, 단위는 초. |
| `offset` | 숫자 | 원본 음원 기준 종료 시각, 단위는 초. |
| `isDrum` | 불리언 | 타악기 여부. |

Vite에서 백엔드가 제공하는 추론 모듈을 런타임에 가져오는 예시는 다음과 같다.

```js
import * as ort from 'onnxruntime-web';

const { VersionedWebScoreClient } = await import(
  /* @vite-ignore */ '/inference/model-version-client.mjs'
);
const model = await VersionedWebScoreClient.create(ort, {
  clientOptions: { executionProviders: ['wasm'] },
});

async function transcribeFile(file, onProgress) {
  return model.transcribeAudio(file, { onProgress });
}
```

원본 MP3 다운로드는 프론트엔드가 선택한 `File`의 객체 URL과 다운로드 링크로 구현한다. 객체 URL과 모델 세션의 해제도 프론트엔드가 담당한다. [브라우저 객체 URL 문서](https://developer.mozilla.org/en-US/docs/Web/API/URL/createObjectURL_static)

## 채보 결과 재생 (프론트 연결용)

[독립 SoundFont 재생 모듈](../soundfont-playback/README.md)은 기존 모델의 `notes`를 그대로 받는다. 추론 API, 원본 MP3 재생·다운로드, `midiBytes` 내보내기는 변경하지 않는다.

| 인터페이스 | 역할 |
| --- | --- |
| `groupPlaybackNotes(notes)` | 악기군별 트랙과 필요 뱅크 목록을 만든다. 재생에 쓰지 않는 GM 번호는 `unsupportedPrograms`에 반환한다. |
| `createGroupedPlayer({ context, workletUrl, banks, destination })` | 앱의 AudioContext 및 같은 origin의 Worklet과 연결한다. 음소거를 쓰려면 출력 `destination`에 GainNode를 전달한다. |
| `player.prepare(notes)` | 최초 재생 버튼에서 호출한다. 필요한 뱅크만 내려받고 재생용 MIDI를 구성한다. 이후 `play()`, `pause()`, `seek(seconds)`, `currentTime`, `duration`, `setMuted(boolean)`을 사용한다. 새 결과를 열기 전에 `dispose()`한다. |

[SoundFont CDN 목록](https://models.midistudio.app/web-student-kd/releases/soundfont-musescore-general-0.2-v1/soundfonts.json)은 모델 `current.json`과 독립적이다. 각 뱅크의 `url`, `bytes`, `sha256`으로 다운로드를 검증하며 R2 키나 서명 URL은 필요 없다. 검증된 파일은 IndexedDB에 저장하고, 저장이 막히거나 공간이 부족해도 네트워크로 재생할 수 있다. 다운로드 또는 해시 검증에 실패하면 채보 화면과 원본 파일은 유지하고 재생만 다시 시도할 수 있게 한다.

## 오류 응답과 복구

| 구간 | 오류 | 처리 담당과 복구 |
| --- | --- | --- |
| 모델 URL 조회 | `503 Service Unavailable`, `MODEL_MANIFEST_UNAVAILABLE` | 백엔드가 아래 JSON으로 응답한다. 프론트엔드는 모델 준비 실패를 표시하고 사용자가 다시 초기화할 수 있게 한다. 배포 담당자는 CDN 또는 R2 설정을 확인한다. |
| 모델 파일 다운로드·최초 로딩 | 네트워크, CORS, 만료 URL, 메타데이터 불일치, 런타임 로딩 오류 | 모델 클라이언트의 Promise가 reject된다. 프론트엔드는 파일 선택을 유지하고 모델 준비를 재시도한다. |
| 모델 버전 갱신 | 새 모델 URL 조회 또는 로딩 실패 | 모델 클라이언트는 준비된 이전 버전으로 계속 처리하고 `onUpdateError`로 알린다. |
| 음원 디코딩·추론·이미지 렌더링 | 브라우저 처리 실패 | 프론트엔드는 해당 단계의 실패를 표시하고 재시도 또는 다른 MP3 선택을 제공한다. |

```json
{
  "error": {
    "code": "MODEL_MANIFEST_UNAVAILABLE",
    "message": "모델 다운로드 정보를 가져올 수 없습니다. 설정을 확인하고 다시 시도해 주세요."
  }
}
```

## 이전 계약에서의 변경

MP3 업로드 `POST /api/transcriptions`, 이미지 조회 `GET /api/transcriptions/{id}/score`, MP3 조회 `GET /api/transcriptions/{id}/audio.mp3`를 브라우저 내부 처리로 대체했다. 프론트엔드는 이 경로를 호출하던 코드를 모델 클라이언트 호출과 객체 URL 처리로 전환한다.

`backend/transcription.py`와 Python CPU 어댑터는 별도 로컬 실행용 소스로 보관한다. 현재 웹 서비스의 모델 인터페이스는 이 문서의 JavaScript 계약을 따른다.
