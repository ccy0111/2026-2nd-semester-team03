# WebScore Inference

브라우저(`WebGPU` + `WASM CPU` 폴백)에서 오디오(MP3/WAV 등)를 받아 `.mid`를 출력하는 추론 패키지입니다.

## 파일 구성
- `config.json` : 모델 설정
- `tokenizer.json` : 894 토큰 어휘표
- `models/` : ONNX 모델 바이너리 (`mel_encoder.onnx`, `token_embed.onnx`, `decoder_kv.int8.onnx` / `decoder_kv.onnx`)
- `web/mel-frontend.mjs` : 오디오 디코딩/리샘플(16 kHz mono) + 5초 청크 512-bin Log-Mel
- `web/midi-decoder.mjs` : 토큰 디코딩 + 청크 간 `tie` 연결 + `.mid` 바이너리 생성
- `web/model-client.mjs` : `onnxruntime-web` (`webgpu` → `wasm`) 추론 클라이언트

## 사용 예시
```js
import * as ort from "onnxruntime-web";
import { WebScoreClient } from "./web/model-client.mjs";

const client = await WebScoreClient.create(ort, {
  baseUrl: "/models",
  useInt8: true,
});

// mp3 File / Blob / ArrayBuffer -> MIDI bytes
const { notes, midiBytes } = await client.transcribeAudio(mp3File);
```

## 테스트
```powershell
node --test tests/test_inference.mjs
```

## Publication status
This is a source-only snapshot. Model weights and training checkpoints are intentionally omitted while training continues. Real ONNX/browser inference has not been validated in this publication; the Node tests use synthetic inputs, mocked downloads and a mock runtime.

This browser package remains separate from the FastAPI API. The new CPU adapter in backend/ implements the shared model contract and SVG score preview; see ../CPU_ADAPTER.md. For hosting, place tokenizer.json and config.json alongside the externally supplied ONNX files at baseUrl (default /models). The metadata files are stored at this package root and are not automatically served from /models.

## R2 배포 연결: 피아노 시운전 모델

이번 `step-36000` 모델은 수집 데이터 시운전용이며, 사용자 관찰상 현재는 피아노 채보가 주 대상입니다. GM Expert 모델의 완성이나 다악기 정확도 검증을 의미하지 않습니다. 이 PR에는 가중치, R2 자격 증명, 실제 배포 주소가 없습니다.

학습 체크포인트(`model.pt`) 자체는 브라우저에서 읽을 수 없습니다. 해당 체크포인트로 ONNX를 내보낸 뒤 **동일 export의** 다음 파일을 하나의 버전 경로(예: `piano/step-36000/`)에 배치합니다.

- `config.json`, `tokenizer.json`
- `mel_encoder.onnx`, `token_embed.onnx`
- `decoder_kv.int8.onnx` (`useInt8: true`) 또는 `decoder_kv.onnx` (`false`)

현재 클라이언트는 float32 입출력/KV 캐시를 가정합니다. 임의 FP16 변환 파일을 대체하지 마세요. ONNX external-data sidecar가 있다면 현재 URL 로더만으로 지원된다고 가정하지 말고 별도 대응 및 검증이 필요합니다. 업로드 완료와 체크섬 확인 전에는 새 버전을 참조하지 마세요.

### 비공개 버킷: 파일별 단기 서명 URL

R2 버킷을 공개로 전환할 필요가 없습니다. 앱의 **서버**가 사용자 인증/권한을 확인하고 각 파일에 대한 단기 GET 서명 URL을 발급한 뒤, 브라우저는 다음 형태로 전달합니다. 이 PR은 로그인이나 URL 발급 API를 구현하지 않습니다. 아래 `authorizedModelUrls`는 그 API에서 받은 값이며, Git/HTML에 고정 저장하지 않습니다.

```js
// All five filenames must be present. Values are independently signed GET URLs.
const client = await WebScoreClient.create(ort, {
  useInt8: true,
  modelUrls: authorizedModelUrls,
});
```

`modelUrls` 키는 위의 파일명과 동일해야 합니다. 한 파일이라도 빠지면 공개 `baseUrl`로 우회하지 않고 실패합니다. 서명 URL을 `baseUrl`에 넣지 마세요. 파일마다 서명이 다르므로 URL 뒤에 파일명을 붙이면 유효하지 않습니다. 만료 또는 403 응답 시 앱이 인증을 다시 확인하고 **전체 URL 묶음**을 갱신해 로드를 재시도해야 합니다. 다운로드가 끝날 수 있도록 적절한 유효시간을 선택하세요.

- R2 API 키/secret은 서버에만 둡니다. 브라우저, 저장소, 로그에 넣지 않습니다.
- 서명 URL은 유효기간 동안 소유자가 재사용할 수 있는 임시 접근 권한입니다. 공유하거나 로그에 남기지 않습니다.
- R2 CORS에 실제 앱 origin의 `GET` 요청을 허용해야 합니다. CORS는 사용자 인증을 대신하지 않습니다.
- 브라우저로 전달된 가중치는 사용자가 복사할 수 있습니다. 인증은 다운로드 접근 제어이지 모델 복사 방지가 아닙니다.
- 서명 URL 방식이 곧 커스텀 도메인 CDN 캐싱을 의미하지는 않습니다.

### 공개 배포를 별도로 선택한 경우

```js
const client = await WebScoreClient.create(ort, {
  baseUrl: "https://models.example.com/piano/step-36000",
  useInt8: true,
});
```

이는 예시 주소입니다. 이 PR은 버킷 공개, 도메인 설정, 업로드 또는 배포를 수행하지 않습니다. 실제 승인된 배포 URL을 앱 설정에서 주입하고 CORS를 맞춰야 합니다.

### 인수 확인

Node 로더 테스트는 URL 전달/서명 보존/누락 및 403 처리와 mock runtime을 검증합니다. Python API 회귀 테스트도 별도입니다. **실제 R2 다운로드, ONNX 수치 일치, WebGPU/WASM 실행, 속도 및 피아노 채보 품질 검증을 대체하지 않습니다.** 프런트엔드는 아직 이 패키지와 연결되어 있지 않으며, FastAPI CPU 경로에 R2 URL을 넣어 브라우저 추론으로 바뀌는 구조가 아닙니다.
