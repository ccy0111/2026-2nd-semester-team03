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

The MVP backend serves these JavaScript modules under `/inference/`. `VersionedWebScoreClient` defaults to the CDN manifest URL in `web/model-config.mjs`. Pass `manifestUrl: '/api/model-manifest'` to use the backend URL provider; see `../docs/API.md`. Place tokenizer.json and config.json alongside the compatible ONNX files in the CDN version directory. The Python CPU adapter in `backend/` is available for standalone local use; see [CPU adapter](../docs/CPU_ADAPTER.md).

## R2 배포 연결: 피아노 시운전 모델

이번 `step-36000` 모델은 수집 데이터 시운전용이며, 사용자 관찰상 현재는 피아노 채보가 주 대상입니다. GM Expert 모델의 완성이나 다악기 정확도 검증을 의미하지 않습니다. 이 PR에는 가중치, R2 자격 증명, 실제 배포 주소가 없습니다.

학습 체크포인트(`model.safetensors`) 자체는 브라우저에서 읽을 수 없습니다. 해당 체크포인트로 ONNX를 내보낸 뒤 **동일 export의** 다음 파일을 하나의 버전 경로(예: `piano/step-36000/`)에 배치합니다.

- `config.json`, `tokenizer.json`
- `mel_encoder.onnx`, `token_embed.onnx`
- `decoder_kv.int8.onnx` (`useInt8: true`) 또는 `decoder_kv.onnx` (`false`)

현재 클라이언트는 float32 입출력/KV 캐시를 가정합니다. 임의 FP16 변환 파일을 대체하지 마세요. ONNX external-data sidecar가 있다면 현재 URL 로더만으로 지원된다고 가정하지 말고 별도 대응 및 검증이 필요합니다. 업로드 완료와 체크섬 확인 전에는 새 버전을 참조하지 마세요.

### 비공개 버킷: 파일별 단기 서명 URL

앱의 서버가 R2 파일별 단기 GET 서명 URL을 발급한다. 현재 MVP는 `MODEL_SOURCE=r2` 설정 시 `/api/model-manifest`에서 이 URL 묶음을 웹 사용자에게 제공한다. 설정과 카탈로그 형식은 `../docs/MODEL_AUTO_UPDATE.md`를 따른다. 아래 `authorizedModelUrls`는 서버가 발급한 값이다.

```js
// All five filenames must be present. Values are independently signed GET URLs.
const client = await WebScoreClient.create(ort, {
  useInt8: true,
  modelUrls: authorizedModelUrls,
});
```

`modelUrls` 키는 위의 파일명과 동일해야 합니다. 한 파일이라도 빠지면 공개 `baseUrl`로 우회하지 않고 실패합니다. 서명 URL을 `baseUrl`에 넣지 마세요. 파일마다 서명이 다르므로 URL 뒤에 파일명을 붙이면 유효하지 않습니다. 만료 또는 403 응답 시 앱이 인증을 다시 확인하고 **전체 URL 묶음**을 갱신해 로드를 재시도해야 합니다. 다운로드가 끝날 수 있도록 적절한 유효시간을 선택하세요.

- R2 API 키/secret은 서버에만 둡니다. 브라우저, 저장소, 로그에 넣지 않습니다.
- 로더는 네트워크/메타데이터 파싱/ONNX 세션 생성 오류를 파일명 기반 메시지로 바꾸며 원본 오류나 `cause`를 전달하지 않습니다. 다만 브라우저 개발자 도구의 요청 목록이나 ONNX Runtime 자체 로그까지 숨기는 기능은 아닙니다. 앱에서도 URL 묶음과 원시 응답을 로그에 남기지 마세요.
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

Node 로더 테스트는 URL 전달/서명 보존/누락 및 403 처리와 mock runtime을 검증합니다. Python API 테스트는 모델 URL 응답과 JavaScript 모듈 제공을 검증합니다. 실제 프론트엔드 화면 연결, CDN 다운로드, WebGPU/WASM 실행은 배포 환경과 실제 모델로 확인합니다.

## Inference optimization controls (opt-in)

Use the WebGPU build explicitly when testing a GPU path:

```js
import * as ort from 'onnxruntime-web/webgpu';
const client = await WebScoreClient.create(ort, {
  modelUrls: authorizedModelUrls,
  useInt8: true,
  executionProviders: ['wasm'], // verified fallback, including Node WASM tests
  cacheEmbeddings: true,       // one batched lookup, ~3.5 MiB extra CPU memory
});
// After all in-flight transcription is finished:
await client.dispose();
```

`cacheEmbeddings` defaults to false. When enabled, the existing exported embedding
model evaluates all vocabulary IDs once and caches the exact resulting float32
table. No embedding weights are approximated. Repeated token embedding session
calls are replaced by array gathers. CPU encoder/decoder preprocessing and token
sampling remain unchanged. `dispose()` releases the sessions and table; do not
call it during an active transcription. Concurrent `transcribePcm()` calls on
one client are rejected.

### Last-position output projection

The autoregressive client only needs the last-position logits, but the original
export computes the output projection for every prefill position. To create a
separate, compatible export:

```sh
python -m pip install -r webscore_inference/tools/requirements.txt
python webscore_inference/tools/optimize_last_logits.py SOURCE_EXPORT NEW_EXPORT
```

The tool verifies the known FP32 WebScore final normalization/MatMul pattern,
slices the final hidden state before output normalization/projection, and
optionally re-quantizes the optimized decoder. It refuses to overwrite any
existing export. Attention and KV outputs are unchanged. The new config carries
`decoder_logits: "last"`; the client accepts both `[1,seq,vocab]` and
`[1,1,vocab]` outputs. Deploy all files from the same new export together, and
validate numerical/sequence behavior before changing an app's model version.

### Experimental GPU-resident KV cache (not validated on a real browser)

`gpuKvCache: true` keeps decoder `present_k` and `present_v` outputs on GPU while
leaving logits on CPU for sampling. It requires `useInt8:false` and WebGPU as the
first execution provider. Previous GPU tensors are released after the next run,
and the final cache is released on EOS/error. This is **off by default**: do not
enable it in production without device-specific correctness, latency and memory
tests. Dynamic INT8 operator coverage is not assumed to be GPU-native; selecting
WebGPU does not establish that every operator ran there.

No graph capture is enabled: the current KV sequence dimension grows dynamically.
No FP16 model is substituted: the client's I/O contract remains float32. The
existing provider default remains `['webgpu','wasm']`, but automatic device
selection and browser-specific fallback are not yet validated. Use explicit
`['wasm']` for the verified CPU path.

WASM multithreading requires cross-origin isolation in a browser and should be
configured at the application/runtime level. Web Worker execution can improve UI
responsiveness; it is not a promise of faster model inference. This source-only
package does not modify hosting headers or another team's UI.

The mel frontend also skips exactly-zero filter coefficients and reuses FFT
scratch buffers. This arithmetic-preserving optimization is enabled normally;
regression tests compare sparse and dense summation exactly and check buffer
reset across noise/tone/silence. It changes preprocessing cost, not the learned
model, chunk boundary accuracy, or predicted instrument identity.
