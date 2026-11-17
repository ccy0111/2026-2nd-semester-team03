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
This is a source-only snapshot. Model weights and training checkpoints are intentionally omitted while training continues. Real ONNX/browser inference has not been validated in this publication; the 5 Node tests use synthetic inputs and a mock runtime.

This browser package remains separate from the FastAPI API. The new CPU adapter in backend/ implements the shared model contract and SVG score preview; see ../CPU_ADAPTER.md. For hosting, place tokenizer.json and config.json alongside the externally supplied ONNX files at baseUrl (default /models). The metadata files are stored at this package root and are not automatically served from /models.
