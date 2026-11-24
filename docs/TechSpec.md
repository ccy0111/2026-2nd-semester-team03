# 채보 플랫폼 TechSpec

[PRD](PRD.md)의 MVP를 FastAPI 정적 서버와 브라우저 ONNX 추론으로 구현한다. 백엔드는 화면과 모델 다운로드 정보를 제공하고, 음원과 추론 결과는 브라우저 메모리에서 처리한다.

## 구성과 책임

| 구성 | 구현 위치 | 책임 |
| --- | --- | --- |
| 웹 서버 | `backend/main.py` | `/`, `/static/`, `/inference/`, `/api/model-manifest` 제공 |
| 모델 URL 제공 | `backend/model_manifest.py` | CDN manifest 조회, 고정 버전 URL 생성, 선택적 R2 서명 URL 발급 |
| UI와 음원 디코딩 | `frontend/scripts/` | 파일 선택, 16 kHz 모노 변환, Worker 통신, 진행률과 피아노롤 SVG 표시 |
| 추론 Worker | `frontend/scripts/model.worker.js` | ONNX Runtime Web 1.31.1과 WASM CPU로 모델 로딩·추론 실행 |
| 추론 코어 | `webscore_inference/web/` | Log-Mel 전처리, 토큰 생성·해석, 청크 연결, MIDI 생성, 모델 버전 교체 |

현재 프론트엔드는 HTML·CSS·JavaScript 모듈로 구성된다. Python CPU 어댑터는 별도 로컬 실험용이며 웹 요청의 추론 경로에는 포함하지 않는다.

## 음원 처리 흐름

1. 메인 스레드가 선택한 MP3를 Web Audio API로 디코딩하고 16 kHz 모노 `Float32Array`로 변환한다.
2. PCM 버퍼를 transferable로 Worker에 전달한다.
3. PCM을 5초 80,000샘플 단위로 나누고 마지막 구간을 0으로 패딩한다.
4. 2,048샘플 Hann window, 160샘플 hop, 512-bin HTK Log-Mel로 `[1, 501, 512]` 입력을 만든다.
5. 인코더·토큰 임베딩·KV 디코더로 토큰을 생성하고, 음표와 MIDI로 변환한다. 마지막 패딩 구간의 예측은 제외하거나 원본 길이로 자른다.
6. UI가 반환된 음표를 SVG로 그린다. 원본 MP3 다운로드는 `Blob` URL로 구현하고 사용 후 해제한다.

## 인터페이스

`GET /api/model-manifest`는 `{ schemaVersion: 1, format: "webscore-onnx-v1", version, precision, files }`를 반환한다. 설정·조회 실패 시 `503`과 `MODEL_MANIFEST_UNAVAILABLE`을 반환하며, 응답은 `Cache-Control: no-store, private`로 제공한다.

- `VersionedWebScoreClient.create(ort, options)`: 모델 버전을 확인하고 세션을 준비한다.
- `model.transcribePcm(pcm, options)` 또는 `model.transcribeAudio(file, options)`: `{ notes, midiBytes, tokensByChunk, chunkStatuses }`를 반환한다.
- 음표는 `{ program, pitch, onset, offset, isDrum }`이며 시간 단위는 초다. `program`은 `0..127`, 타악기는 `128`이다.
- 진행 콜백은 `{ completedChunks, totalChunks, chunkStatus }`를 전달한다.
- 현재 Worker와 `ModelService`는 UI에 `{ notes, chunkStatuses }`만 전달한다.

상세 HTTP 계약과 환경 변수는 [API 문서](API.md)를 따른다.

## 청크 경계와 오류 복구

기본 경계 모드는 `pitch-tie`다. 정상 추론은 매 청크 BOS부터 시작하고, 모델의 선행 선언과 이전 청크의 열린 음표를 악기·음높이로 우선 연결한다. 정확한 일치가 없을 때 같은 음높이의 후보가 양쪽에 하나씩만 있으면 연결하며, 모호한 연결은 경계에서 종료한다.

청크당 기본 생성 한도는 1,024토큰이다. EOS 없이 한도에 도달한 `TokenLimitError`만 다음과 같이 복구한다.

1. 실패한 토큰을 버리고, 이전 열린 음표를 해당 청크 시작 시각에 종료한다.
2. 새 KV 캐시와 빈 `tie` 프롬프트로 같은 청크를 한 번 재시도한다.
3. 성공하면 `recovered`, 다시 한도를 소진하면 해당 구간을 비우고 `failed`로 기록한 뒤 다음 청크를 처리한다.

정상 구간은 `ok`다. `chunkStatuses`에는 구간 시각, 상태, 시도 횟수, 경계 초기화 여부가 포함된다. UI는 이를 확인해 누락을 안내해야 한다. 네트워크·런타임 오류와 취소는 청크 복구 대상에 포함하지 않는다. 상세 정책은 [EOS 복구 문서](EOS_CHUNK_RECOVERY.md)를 따른다.

## 모델 배포와 자원 관리

- 기본 클라이언트는 `model-config.mjs`의 CDN `current.json`을 직접 조회한다. 서버 설정을 적용하려면 Worker의 생성 옵션에 `manifestUrl: '/api/model-manifest'`를 지정한다.
- 같은 export의 `config.json`, `tokenizer.json`, `mel_encoder.onnx`, `token_embed.onnx`, `decoder_kv.int8.onnx`를 한 버전으로 배포한다. FP32는 디코더 파일명이 `decoder_kv.onnx`다.
- 최초 로딩, 기본 60초 주기, 추론 전에 버전을 확인한다. 추론과 업데이트를 직렬화하고 새 세션 검증 후 교체한다. 갱신 실패 시 기존 세션을 유지한다.
- CDN은 앱 origin의 GET을 허용해야 한다. R2 접근 키는 서버에만 두고 서명 URL이나 원시 제공자 오류를 로그에 남기지 않는다.
- 페이지 종료 시 Worker를 종료한다. 직접 사용하는 버전 클라이언트는 `dispose()`로 대기 작업 완료 후 세션을 해제한다.

배포 설정과 R2 카탈로그 형식은 [모델 배포 문서](MODEL_AUTO_UPDATE.md)를 따른다.

## 구현 잔여와 검증

현재 UI에는 20초 개발 제한이 남아 있다. MVP 완료 전 시간 제한 해제, 원본 MP3 다운로드와 객체 URL 해제, 모델 준비 재시도, 청크 누락 안내를 구현해야 한다.

저장소 루트에서 다음 자동 테스트를 실행한다.

```sh
python -m unittest discover -s tests -v
node --test webscore_inference/tests/*.mjs
```

자동 테스트는 주로 합성 입력과 모의 런타임으로 계약·복구·버전 교체를 검증한다. 실제 CDN 모델과 MP3를 사용한 브라우저 검증에서는 전체 곡 처리, 다운로드·재시도, 누락 안내, 음원의 서버 미전송, 품질·속도·메모리 사용량을 확인한다.
