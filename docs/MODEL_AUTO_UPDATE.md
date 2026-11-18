# 모델 배포와 버전 업데이트

브라우저의 `VersionedWebScoreClient`는 `webscore_inference/web/model-config.mjs`의 CDN `current.json`을 기본 주소로 사용한다. 서버의 `GET /api/model-manifest`를 사용하려면 클라이언트 생성 옵션에 `manifestUrl: '/api/model-manifest'`를 지정한다. [HTTP API와 JavaScript 인터페이스](API.md)

## 공개 CDN 설정

기본 설정은 모델 담당자가 배포한 [현재 버전 목록](https://models.midistudio.app/web-student-kd/current.json)을 조회한다. 이 목록에는 버전 ID, 정밀도, 같은 export의 파일 URL 다섯 개가 들어 있다. 백엔드가 각 API 요청마다 목록을 읽으므로 배포 담당자가 `current.json`을 바꾸면 브라우저의 다음 버전 확인에 반영된다.

| 환경 변수 | 값 |
| --- | --- |
| `MODEL_SOURCE` | `manifest`, 기본값 |
| `MODEL_MANIFEST_URL` | `https://models.midistudio.app/web-student-kd/current.json`, 기본값 |

현재 제공받은 초기 export는 `step-00036000-int8-v1`이다. 해당 버전의 `config.json`은 [모델 설정](https://models.midistudio.app/web-student-kd/releases/step-00036000-int8-v1/config.json)에서 확인한다.

고정 버전을 제공할 때는 아래 설정을 사용한다.

| 환경 변수 | 값 |
| --- | --- |
| `MODEL_SOURCE` | `cdn` |
| `MODEL_BASE_URL` | 같은 export의 파일 다섯 개를 배치한 HTTPS 버전 디렉터리 URL |
| `MODEL_VERSION` | 해당 export의 고유 버전 ID |
| `MODEL_PRECISION` | `int8`, 기본값. FP32 디코더를 제공하면 `fp32`. |

INT8 버전에는 `config.json`, `tokenizer.json`, `mel_encoder.onnx`, `token_embed.onnx`, `decoder_kv.int8.onnx`가 필요하다. FP32는 디코더 파일명만 `decoder_kv.onnx`로 바뀐다. 모델 파일을 변경하면 새 버전 경로와 ID를 사용한다. CDN의 CORS는 실제 웹페이지 origin의 GET을 허용하도록 설정한다.

## 비공개 R2 설정

`pip install -r requirements-model-manifest.txt`로 서버 의존성을 설치하고 `MODEL_SOURCE=r2`를 설정한다. 모델 URL API는 웹 사용자에게 파일별 서명 URL을 제공한다.

| 환경 변수 | 의미와 기본값 |
| --- | --- |
| `R2_MODEL_ACCOUNT_ID` | R2 계정 ID |
| `R2_MODEL_BUCKET` | 모델 버킷 이름 |
| `R2_MODEL_ACCESS_KEY_ID` | 서버의 버킷 읽기 키 |
| `R2_MODEL_SECRET_ACCESS_KEY` | 서버의 버킷 읽기 비밀 키 |
| `R2_MODEL_CATALOG_KEY` | `web-student-kd/current.json` |
| `R2_MODEL_RELEASE_PREFIX` | `web-student-kd/releases` |
| `R2_MODEL_URL_TTL_SECONDS` | `3600`, 허용 범위는 60–3600초 |

R2 카탈로그에는 파일별 키, 크기, SHA256을 저장한다. 아래 예시의 `files`는 항목 하나를 보여 주며, 실제 카탈로그에는 같은 버전의 파일 다섯 개를 모두 넣는다.

```json
{
  "schemaVersion": 1,
  "format": "webscore-onnx-v1",
  "version": "piano-v1",
  "precision": "int8",
  "files": {
    "config.json": {
      "key": "web-student-kd/releases/piano-v1/config.json",
      "size": 343,
      "sha256": "REPLACE_WITH_ACTUAL_64_HEX_SHA256"
    }
  }
}
```

각 키는 `{release_prefix}/{version}/{filename}` 형식을 사용한다. 배포 담당자가 실제 크기와 체크섬으로 업로드를 확인한 뒤 카탈로그를 갱신한다. 서버는 카탈로그 한 번의 조회에서 같은 버전의 URL을 생성하며, API 응답과 오류에는 R2 접근 키를 넣지 않는다.

호스트가 추후 별도 인증을 추가할 때는 `create_model_manifest_router(service, authorize)`를 사용할 수 있다. 현재 MVP의 공개 모델 URL API는 `backend/main.py`에서 제공한다.

## 브라우저 업데이트 동작

- `VersionedWebScoreClient`는 최초 로딩, 기본 60초 주기, 추론 시작 전에 버전을 확인한다. `pollIntervalMs: 0`으로 주기 확인을 끌 수 있다.
- 현재 처리 중인 곡은 같은 모델로 끝낸 뒤 다음 버전을 로딩한다. 새 세션이 모두 준비되고 인터페이스 검사를 통과하면 교체한다.
- 새 버전 로딩에 실패하면 로딩된 이전 모델을 사용한다. 최초 로딩 실패는 호출자에게 오류를 반환한다.
- 같은 버전의 서명 URL이 갱신되어도 이미 로딩된 세션을 재사용한다.
- 매니페스트 조회 기본 제한 시간은 15초다. `dispose()`는 예약 작업이 끝난 뒤 세션을 해제한다.

버전 교체·세션 해제·실패 복구 테스트는 `node --test webscore_inference/tests/test_model_versions.mjs`로 실행한다. 실제 CDN 다운로드와 ONNX 실행은 배포 주소와 호환 모델을 준비한 뒤 확인한다.
