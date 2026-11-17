# 채보 플랫폼 MVP API

브라우저는 화면을 제공한 로컬 FastAPI 서버에 MP3 한 개를 전송한다. 서버는 사용자 PC의 CPU에서 채보를 완료한 뒤, 같은 서버에서 채보 이미지와 MP3를 받을 수 있는 URL을 반환한다. 이 문서는 [MVP 요구사항](MVP_REQUIREMENTS.md)의 요청·응답과 실패 시 복구 동작을 정의한다.

## 담당자와 구현 위치

| 담당 에이전트 | 구현 위치 | 담당 작업 |
| --- | --- | --- |
| 프론트엔드 | `frontend/` | 파일 선택·제출 화면을 만들고, 처리 중 상태와 결과를 표시한다. 응답의 이미지 URL과 다운로드 URL을 화면에 연결하고, `index.html`과 정적 자산을 제공할 위치를 정한다. |
| 백엔드 | `backend/main.py` | 화면과 API를 제공한다. 업로드 파일을 저장하고 모델을 호출한 뒤, 이미지와 원본 MP3를 결과 URL에서 제공한다. |
| 모델 | `backend/transcription.py`, `models/` | 팀에서 제공한 모델 파일을 `models/`에 배치하고, 모델 로딩·음원 전처리·CPU 추론·채보 이미지 생성을 구현한다. |

## 브라우저와 백엔드 간 HTTP API

서버 진입점은 `backend.main:app`으로 두고, 저장소 루트에서 `uvicorn backend.main:app --host 127.0.0.1 --port 8000`으로 실행한다. 프론트엔드는 현재 페이지와 같은 호스트·포트로 요청한다.

FastAPI에 제공할 프론트엔드 디렉터리는 `FRONTEND_DIR` 환경 변수로 지정하며 기본값은 `frontend/`다. 상대 경로는 저장소 루트를 기준으로 해석한다. React·TypeScript를 Vite로 빌드하는 경우 `FRONTEND_DIR=frontend/dist`로 지정하고, 빌드의 `base`를 `/static/`으로 설정한다. 프론트엔드의 API 호출에는 `/api/`로 시작하는 경로를 사용한다. [Vite 공식 문서](https://vite.dev/guide/build.html#public-base-path)

| 메서드와 경로 | 요청 | 성공 응답 | 구현 담당 | 프론트엔드 사용 방식 |
| --- | --- | --- | --- | --- |
| `GET /` | 본문 없음 | `200 OK`, `text/html`로 지정된 프론트엔드 디렉터리의 `index.html` 제공 | 백엔드가 경로 제공, 프론트엔드가 HTML 작성 또는 빌드 | 사용자가 로컬 주소에서 화면을 연다. |
| `GET /static/{path}` | 지정된 프론트엔드 디렉터리 기준의 자산 경로 | `200 OK`, 해당 정적 파일 | 백엔드가 지정된 프론트엔드 디렉터리를 `/static`에 마운트, 프론트엔드가 자산 작성 또는 빌드 | HTML은 JS·CSS 등 정적 자산을 `/static/` 아래 경로로 참조한다. |
| `POST /api/transcriptions` | `multipart/form-data`의 `file` 필드에 MP3 파일 한 개 | `200 OK`, `application/json`으로 `image_url`, `download_url` 반환 | 백엔드가 업로드·모델 호출·결과 저장 구현, 프론트엔드가 제출 요청 구현 | `FormData`의 `file`에 파일을 넣어 `fetch`로 전송한다. 응답을 기다리는 동안 처리 중 상태를 표시하고 제출 버튼을 비활성화한다. |
| `GET /api/transcriptions/{id}/score` | `image_url`에 포함된 작업 ID | `200 OK`, `image/png` 또는 `image/svg+xml` 이미지 바이트 | 백엔드가 파일 응답 구현, 모델 에이전트가 이미지 생성 | `image_url`을 `fetch`로 조회한다. 성공한 이미지의 `Blob`을 객체 URL로 만들어 `<img>`에 표시한다. |
| `GET /api/transcriptions/{id}/audio.mp3` | `download_url`에 포함된 작업 ID | `200 OK`, `audio/mpeg`로 업로드한 원본 바이트 반환. 헤더는 `Content-Disposition: attachment; filename="audio.mp3"`. | 백엔드가 파일 응답 구현, 프론트엔드가 다운로드 동작 구현 | 버튼을 누르면 `download_url`을 `fetch`로 조회한다. 성공한 MP3의 `Blob`을 객체 URL로 만들어 `audio.mp3`로 내려받게 한다. |

### 업로드 성공 응답

`image_url`과 `download_url`은 둘 다 `/`로 시작하는 같은 서버의 경로다. 백엔드는 업로드마다 UUIDv4를 하나 만들고 두 URL에 같은 ID를 넣는다.

```json
{
  "image_url": "/api/transcriptions/9af66c6c-baff-4916-9122-22011a94a4a3/score",
  "download_url": "/api/transcriptions/9af66c6c-baff-4916-9122-22011a94a4a3/audio.mp3"
}
```

백엔드는 서버 실행 중 유지되는 로컬 임시 디렉터리 안에 작업 ID별 폴더를 만든다. 원본 업로드를 `audio.mp3`로 저장하고 모델이 반환한 이미지를 같은 폴더에 저장한다. 두 파일을 저장한 뒤 `POST`에 응답하며, 이후 브라우저의 `GET` 요청에서 각 파일을 제공한다.

## 오류 응답과 복구

업로드와 결과 조회 API가 오류를 응답할 때는 `Content-Type: application/json`과 아래 본문 구조를 사용한다. 프론트엔드는 `error.code`로 복구 동작을 선택하고, 사용자에게 `error.message`를 표시한다.

```json
{
  "error": {
    "code": "UNSUPPORTED_AUDIO_FORMAT",
    "message": "MP3 파일을 선택해 주세요."
  }
}
```

| 요청 | HTTP 상태 | `error.code` | 백엔드가 반환하는 조건 | 프론트엔드 복구 동작 |
| --- | --- | --- | --- | --- |
| 업로드 | `400 Bad Request` | `INVALID_UPLOAD` | 멀티파트 요청을 읽을 수 없거나, `file` 필드에 파일이 정확히 한 개 들어 있지 않거나, 파일이 비어 있다. | MP3 파일 한 개를 다시 선택하도록 안내한다. |
| 업로드 | `415 Unsupported Media Type` | `UNSUPPORTED_AUDIO_FORMAT` | 모델 연동 코드가 파일의 실제 형식을 확인했고, MP3가 아니다. | MP3 파일을 다시 선택하도록 안내한다. |
| 업로드 | `422 Unprocessable Content` | `INVALID_AUDIO` | 파일 형식을 식별할 수 없거나 MP3를 디코딩할 수 없다. | 읽을 수 있는 MP3를 다시 선택하도록 안내한다. |
| 업로드 | `503 Service Unavailable` | `MODEL_UNAVAILABLE` | 서버 시작 때 CPU 모델을 로딩하지 못했다. | 모델 파일과 실행 환경을 확인하고 로컬 서버를 재시작한 뒤, 선택한 파일을 다시 제출하도록 안내한다. |
| 업로드 | `500 Internal Server Error` | `TRANSCRIPTION_FAILED` | 음원 전처리 이후 모델 추론에 실패했다. | 선택한 파일을 유지하고, 사용자가 다시 제출할 수 있게 한다. |
| 업로드·결과 조회 | `500 Internal Server Error` | `RESULT_FAILED` | 채보 이미지를 생성하지 못했거나, 작업 파일을 저장하지 못했거나, 존재하는 결과 파일을 열지 못했다. | 업로드에서 실패하면 다시 제출할 수 있게 한다. 조회에서 실패하면 해당 결과만 같은 URL로 다시 요청할 수 있게 한다. |
| 결과 조회 | `404 Not Found` | `RESULT_NOT_FOUND` | ID가 UUIDv4 형식이 아니거나 해당 결과가 없다. 서버 재시작으로 임시 파일이 사라진 경우도 포함한다. | 이미 있는 결과는 유지하고, 사라진 결과가 필요하면 원본 음원을 다시 제출하도록 안내한다. |
| 업로드·결과 조회 | `500 Internal Server Error` | `INTERNAL_ERROR` | 위에서 분류하지 못한 서버 예외다. | 업로드면 다시 제출, 결과 조회면 같은 URL로 다시 요청할 수 있게 한다. |

### 백엔드 처리 순서

1. 서버 시작 시 모델을 로딩한다. `ModelLoadError`가 발생하면 화면 제공은 계속하고, 업로드를 받을 때 로딩 실패 상태를 사용한다.
2. 업로드 형식과 파일 개수·크기를 확인한다. 여기서 크기 검사는 빈 파일을 판별하는 데 사용한다. 멀티파트 파서 오류와 FastAPI의 업로드 요청 검증 오류는 `400 INVALID_UPLOAD`로 변환한다. 요청이 유효하고 모델을 로딩하지 못한 상태면 `503 MODEL_UNAVAILABLE`을 반환한다. [FastAPI 예외 처리 문서](https://fastapi.tiangolo.com/tutorial/handling-errors/)
3. 음원을 저장한 뒤 모델을 호출한다. 모델 어댑터는 디코더가 판별한 실제 음원 형식으로 검사한다. 모델 예외는 아래 Python 계약에 따라 변환한다. 저장과 이미지 생성이 끝나면 결과 URL을 반환하고, 그 전에 실패하면 해당 요청에서 생성한 임시 파일을 정리한다.
4. 결과 조회에서는 상태와 응답 헤더를 보내기 전에 결과 파일을 열어 본다. 파일이 없으면 `404 RESULT_NOT_FOUND`, 파일이 있으나 열지 못하면 `500 RESULT_FAILED`를 반환한다. 결과 조회 응답에 `Cache-Control: no-store`를 넣어 재시도가 서버의 최신 상태를 확인하도록 한다.

### 프론트엔드 처리 순서

1. `fetch` 응답은 본문을 읽기 전에 `response.ok`를 확인한다. 오류 응답이면 JSON의 `error.code`와 `error.message`를 읽는다. 연결 오류가 나면 로컬 서버가 실행 중인지 확인하도록 안내하고, 본문 읽기 실패나 계약에 맞지 않는 응답은 응답 처리 오류로 안내한다. [Fetch 응답 처리 문서](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch#handling_the_response)
2. 업로드 요청이 끝나면 제출 버튼을 다시 활성화한다. 네트워크 오류나 성공 JSON 누락으로 결과 URL을 못 받았다면 선택한 파일을 유지하고 다시 제출할 수 있게 한다. 업로드를 반복하면 채보가 중복 실행될 수 있으므로 재제출은 사용자 동작으로 시작한다.
3. 채보 이미지와 MP3 다운로드는 개별 상태로 관리한다. 네트워크·본문 읽기 오류, 예상과 다른 응답 형식 또는 이미지 표시 실패가 발생하면 해당 결과만 같은 URL로 다시 요청할 수 있게 한다. 사용할 수 있는 다른 결과는 화면에 유지한다. `404 RESULT_NOT_FOUND`는 원본 재제출로 복구한다. 이미지 교체나 화면 종료 시 사용이 끝난 객체 URL을 해제한다.

## 백엔드와 모델 간 Python 인터페이스

모델 에이전트가 `backend/transcription.py`에서 아래 인터페이스를 구현하고, 백엔드 에이전트가 `backend/main.py`에서 호출한다.

| 인터페이스 | 입력 | 반환 | 호출 시점과 역할 |
| --- | --- | --- | --- |
| `load_model(model_dir: Path) -> Transcriber` | 모델 파일이 들어 있는 디렉터리의 절대 경로. 백엔드는 저장소 루트의 `models/` 경로를 전달한다. | CPU에서 추론할 수 있도록 로딩한 `Transcriber` 인스턴스 | 백엔드가 서버 시작 시 한 번 호출하고, 반환된 인스턴스를 업로드 처리에 사용한다. 모델 에이전트가 디렉터리 안의 파일명과 로딩 방식을 결정한다. |
| `Transcriber.transcribe(audio_path: Path) -> ScoreImage` | 백엔드가 저장한 원본 MP3의 절대 경로 | `ScoreImage` 객체. `content`는 완성된 이미지의 `bytes`, `media_type`은 `image/png` 또는 `image/svg+xml`. | 업로드 한 건당 한 번 호출한다. 모델 에이전트가 MP3 전처리, CPU 추론, 악보 렌더링을 수행한다. 백엔드는 `content`를 파일로 저장하고 `media_type`을 이미지 응답의 `Content-Type`으로 사용한다. |

모델 에이전트는 `backend/transcription.py`에서 아래 이름의 `Exception` 하위 클래스를 공개한다. 백엔드는 예외 클래스에 따라 HTTP 응답을 선택한다.

| 발생 위치 | Python 예외 | 발생 조건 | 백엔드 변환 |
| --- | --- | --- | --- |
| `load_model()` | `ModelLoadError` | 모델 파일 또는 CPU 실행 환경 문제로 초기화할 수 없다. | 로딩 실패 상태를 기록하고 유효한 업로드 요청에 `503 MODEL_UNAVAILABLE`. |
| `transcribe()` | `UnsupportedAudioFormatError` | 입력이 MP3 이외의 형식으로 확인됐다. | `415 UNSUPPORTED_AUDIO_FORMAT` |
| `transcribe()` | `InvalidAudioError` | 형식을 판별할 수 없거나 필요한 MP3 샘플을 읽을 수 없다. | `422 INVALID_AUDIO` |
| `transcribe()` | `InferenceError` | CPU 모델 추론에 실패했다. | `500 TRANSCRIPTION_FAILED` |
| `transcribe()` | `ScoreRenderError` | 추론 결과로 유효한 PNG 또는 SVG를 만들 수 없다. | `500 RESULT_FAILED` |

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class ScoreImage:
    content: bytes
    media_type: Literal["image/png", "image/svg+xml"]


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> ScoreImage: ...


def load_model(model_dir: Path) -> Transcriber: ...
```
