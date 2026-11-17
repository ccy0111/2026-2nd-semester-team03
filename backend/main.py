import logging
import os
import shutil
from asyncio import Lock
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, Iterator
from uuid import UUID, uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from python_multipart.exceptions import MultipartParseError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import transcription


ROOT = Path(__file__).resolve().parents[1]
API_PATH = "/api/transcriptions"
SCORE_FILES = {"image/png": "score.png", "image/svg+xml": "score.svg"}
RESULT_HEADERS = {"Cache-Control": "no-store"}
FILE_CHUNK_SIZE = 64 * 1024
ERRORS = {
    "INVALID_UPLOAD": (400, "비어 있지 않은 MP3 파일 한 개를 제출해 주세요."),
    "UNSUPPORTED_AUDIO_FORMAT": (415, "MP3 파일을 선택해 주세요."),
    "INVALID_AUDIO": (422, "음원을 읽을 수 없습니다. 다른 MP3를 선택해 주세요."),
    "MODEL_UNAVAILABLE": (503, "모델 파일과 실행 환경을 확인하고 서버를 재시작해 주세요."),
    "TRANSCRIPTION_FAILED": (500, "채보에 실패했습니다. 음원을 다시 제출해 주세요."),
    "RESULT_FAILED": (500, "결과 파일을 처리하지 못했습니다. 다시 시도해 주세요."),
    "RESULT_NOT_FOUND": (404, "결과 파일이 없습니다. 원본 음원을 다시 제출해 주세요."),
    "INTERNAL_ERROR": (500, "서버에서 오류가 발생했습니다. 다시 시도해 주세요."),
}

logger = logging.getLogger(__name__)
router = APIRouter(prefix=API_PATH)


class ApiError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)

    def response(self) -> JSONResponse:
        status, message = ERRORS[self.code]
        return JSONResponse(
            {"error": {"code": self.code, "message": message}},
            status_code=status,
            headers=RESULT_HEADERS,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    with TemporaryDirectory(prefix="transcriptions-") as directory:
        app.state.results_dir = Path(directory)
        app.state.transcription_lock = Lock()
        app.state.transcriber = None
        try:
            app.state.transcriber = await run_in_threadpool(
                transcription.load_model, ROOT / "models"
            )
        except transcription.ModelLoadError as exc:
            logger.warning("Transcription model unavailable: %s", exc)
        yield


def render_score(model: transcription.Transcriber, audio_path: Path) -> transcription.ScoreImage:
    try:
        image = model.transcribe(audio_path)
    except transcription.UnsupportedAudioFormatError as exc:
        raise ApiError("UNSUPPORTED_AUDIO_FORMAT") from exc
    except transcription.InvalidAudioError as exc:
        raise ApiError("INVALID_AUDIO") from exc
    except transcription.InferenceError as exc:
        raise ApiError("TRANSCRIPTION_FAILED") from exc
    except transcription.ScoreRenderError as exc:
        raise ApiError("RESULT_FAILED") from exc

    if (
        image.media_type not in SCORE_FILES
        or not isinstance(image.content, bytes)
        or not image.content
    ):
        raise ApiError("RESULT_FAILED")
    return image


def save_transcription(model: transcription.Transcriber, work: Path, source: BinaryIO) -> None:
    audio = work / "audio.mp3"
    with ExitStack() as cleanup:
        try:
            work.mkdir()
            cleanup.callback(shutil.rmtree, work, ignore_errors=True)
            with audio.open("wb") as output:
                shutil.copyfileobj(source, output)
        except OSError as exc:
            raise ApiError("RESULT_FAILED") from exc

        image = render_score(model, audio)
        try:
            (work / SCORE_FILES[image.media_type]).write_bytes(image.content)
        except OSError as exc:
            raise ApiError("RESULT_FAILED") from exc

        # Keep both files available for GET requests after the upload replies.
        cleanup.pop_all()


@router.post(
    "",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {"file": {"type": "string", "format": "binary"}},
                    }
                }
            },
        }
    },
)
async def create_transcription(request: Request) -> dict[str, str]:
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "multipart/form-data":
        raise ApiError("INVALID_UPLOAD")

    try:
        form = await request.form(max_files=1)
    except (StarletteHTTPException, MultipartParseError) as exc:
        raise ApiError("INVALID_UPLOAD") from exc

    try:
        uploads = form.getlist("file")
        if len(uploads) != 1 or not isinstance(uploads[0], UploadFile):
            raise ApiError("INVALID_UPLOAD")
        upload = uploads[0]
        if upload.size == 0:
            raise ApiError("INVALID_UPLOAD")

        state = request.app.state
        if state.transcriber is None:
            raise ApiError("MODEL_UNAVAILABLE")

        job_id = str(uuid4())
        # Keep the shared model safe even when uploads arrive from multiple tabs.
        async with state.transcription_lock:
            await run_in_threadpool(
                save_transcription, state.transcriber, state.results_dir / job_id, upload.file
            )
    finally:
        await form.close()

    return {
        "image_url": f"{API_PATH}/{job_id}/score",
        "download_url": f"{API_PATH}/{job_id}/audio.mp3",
    }


def file_chunks(file: BinaryIO) -> Iterator[bytes]:
    with file:
        while chunk := file.read(FILE_CHUNK_SIZE):
            yield chunk


def result_response(
    request: Request, id: str, files: dict[str, str], *, download: bool = False
) -> StreamingResponse:
    try:
        job_id = UUID(id)
    except ValueError as exc:
        raise ApiError("RESULT_NOT_FOUND") from exc
    if job_id.version != 4 or str(job_id) != id.lower():
        raise ApiError("RESULT_NOT_FOUND")

    work = request.app.state.results_dir / str(job_id)
    for media_type, filename in files.items():
        try:
            file = (work / filename).open("rb")
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ApiError("RESULT_FAILED") from exc

        headers = dict(RESULT_HEADERS)
        if download:
            headers["Content-Disposition"] = 'attachment; filename="audio.mp3"'
        return StreamingResponse(
            file_chunks(file),
            media_type=media_type,
            headers=headers,
            background=BackgroundTask(file.close),
        )

    raise ApiError("RESULT_NOT_FOUND")


@router.get("/{id}/score")
def get_score(request: Request, id: str) -> StreamingResponse:
    return result_response(request, id, SCORE_FILES)


@router.get("/{id}/audio.mp3")
def get_audio(request: Request, id: str) -> StreamingResponse:
    return result_response(request, id, {"audio/mpeg": "audio.mp3"}, download=True)


def create_app() -> FastAPI:
    frontend_dir = (ROOT / os.environ.get("FRONTEND_DIR", "frontend")).resolve()
    app = FastAPI(title="채보 플랫폼 MVP API", lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return ApiError("INTERNAL_ERROR").response()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        entrypoint = frontend_dir / "index.html"
        if not entrypoint.is_file():
            raise HTTPException(404, "프론트엔드 index.html이 아직 준비되지 않았습니다.")
        return FileResponse(entrypoint, media_type="text/html")

    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    return app


app = create_app()
