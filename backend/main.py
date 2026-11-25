"""Serve the web application and model URLs for browser-side inference."""

import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .model_manifest import HEADERS, load_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_LAYER_SHA = "96b8c44f046412c9a55a733af30c66efeadd0add5e12731e5c637ec4c85d4f60"
LOCAL_LAYER_BYTES = 318110962


def _local_layer_decoder() -> Path | None:
    raw = os.environ.get("WEBSCORE_LOCAL_LAYER_DECODER")
    if not raw:
        return None
    try:
        p = Path(raw).resolve(strict=True)
        if not p.is_file() or p.stat().st_size != LOCAL_LAYER_BYTES:
            return None
        with p.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != LOCAL_LAYER_SHA:
                return None
        return p
    except OSError:
        return None


def create_app() -> FastAPI:
    frontend_dir = (ROOT / os.environ.get("FRONTEND_DIR", "frontend")).resolve()
    app = FastAPI(title="채보 플랫폼 MVP API")
    layer_model = _local_layer_decoder()

    @app.api_route("/api/local-preview/v2-grpo100-layerkv/decoder.onnx",
                   methods=["HEAD", "GET"], include_in_schema=False)
    def local_layer_decoder() -> FileResponse:
        if layer_model is None or not layer_model.is_file():
            raise HTTPException(404, "Local FP16 decoder not enabled")
        return FileResponse(layer_model, media_type="application/octet-stream",
                            headers={"Cache-Control": "private, max-age=3600"},
                            filename=None)

    @app.get(
        "/api/model-manifest",
        responses={503: {"description": "Model download settings are unavailable"}},
    )
    def model_manifest() -> JSONResponse:
        try:
            manifest = load_runtime_manifest()
        except Exception:
            # Provider errors may contain credentials or signed download URLs.
            return JSONResponse(
                {
                    "error": {
                        "code": "MODEL_MANIFEST_UNAVAILABLE",
                        "message": "모델 다운로드 정보를 가져올 수 없습니다. 설정을 확인하고 다시 시도해 주세요.",
                    }
                },
                status_code=503,
                headers=HEADERS,
            )
        return JSONResponse(manifest, headers=HEADERS)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        entrypoint = frontend_dir / "index.html"
        if not entrypoint.is_file():
            raise HTTPException(404, "프론트엔드 index.html이 아직 준비되지 않았습니다.")
        return FileResponse(entrypoint, media_type="text/html")

    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    app.mount(
        "/inference",
        StaticFiles(directory=ROOT / "webscore_inference" / "web"),
        name="inference",
    )
    return app


app = create_app()
