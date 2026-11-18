"""Serve the web application and model URLs for browser-side inference."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .model_manifest import HEADERS, load_runtime_manifest


ROOT = Path(__file__).resolve().parents[1]


def create_app() -> FastAPI:
    frontend_dir = (ROOT / os.environ.get("FRONTEND_DIR", "frontend")).resolve()
    app = FastAPI(title="채보 플랫폼 MVP API")

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
