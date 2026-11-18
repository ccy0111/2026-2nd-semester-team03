"""Model URL manifests for browser inference, with optional private-R2 signing.

The MVP app exposes load_runtime_manifest() through its public manifest endpoint.
Hosts that add authentication may use create_model_manifest_router() instead.
"""
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urljoin, urlsplit
from urllib.request import Request as UrlRequest, urlopen

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

HEADERS = {"Cache-Control": "no-store, private", "Pragma": "no-cache", "Vary": "Cookie, Authorization"}
CATALOG_LIMIT = 64 * 1024
BASE_FILES = ("config.json", "tokenizer.json", "mel_encoder.onnx", "token_embed.onnx")
VERSION_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
DEFAULT_MANIFEST_URL = "https://models.midistudio.app/web-student-kd/current.json"


def validate_model_url(value, *, allow_query=True):
    if not isinstance(value, str) or not value:
        raise ValueError("Model URL must be a nonempty string")
    url = urlsplit(value)
    local = url.hostname in ("localhost", "127.0.0.1", "::1")
    if (
        not url.hostname
        or not (url.scheme == "https" or (local and url.scheme == "http"))
        or url.username is not None
        or url.password is not None
        or url.fragment
        or (url.query and not allow_query)
        or any(character.isspace() for character in value)
        or "\\" in value
        or (url.port is not None and url.port < 1)
    ):
        raise ValueError("Invalid model URL")
    return value


def load_cdn_manifest(url):
    request = UrlRequest(
        validate_model_url(url),
        headers={"Accept": "application/json", "Cache-Control": "no-cache", "User-Agent": "MidiStudio/1.0"},
    )
    with urlopen(request, timeout=5) as response:
        source_url = validate_model_url(response.geturl())
        raw = response.read(CATALOG_LIMIT + 1)
    if len(raw) > CATALOG_LIMIT:
        raise ValueError("Model manifest too large")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schemaVersion") != 1 or value.get("format") != "webscore-onnx-v1":
        raise ValueError("Invalid model manifest")
    version, precision = value.get("version"), value.get("precision")
    if not isinstance(version, str) or not re.fullmatch(VERSION_PATTERN, version) or ".." in version or precision not in ("int8", "fp32"):
        raise ValueError("Invalid model version or precision")
    decoder = "decoder_kv.int8.onnx" if precision == "int8" else "decoder_kv.onnx"
    names = (*BASE_FILES, decoder)
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != set(names):
        raise ValueError("Incomplete model manifest")
    urls = {}
    for name in names:
        if not isinstance(files[name], str) or not files[name]:
            raise ValueError("Invalid model file URL")
        urls[name] = validate_model_url(urljoin(source_url, files[name]))
    return {"schemaVersion": 1, "format": "webscore-onnx-v1", "version": version,
            "precision": precision, "files": urls}


def load_runtime_manifest():
    """Return one model version's URLs using server-side deployment settings."""
    source = os.getenv("MODEL_SOURCE", "manifest")
    if source == "manifest":
        return load_cdn_manifest(os.getenv("MODEL_MANIFEST_URL", DEFAULT_MANIFEST_URL))
    if source == "cdn":
        base_url = os.environ["MODEL_BASE_URL"].rstrip("/")
        version = os.environ["MODEL_VERSION"]
        precision = os.getenv("MODEL_PRECISION", "int8")
        validate_model_url(base_url, allow_query=False)
        if (
            not re.fullmatch(VERSION_PATTERN, version)
            or ".." in version
            or precision not in ("int8", "fp32")
        ):
            raise ValueError("Invalid model version or precision")
        decoder = "decoder_kv.int8.onnx" if precision == "int8" else "decoder_kv.onnx"
        return {
            "schemaVersion": 1,
            "format": "webscore-onnx-v1",
            "version": version,
            "precision": precision,
            "files": {name: f"{base_url}/{name}" for name in (*BASE_FILES, decoder)},
        }
    if source == "r2":
        settings = ManifestSettings.from_env()
        client = make_r2_client(settings)
        try:
            return ModelManifestService(settings, client).runtime_manifest()
        finally:
            client.close()
    raise ValueError("MODEL_SOURCE must be manifest, cdn or r2")


@dataclass(frozen=True)
class ManifestSettings:
    account_id: str
    bucket: str
    catalog_key: str = "web-student-kd/current.json"
    release_prefix: str = "web-student-kd/releases"
    expires_seconds: int = 3600
    access_key_id: str = field(default="", repr=False)
    secret_access_key: str = field(default="", repr=False)

    def __post_init__(self):
        if not re.fullmatch(r"[a-fA-F0-9]{32}", self.account_id):
            raise ValueError("Invalid R2 account identifier")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", self.bucket):
            raise ValueError("Invalid R2 bucket")
        for key in (self.catalog_key, self.release_prefix):
            if not isinstance(key, str) or not key or key.startswith("/") or any(x in key for x in ("..", "?", "#", "\\")) or "//" in key:
                raise ValueError("Invalid model object path")
        if self.release_prefix.endswith("/") or not 60 <= self.expires_seconds <= 3600:
            raise ValueError("Invalid model publishing settings")

    @classmethod
    def from_env(cls):
        try:
            return cls(
                account_id=os.environ["R2_MODEL_ACCOUNT_ID"], bucket=os.environ["R2_MODEL_BUCKET"],
                access_key_id=os.environ["R2_MODEL_ACCESS_KEY_ID"],
                secret_access_key=os.environ["R2_MODEL_SECRET_ACCESS_KEY"],
                catalog_key=os.getenv("R2_MODEL_CATALOG_KEY", "web-student-kd/current.json"),
                release_prefix=os.getenv("R2_MODEL_RELEASE_PREFIX", "web-student-kd/releases"),
                expires_seconds=int(os.getenv("R2_MODEL_URL_TTL_SECONDS", "3600")),
            )
        except (KeyError, ValueError, TypeError):
            raise ValueError("Model manifest server settings are missing or invalid") from None


def make_r2_client(settings: ManifestSettings):
    # Optional dependency: the rest of the project does not require boto3.
    if not settings.access_key_id or not settings.secret_access_key:
        raise ValueError("Model read credentials are required")
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", region_name="auto",
        endpoint_url=f"https://{settings.account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.access_key_id, aws_secret_access_key=settings.secret_access_key,
        config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=10,
                      retries={"total_max_attempts": 2}, s3={"addressing_style": "path"}),
    )


def validate_catalog(value, settings):
    if not isinstance(value, dict) or value.get("schemaVersion") != 1 or value.get("format") != "webscore-onnx-v1":
        raise ValueError("Invalid model catalog")
    version, precision = value.get("version"), value.get("precision")
    if not isinstance(version, str) or not re.fullmatch(VERSION_PATTERN, version) or ".." in version or precision not in ("int8", "fp32"):
        raise ValueError("Invalid model release")
    names = BASE_FILES + (("decoder_kv.int8.onnx" if precision == "int8" else "decoder_kv.onnx"),)
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != set(names):
        raise ValueError("Incomplete model release")
    for name in names:
        item = files[name]
        expected = f"{settings.release_prefix}/{version}/{name}"
        if not isinstance(item, dict) or item.get("key") != expected:
            raise ValueError("Model object is outside the release")
        size, sha = item.get("size"), item.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= 2 * 1024**3 or not isinstance(sha, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", sha):
            raise ValueError("Invalid model artifact metadata")
    return value


class ModelManifestService:
    def __init__(self, settings: ManifestSettings, s3_client):
        self.settings, self.s3 = settings, s3_client

    def runtime_manifest(self):
        # Fixed server-selected bucket/key; request parameters never select objects.
        obj = self.s3.get_object(Bucket=self.settings.bucket, Key=self.settings.catalog_key)
        body = obj["Body"]
        try:
            raw = body.read(CATALOG_LIMIT + 1)
            if len(raw) > CATALOG_LIMIT:
                raise ValueError("Model catalog too large")
            catalog = validate_catalog(json.loads(raw), self.settings)
        finally:
            body.close()
        # All URLs created from one catalog snapshot, never from separate latest files.
        files = {
            name: self.s3.generate_presigned_url(
                "get_object", Params={"Bucket": self.settings.bucket, "Key": item["key"]},
                ExpiresIn=self.settings.expires_seconds, HttpMethod="GET",
            )
            for name, item in catalog["files"].items()
        }
        return {"schemaVersion": 1, "format": "webscore-onnx-v1", "version": catalog["version"],
                "precision": catalog["precision"], "files": files}


def create_model_manifest_router(service: ModelManifestService, authorize: Callable):
    """Host hook must authenticate AND authorize the request and return True.

    No route is enabled unless the host explicitly mounts this router. No API key
    or bearer token is sent by query string. Hook/provider failures are redacted.
    """
    if not callable(authorize):
        raise TypeError("An explicit model authorization hook is required")
    router = APIRouter()

    @router.get("/api/model-manifest")
    async def model_manifest(request: Request):
        try:
            if inspect.iscoroutinefunction(authorize):
                allowed = await authorize(request)
            else:
                allowed = await run_in_threadpool(authorize, request)
                if inspect.isawaitable(allowed):
                    allowed = await allowed
        except Exception:
            allowed = False
        if allowed is not True:
            return JSONResponse({"error": "MODEL_ACCESS_DENIED"}, status_code=403, headers=HEADERS)
        try:
            result = await run_in_threadpool(service.runtime_manifest)
        except Exception:
            # Do not serialize exception causes (they can contain signed URLs/keys).
            return JSONResponse({"error": "MODEL_MANIFEST_UNAVAILABLE"}, status_code=503, headers=HEADERS)
        return JSONResponse(result, headers=HEADERS)

    return router
