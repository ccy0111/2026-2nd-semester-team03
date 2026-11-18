"""HTTP contracts for browser inference and static module delivery."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import create_app


class BrowserApiTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {
            "MODEL_SOURCE": "cdn",
            "MODEL_BASE_URL": "https://models.example.com/piano-v1/",
            "MODEL_VERSION": "piano-v1",
        }, clear=True))

    def test_manifest_exposes_one_complete_cdn_version(self):
        with TestClient(create_app()) as client:
            response = client.get("/api/model-manifest")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertIn("no-store", response.headers["cache-control"])
        manifest = response.json()
        self.assertEqual(manifest["schemaVersion"], 1)
        self.assertEqual(manifest["format"], "webscore-onnx-v1")
        self.assertEqual(manifest["version"], "piano-v1")
        self.assertEqual(manifest["precision"], "int8")
        self.assertEqual(set(manifest["files"]), {
            "config.json", "tokenizer.json", "mel_encoder.onnx",
            "token_embed.onnx", "decoder_kv.int8.onnx",
        })
        for name, url in manifest["files"].items():
            self.assertEqual(url, f"https://models.example.com/piano-v1/{name}")

    def test_fp32_manifest_uses_matching_decoder(self):
        with patch.dict(os.environ, {"MODEL_PRECISION": "fp32"}):
            with TestClient(create_app()) as client:
                manifest = client.get("/api/model-manifest").json()
        self.assertEqual(manifest["precision"], "fp32")
        self.assertIn("decoder_kv.onnx", manifest["files"])
        self.assertNotIn("decoder_kv.int8.onnx", manifest["files"])

    def test_missing_settings_returns_documented_error(self):
        for key in ("MODEL_BASE_URL", "MODEL_VERSION"):
            with self.subTest(key=key), patch.dict(os.environ, {}, clear=True):
                os.environ.update({
                    "MODEL_SOURCE": "cdn",
                    "MODEL_BASE_URL": "https://models.example.com/v1",
                    "MODEL_VERSION": "v1",
                })
                del os.environ[key]
                with TestClient(create_app()) as client:
                    response = client.get("/api/model-manifest")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["error"]["code"], "MODEL_MANIFEST_UNAVAILABLE")
                self.assertTrue(response.json()["error"]["message"])
                self.assertIn("no-store", response.headers["cache-control"])

    def test_invalid_deployment_settings_are_rejected(self):
        for settings in (
            {"MODEL_BASE_URL": "http://models.example.com/v1"},
            {"MODEL_BASE_URL": "https://models.example.com/v1?signature=SECRET"},
            {"MODEL_BASE_URL": "https://user:SECRET@models.example.com/v1"},
            {"MODEL_BASE_URL": "https://models.example.com/v1#part"},
            {"MODEL_BASE_URL": "https://models.example.com:invalid/v1"},
            {"MODEL_BASE_URL": "https://models.example.com/a b"},
            {"MODEL_VERSION": "../v1"},
            {"MODEL_PRECISION": "fp16"},
            {"MODEL_SOURCE": "unknown"},
        ):
            with self.subTest(settings=settings), patch.dict(os.environ, settings):
                with TestClient(create_app()) as client:
                    response = client.get("/api/model-manifest")
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("SECRET", response.text)

    def test_browser_modules_and_relative_imports_are_served(self):
        with TestClient(create_app()) as client:
            for name in (
                "model-version-client.mjs", "model-client.mjs",
                "mel-frontend.mjs", "midi-decoder.mjs", "model-config.mjs",
            ):
                with self.subTest(name=name):
                    response = client.get(f"/inference/{name}")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("javascript", response.headers["content-type"])
                    self.assertIn("export ", response.text)

    def test_built_frontend_is_served_from_configured_directory(self):
        with TemporaryDirectory() as directory:
            frontend = Path(directory)
            (frontend / "index.html").write_text("<html>frontend</html>", encoding="utf-8")
            (frontend / "assets").mkdir()
            (frontend / "assets" / "app.js").write_text("export const ready = true;", encoding="utf-8")
            with patch.dict(os.environ, {"FRONTEND_DIR": directory}):
                with TestClient(create_app()) as client:
                    self.assertEqual(client.get("/").text, "<html>frontend</html>")
                    self.assertEqual(client.get("/static/assets/app.js").status_code, 200)
                    self.assertEqual(client.get("/api/model-manifest").status_code, 200)

    def test_model_provider_failure_is_redacted(self):
        with patch("backend.main.load_runtime_manifest", side_effect=RuntimeError("SECRET URL")):
            with TestClient(create_app()) as client:
                response = client.get("/api/model-manifest")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("SECRET", response.text)

    def test_transcription_is_client_side_and_api_schema_has_manifest(self):
        with TestClient(create_app()) as client:
            schema = client.get("/openapi.json").json()
            self.assertIn("/api/model-manifest", schema["paths"])
            self.assertNotIn("/api/transcriptions", schema["paths"])
            self.assertEqual(client.post("/api/transcriptions").status_code, 404)


if __name__ == "__main__":
    unittest.main()
