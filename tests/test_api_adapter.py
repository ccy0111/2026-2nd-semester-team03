"""Existing HTTP contract with model-independent adapters; never load real weights."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import create_app
from backend import transcription as t

class AdapterApiTests(unittest.TestCase):
    def test_missing_model_is_503(self):
        with patch('backend.main.transcription.load_model', side_effect=t.ModelLoadError('absent')):
            with TestClient(create_app()) as client:
                r = client.post('/api/transcriptions', files={'file': ('a.mp3', b'audio', 'audio/mpeg')})
                self.assertEqual(r.status_code, 503)
                self.assertEqual(r.json()['error']['code'], 'MODEL_UNAVAILABLE')

    def test_adapter_errors_preserve_documented_http_status(self):
        cases = [(t.UnsupportedAudioFormatError,415,'UNSUPPORTED_AUDIO_FORMAT'),
                 (t.InvalidAudioError,422,'INVALID_AUDIO'),
                 (t.InferenceError,500,'TRANSCRIPTION_FAILED'),
                 (t.ScoreRenderError,500,'RESULT_FAILED')]
        for error, status, code in cases:
            with self.subTest(error=error):
                class Fake:
                    def transcribe(self, path):
                        raise error('test')
                with patch('backend.main.transcription.load_model', return_value=Fake()):
                    with TestClient(create_app()) as client:
                        r = client.post('/api/transcriptions', files={'file': ('a.mp3', b'audio', 'audio/mpeg')})
                        self.assertEqual(r.status_code,status)
                        self.assertEqual(r.json()['error']['code'],code)
                        self.assertEqual(list(client.app.state.results_dir.iterdir()), [])

    def test_svg_and_original_mp3_routes(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
        original = b'original-test-mp3'
        class Fake:
            def transcribe(self, path):
                assert path.read_bytes() == original
                return t.ScoreImage(svg,'image/svg+xml')
        with patch('backend.main.transcription.load_model', return_value=Fake()):
            with TestClient(create_app()) as client:
                r = client.post('/api/transcriptions', files={'file': ('a.mp3', original, 'audio/mpeg')})
                self.assertEqual(r.status_code,200)
                score = client.get(r.json()['image_url'])
                self.assertEqual(score.content,svg)
                self.assertTrue(score.headers['content-type'].startswith('image/svg+xml'))
                audio = client.get(r.json()['download_url'])
                self.assertEqual(audio.content,original)
                self.assertIn('attachment',audio.headers['content-disposition'])

if __name__ == '__main__': unittest.main()
