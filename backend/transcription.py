"""Contract implemented by the CPU transcription model adapter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True)
class ScoreImage:
    content: bytes
    media_type: Literal["image/png", "image/svg+xml"]


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> ScoreImage: ...


class ModelLoadError(Exception):
    """The CPU model could not be initialized."""


class UnsupportedAudioFormatError(Exception):
    """The input is a recognized format other than MP3."""


class InvalidAudioError(Exception):
    """The input could not be identified or decoded."""


class InferenceError(Exception):
    """Model inference failed."""


class ScoreRenderError(Exception):
    """The predicted notes could not be rendered as a score image."""


def load_model(model_dir: Path) -> Transcriber:
    # The model owner supplies the loader and Transcriber implementation here.
    raise ModelLoadError(f"CPU model adapter is not configured for {model_dir}")
