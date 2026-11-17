"""Public model adapter contract and CPU WebScore loader.

Weights are supplied locally; this module never downloads or trains a model.
"""
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
    """The input is recognized, but is not MP3."""


class InvalidAudioError(Exception):
    """The input cannot be identified or decoded."""


class InferenceError(Exception):
    """Model inference failed."""


class ScoreRenderError(Exception):
    """The predicted notes could not be rendered."""


def load_model(model_dir: Path) -> Transcriber:
    try:
        from .webscore_cpu import WebScoreTranscriber
        return WebScoreTranscriber.load(Path(model_dir))
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError("Could not initialize the local CPU WebScore adapter") from exc
