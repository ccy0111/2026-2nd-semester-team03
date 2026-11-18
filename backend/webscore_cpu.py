"""CPU implementation of the browser WebScore ONNX contract."""
import json
import math
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .transcription import (
    InferenceError, InvalidAudioError, ModelLoadError, ScoreImage,
    ScoreRenderError, UnsupportedAudioFormatError,
)
from .webscore_score import render_score

SR = 16000
CHUNK = 80000
MAX_AUDIO_SECONDS = 1800
MAX_TOKENS = 1024


def read_metadata(model_dir):
    """Require metadata alongside the supplied model export."""
    values = []
    for name in ("config.json", "tokenizer.json"):
        path = model_dir / name
        try:
            values.append(json.loads(path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError) as exc:
            raise ModelLoadError(f"Missing or invalid {name}") from exc
    cfg, tok = values
    if not isinstance(cfg, dict) or not isinstance(tok, dict):
        raise ModelLoadError("Metadata must be JSON objects")
    for key in ("dim", "num_heads", "num_kv_heads", "num_layers", "card"):
        if type(cfg.get(key)) is not int or cfg[key] <= 0:
            raise ModelLoadError(f"Invalid config: {key}")
    if cfg["dim"] % cfg["num_heads"] or cfg["num_heads"] % cfg["num_kv_heads"]:
        raise ModelLoadError("Invalid attention head dimensions")
    if any(cfg.get(k) != v for k, v in {"sample_rate": SR, "n_fft": 2048,
                                     "frame_rate": 100, "n_mel_bins": 512}.items()):
        raise ModelLoadError("Unsupported WebScore frontend configuration")
    labels = tok.get("id_to_token")
    if not isinstance(labels, list) or len(labels) != cfg["card"] or not all(isinstance(s, str) for s in labels) or len(set(labels)) != len(labels):
        raise ModelLoadError("Tokenizer/config vocabulary mismatch")
    expected = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for key, label in zip(("pad_id", "bos_id", "eos_id", "unk_id"), expected):
        idx = tok.get(key)
        if type(idx) is not int or not 0 <= idx < len(labels) or labels[idx] != label:
            raise ModelLoadError(f"Invalid tokenizer {key}")
    for tk, ck in (("pad_id", "pad_token_id"), ("bos_id", "bos_token_id"),
                   ("eos_id", "eos_token_id"), ("unk_id", "unk_token_id")):
        if tok[tk] != cfg.get(ck):
            raise ModelLoadError("Config/tokenizer special token IDs differ")
    if tok.get("frame_rate_hz") != 100 or tok.get("chunk_seconds") != 5:
        raise ModelLoadError("Unsupported tokenizer timing")
    required = ["tie", "vel:0", "vel:1"]
    required += [f"pitch:{i}" for i in range(128)]
    required += [f"inst:{i}" for i in range(128)]
    required += [f"drum:{i}" for i in range(128)]
    required += [f"time:{i}" for i in range(501)]
    if any(label not in labels for label in required):
        raise ModelLoadError("Unsupported tokenizer event vocabulary")
    return cfg, tok


def decode_mp3(path, ffprobe, ffmpeg):
    """Inspect decoded format, not extension; no shell execution or network input."""
    try:
        probe = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
                                "format=format_name,duration:stream=codec_type,codec_name",
                                "-of", "json", str(Path(path).resolve())],
                               capture_output=True, timeout=30, check=True)
        info = json.loads(probe.stdout)
        fmt = info.get("format", {})
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        if not streams:
            raise InvalidAudioError("No audio stream")
        if "mp3" not in fmt.get("format_name", "").split(",") or streams[0].get("codec_name") != "mp3":
            raise UnsupportedAudioFormatError("Only MP3 is supported")
        # Duration metadata is advisory; the decoded sample count is checked too.
        duration = float(fmt.get("duration", 0))
        if math.isfinite(duration) and duration > MAX_AUDIO_SECONDS:
            raise InvalidAudioError("MP3 exceeds the 30-minute processing limit")
        with TemporaryDirectory(prefix="webscore-pcm-") as tmp:
            pcm_path = Path(tmp) / "decoded.f32"
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-i", str(Path(path).resolve()),
                            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR),
                            "-t", str(MAX_AUDIO_SECONDS + 1), "-f", "f32le", str(pcm_path)],
                           capture_output=True, timeout=180, check=True)
            if pcm_path.stat().st_size > MAX_AUDIO_SECONDS * SR * 4:
                raise InvalidAudioError("MP3 exceeds the 30-minute processing limit")
            pcm = np.fromfile(pcm_path, dtype="<f4")
        if not len(pcm) or not np.isfinite(pcm).all():
            raise InvalidAudioError("MP3 has no finite audio samples")
        return pcm
    except (InvalidAudioError, UnsupportedAudioFormatError):
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise InvalidAudioError("Unable to inspect or decode the MP3") from exc


class MelFrontend:
    def __init__(self):
        self.window = (0.5 * (1 - np.cos(2 * np.pi * np.arange(2048) / 2048))).astype(np.float32)
        mel_max = 2595 * np.log10(1 + 8000 / 700)
        points = 700 * (10 ** (np.linspace(0, mel_max, 514) / 2595) - 1)
        freq = np.linspace(0, 8000, 1025)
        down = (freq[:, None] - points[:-2]) / (points[1:-1] - points[:-2])
        up = (points[2:] - freq[:, None]) / (points[2:] - points[1:-1])
        self.fb = np.maximum(0, np.minimum(down, up)).astype(np.float32)

    def compute(self, pcm):
        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[:min(len(pcm), CHUNK)] = pcm[:CHUNK]
        padded = np.pad(chunk, (1024, 1024), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, 2048)[::160]
        mag = np.abs(np.fft.rfft(frames * self.window, axis=1)).astype(np.float32)
        return np.log(mag @ self.fb + np.float32(1e-6)).astype(np.float32)[None, :, :]


def decode_tokens(ids, labels, start, active, completed):
    """Maintain program/pitch notes over chunk boundaries, matching browser events."""
    events = []
    for idx in ids:
        label = labels[idx]
        if label == "<eos>":
            break
        events.append(label)
    tie_section = "tie" in events
    declared = set()
    time, program, velocity = 0.0, 0, 1
    for label in events:
        if label == "tie":
            tie_section = False
            for key in list(active):
                if key not in declared:
                    note = active.pop(key)
                    completed.append((*note, max(note[2] + .01, start)))
            continue
        if ":" not in label:
            continue
        kind, raw = label.split(":", 1)
        value = int(raw)
        if tie_section:
            if kind == "inst":
                program = value
            elif kind == "pitch":
                declared.add((program, value))
            continue
        if kind == "time":
            time = value / 100
        elif kind == "inst":
            program = value
        elif kind == "vel":
            velocity = value
        elif kind == "drum":
            completed.append((128, value, start + time, start + time + .05))
        elif kind == "pitch":
            key = (program, value)
            if key in active:
                note = active.pop(key)
                completed.append((*note, max(note[2] + .01, start + time)))
            if velocity > 0:
                active[key] = (program, value, start + time)


class WebScoreTranscriber:
    def __init__(self, cfg, tok, sessions, ffprobe, ffmpeg):
        self.cfg, self.tok = cfg, tok
        self.labels = tok["id_to_token"]
        self.ids = {s: i for i, s in enumerate(self.labels)}
        self.encoder, self.embed, self.decoder = sessions
        self.ffprobe, self.ffmpeg = ffprobe, ffmpeg
        self.frontend = MelFrontend()

    @classmethod
    def load(cls, model_dir):
        cfg, tok = read_metadata(model_dir)
        ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
        if not ffprobe or not ffmpeg:
            raise ModelLoadError("FFmpeg and ffprobe must be available on PATH")
        dec = model_dir / "decoder_kv.int8.onnx"
        if not dec.is_file():
            dec = model_dir / "decoder_kv.onnx"
        files = [model_dir / "mel_encoder.onnx", model_dir / "token_embed.onnx", dec]
        if not all(f.is_file() for f in files):
            raise ModelLoadError("Local ONNX weights are missing; no model is downloaded automatically")
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            sessions = [ort.InferenceSession(str(f), sess_options=opts,
                                            providers=["CPUExecutionProvider"]) for f in files]
            contracts = [({"mel"}, {"cond_embeds"}), ({"input_ids"}, {"token_embeds"}),
                         ({"inputs_embeds", "past_k", "past_v", "position_offset", "attn_mask"},
                          {"logits", "present_k", "present_v"})]
            for session, (inputs, outputs) in zip(sessions, contracts):
                if {x.name for x in session.get_inputs()} != inputs or not outputs <= {x.name for x in session.get_outputs()}:
                    raise ModelLoadError("ONNX input/output contract does not match WebScore")
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError("Could not load local ONNX weights on CPU") from exc
        return cls(cfg, tok, sessions, ffprobe, ffmpeg)

    def generate(self, pcm, prompt):
        cond, = self.encoder.run(["cond_embeds"], {"mel": self.frontend.compute(pcm)})
        embeds, = self.embed.run(["token_embeds"], {"input_ids": np.array([[self.tok["bos_id"], *prompt]], dtype=np.int64)})
        inputs = np.concatenate((cond, embeds), axis=1).astype(np.float32)
        c = self.cfg
        shape = (c["num_layers"], 1, c["num_kv_heads"], 0, c["dim"] // c["num_heads"])
        k, v = np.empty(shape, np.float32), np.empty(shape, np.float32)
        position = 0
        generated = list(prompt)
        for _ in range(MAX_TOKENS):
            seq, past = inputs.shape[1], k.shape[3]
            mask = np.where(np.arange(past + seq)[None, :] <= past + np.arange(seq)[:, None],
                            0, -1e9).astype(np.float32)[None, None]
            logits, k, v = self.decoder.run(["logits", "present_k", "present_v"], {
                "inputs_embeds": inputs, "past_k": k, "past_v": v,
                "position_offset": np.array([position], dtype=np.int64), "attn_mask": mask,
            })
            row = np.array(logits[0, -1], copy=True)
            if len(row) != len(self.labels) or not np.isfinite(row).all():
                raise InferenceError("Invalid model logits")
            row[[self.tok[x] for x in ("pad_id", "bos_id", "unk_id")]] = -np.inf
            token = int(row.argmax())
            generated.append(token)
            if token == self.tok["eos_id"]:
                return generated
            position += seq
            inputs, = self.embed.run(["token_embeds"], {"input_ids": np.array([[token]], dtype=np.int64)})
        # Do not silently render an incomplete sequence as a completed transcription.
        raise InferenceError("Model did not emit EOS within the per-chunk token limit")

    def transcribe(self, audio_path):
        pcm = decode_mp3(audio_path, self.ffprobe, self.ffmpeg)
        notes, active = [], {}
        try:
            for offset in range(0, len(pcm), CHUNK):
                prompt = []
                if offset:
                    current = None
                    for program, pitch in sorted(active):
                        if program != current:
                            prompt.append(self.ids[f"inst:{program}"])
                            current = program
                        prompt.append(self.ids[f"pitch:{pitch}"])
                    prompt.append(self.ids["tie"])
                tokens = self.generate(pcm[offset:offset + CHUNK], prompt)
                decode_tokens(tokens, self.labels, offset / SR, active, notes)
            end = len(pcm) / SR
            notes.extend((*n, max(n[2] + .01, end)) for n in active.values())
            # Discard predictions entirely inside the zero-padding after the source.
            notes = [(p, pitch, start, min(stop, end)) for p, pitch, start, stop in notes
                     if 0 <= start < end and 0 <= pitch <= 127 and 0 <= p <= 128]
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("WebScore CPU inference failed") from exc
        try:
            image = render_score(notes)
            return ScoreImage(image, "image/svg+xml")
        except Exception as exc:
            raise ScoreRenderError("Unable to render the score preview") from exc
