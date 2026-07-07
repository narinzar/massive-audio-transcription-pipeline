"""Transcribe audio chunks with Whisper.

The default backend is faster-whisper (CTranslate2), which supports batched
GPU inference and is markedly quicker than the reference openai-whisper
implementation. openai-whisper is kept as a fallback backend.

Design notes:
    - The model is loaded once per worker process and cached module-level, so a
      process pool pays the load cost once, not once per chunk.
    - `transcribe_chunk` slices the requested [start, end) window from an
      already-loaded mono waveform and returns segments with timestamps shifted
      back into the original clip's timeline.
    - A segment is a dict: {"start": float, "end": float, "text": str}.

The heavy model call lives behind `_run_model`, which tests monkeypatch so no
model download is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Cached backend handle, keyed by (backend, model_name, device, compute_type).
_MODEL_CACHE: Dict[tuple, Any] = {}


@dataclass
class TranscribeConfig:
    backend: str = "faster-whisper"  # or "openai-whisper"
    model_name: str = "base"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    compute_type: str = "float16"  # faster-whisper compute type on GPU
    language: Optional[str] = None
    beam_size: int = 5


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model(config: TranscribeConfig) -> Any:
    """Load and cache the transcription backend for this configuration."""
    device = _select_device(config.device)
    key = (config.backend, config.model_name, device, config.compute_type)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if config.backend == "faster-whisper":
        from faster_whisper import WhisperModel

        # CPU cannot use float16; fall back to int8 there.
        compute_type = config.compute_type if device == "cuda" else "int8"
        model = WhisperModel(
            config.model_name, device=device, compute_type=compute_type
        )
    elif config.backend == "openai-whisper":
        import whisper

        model = whisper.load_model(config.model_name, device=device)
    else:
        raise ValueError(f"unknown backend: {config.backend}")

    _MODEL_CACHE[key] = model
    return model


def _run_model(
    model: Any,
    audio: np.ndarray,
    config: TranscribeConfig,
) -> List[Dict[str, float | str]]:
    """Invoke the backend on a mono float32 waveform at 16 kHz.

    Returns a list of {"start", "end", "text"} in the *chunk-local* timeline
    (0 at the chunk's first sample). Tests monkeypatch this function.
    """
    if config.backend == "faster-whisper":
        segments, _info = model.transcribe(
            audio,
            language=config.language,
            beam_size=config.beam_size,
        )
        return [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments
        ]
    else:  # openai-whisper
        result = model.transcribe(audio, language=config.language)
        return [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"].strip(),
            }
            for s in result.get("segments", [])
        ]


def transcribe_chunk(
    waveform: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    config: TranscribeConfig,
    model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Transcribe the [start, end) window of `waveform`.

    Returns segments with timestamps expressed in the ORIGINAL clip timeline
    (chunk-local times are offset by `start`).
    """
    if model is None:
        model = load_model(config)

    a = max(0, int(round(start * sample_rate)))
    b = min(len(waveform), int(round(end * sample_rate)))
    clip = waveform[a:b].astype(np.float32)

    local_segments = _run_model(model, clip, config)
    out: List[Dict[str, Any]] = []
    for seg in local_segments:
        out.append(
            {
                "start": float(seg["start"]) + start,
                "end": float(seg["end"]) + start,
                "text": seg["text"],
            }
        )
    return out
