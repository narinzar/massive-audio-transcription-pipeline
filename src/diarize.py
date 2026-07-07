"""Lightweight speaker segmentation and label merging.

This is a deliberately simple, dependency-free stand-in for a real diarizer.
It segments on pauses and assigns speaker labels by clustering a cheap per-
segment feature (mean log-energy) into `num_speakers` groups with 1-D k-means.
It is documented as swappable: replace `diarize_waveform` with a call to
pyannote.audio (which needs an HF_TOKEN) and keep `merge_speaker_labels`
unchanged, since the merge step only consumes (start, end, speaker) turns.

The point of this module in the pipeline is to demonstrate the merge contract:
given transcript segments and speaker turns, attach a speaker to each segment
by maximum time overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def _segment_on_pauses(
    energy: np.ndarray,
    frame_sec: float,
    silence_threshold: float,
    min_silence: float,
    min_turn: float,
) -> List[tuple[float, float]]:
    """Return voiced (start, end) spans separated by silence gaps."""
    voiced = energy >= silence_threshold
    min_silence_frames = max(1, int(round(min_silence / frame_sec)))
    spans: List[tuple[float, float]] = []
    seg_start: Optional[int] = None
    silence_run = 0
    for i, v in enumerate(voiced):
        if v:
            if seg_start is None:
                seg_start = i
            silence_run = 0
        else:
            if seg_start is not None:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    seg_end = i - silence_run + 1
                    spans.append((seg_start * frame_sec, seg_end * frame_sec))
                    seg_start = None
    if seg_start is not None:
        spans.append((seg_start * frame_sec, len(energy) * frame_sec))
    # Drop spans shorter than min_turn.
    return [(s, e) for s, e in spans if (e - s) >= min_turn]


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    """Tiny 1-D k-means; returns a cluster label per value.

    Deterministic init via quantiles so results are reproducible without a
    random seed dependency.
    """
    if len(values) == 0:
        return np.array([], dtype=int)
    k = min(k, len(np.unique(values)))
    if k <= 1:
        return np.zeros(len(values), dtype=int)
    quantiles = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(values, quantiles)
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iters):
        dists = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = values[labels == c]
            if len(members):
                centers[c] = members.mean()
    return labels


def diarize_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    num_speakers: int = 2,
    frame_ms: float = 30.0,
    silence_threshold: Optional[float] = None,
    min_silence: float = 0.4,
    min_turn: float = 0.5,
) -> List[SpeakerTurn]:
    """Produce speaker turns for a mono waveform.

    Swap point: for real diarization, replace the body of this function with a
    pyannote.audio pipeline call and return SpeakerTurn objects. Everything
    downstream (merge_speaker_labels) stays the same.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = int(np.ceil(len(waveform) / frame_len))
    if n_frames == 0:
        return []
    padded = np.zeros(n_frames * frame_len, dtype=np.float64)
    padded[: len(waveform)] = waveform.astype(np.float64)
    frames = padded.reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    frame_sec = frame_len / sample_rate

    if silence_threshold is None:
        silence_threshold = 0.5 * float(np.mean(energy))

    spans = _segment_on_pauses(
        energy, frame_sec, silence_threshold, min_silence, min_turn
    )
    if not spans:
        return []

    # Feature per span: mean log-energy over its frames.
    feats = []
    for s, e in spans:
        i0 = int(s / frame_sec)
        i1 = max(i0 + 1, int(e / frame_sec))
        feats.append(float(np.log(np.mean(energy[i0:i1]) + 1e-8)))
    labels = _kmeans_1d(np.asarray(feats), num_speakers)

    return [
        SpeakerTurn(start=s, end=e, speaker=f"SPEAKER_{int(lbl):02d}")
        for (s, e), lbl in zip(spans, labels)
    ]


def merge_speaker_labels(
    segments: List[Dict[str, Any]],
    turns: List[SpeakerTurn],
    default_speaker: str = "SPEAKER_00",
) -> List[Dict[str, Any]]:
    """Attach a speaker to each transcript segment by max time overlap.

    Each returned segment is a shallow copy of the input with a "speaker" key.
    If a segment overlaps no turn, `default_speaker` is used.
    """
    out: List[Dict[str, Any]] = []
    for seg in segments:
        s_start = float(seg["start"])
        s_end = float(seg["end"])
        best_speaker = default_speaker
        best_overlap = 0.0
        for turn in turns:
            overlap = min(s_end, turn.end) - max(s_start, turn.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        merged = dict(seg)
        merged["speaker"] = best_speaker
        out.append(merged)
    return out
