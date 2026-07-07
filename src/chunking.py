"""Split long audio into chunk time ranges.

Two strategies:
    fixed_windows      - fixed-length windows with a small overlap.
    energy_vad_windows - split on low-energy (silence) gaps, then cap segment
                         length so no single chunk is too long for the model.

A chunk is a plain dataclass carrying an integer index and a [start, end)
time range in seconds. The pipeline only needs the time ranges; the actual
audio samples are sliced lazily at transcription time from the source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Chunk:
    """A single audio window.

    index: position in the ordered chunk list (used as the checkpoint key).
    start: window start in seconds (inclusive).
    end:   window end in seconds (exclusive).
    """

    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def fixed_windows(
    total_duration: float,
    window: float = 30.0,
    overlap: float = 1.0,
) -> List[Chunk]:
    """Cover [0, total_duration) with fixed-length overlapping windows.

    Consecutive windows advance by (window - overlap) seconds so that each pair
    of neighbours shares `overlap` seconds of context. The final window is
    clamped to total_duration, so it may be shorter than `window`.

    Guarantees:
        - windows are ordered and contiguous in coverage (union == whole clip).
        - every window has positive duration.
        - overlap between neighbour i and i+1 is exactly `overlap` (except the
          last window, which may be clamped short).
    """
    if total_duration <= 0:
        return []
    if window <= 0:
        raise ValueError("window must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")

    step = window - overlap
    chunks: List[Chunk] = []
    start = 0.0
    index = 0
    while start < total_duration:
        end = min(start + window, total_duration)
        chunks.append(Chunk(index=index, start=start, end=end))
        index += 1
        if end >= total_duration:
            break
        start += step
    return chunks


def _frame_energy(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float,
) -> tuple[np.ndarray, float]:
    """Return per-frame RMS energy and the frame length in seconds."""
    frame_len = max(1, int(sample_rate * frame_ms / 1000.0))
    n_frames = int(np.ceil(len(samples) / frame_len))
    padded = np.zeros(n_frames * frame_len, dtype=np.float64)
    padded[: len(samples)] = samples.astype(np.float64)
    frames = padded.reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    return energy, frame_len / sample_rate


def energy_vad_windows(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: float = 30.0,
    silence_threshold: Optional[float] = None,
    min_silence: float = 0.3,
    max_window: float = 30.0,
    overlap: float = 1.0,
) -> List[Chunk]:
    """Voice-activity split on low-energy gaps.

    Frames whose RMS energy falls below `silence_threshold` are treated as
    silence. A run of silence at least `min_silence` seconds long becomes a
    boundary. Any resulting voiced segment longer than `max_window` is further
    divided by `fixed_windows` so no chunk exceeds the model's comfortable
    input length.

    If `silence_threshold` is None it defaults to a fraction of the mean frame
    energy, which adapts to overall clip loudness.
    """
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # downmix to mono
    total_duration = len(samples) / sample_rate
    if total_duration <= 0:
        return []

    energy, frame_sec = _frame_energy(samples, sample_rate, frame_ms)
    if silence_threshold is None:
        silence_threshold = 0.5 * float(np.mean(energy))

    voiced = energy >= silence_threshold
    min_silence_frames = max(1, int(round(min_silence / frame_sec)))

    # Find voiced-segment boundaries by scanning silence runs.
    segments: List[tuple[float, float]] = []
    seg_start: Optional[int] = None
    silence_run = 0
    for i, is_voiced in enumerate(voiced):
        if is_voiced:
            if seg_start is None:
                seg_start = i
            silence_run = 0
        else:
            if seg_start is not None:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    seg_end = i - silence_run + 1
                    segments.append((seg_start * frame_sec, seg_end * frame_sec))
                    seg_start = None
                    silence_run = 0
    if seg_start is not None:
        segments.append((seg_start * frame_sec, total_duration))

    if not segments:
        # Whole clip was below threshold; fall back to fixed windows.
        return fixed_windows(total_duration, window=max_window, overlap=overlap)

    # Cap long segments and re-index globally.
    chunks: List[Chunk] = []
    index = 0
    for seg_start_s, seg_end_s in segments:
        seg_len = seg_end_s - seg_start_s
        if seg_len <= max_window:
            chunks.append(Chunk(index=index, start=seg_start_s, end=seg_end_s))
            index += 1
        else:
            for sub in fixed_windows(seg_len, window=max_window, overlap=overlap):
                chunks.append(
                    Chunk(
                        index=index,
                        start=seg_start_s + sub.start,
                        end=seg_start_s + sub.end,
                    )
                )
                index += 1
    return chunks


def total_covered_duration(chunks: List[Chunk]) -> float:
    """Union length of all chunk ranges (overlaps counted once).

    Useful for asserting that chunking covers the whole clip in tests.
    """
    if not chunks:
        return 0.0
    ordered = sorted(chunks, key=lambda c: c.start)
    covered = 0.0
    cur_start, cur_end = ordered[0].start, ordered[0].end
    for c in ordered[1:]:
        if c.start <= cur_end:
            cur_end = max(cur_end, c.end)
        else:
            covered += cur_end - cur_start
            cur_start, cur_end = c.start, c.end
    covered += cur_end - cur_start
    return covered
