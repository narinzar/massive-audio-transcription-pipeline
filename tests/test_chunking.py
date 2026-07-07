"""Chunking coverage and overlap tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import (  # noqa: E402
    energy_vad_windows,
    fixed_windows,
    total_covered_duration,
)


def test_fixed_windows_cover_whole_clip():
    total = 100.0
    chunks = fixed_windows(total, window=30.0, overlap=1.0)
    # First chunk starts at 0, last chunk ends exactly at total.
    assert chunks[0].start == 0.0
    assert chunks[-1].end == pytest.approx(total)
    # Union of ranges covers the entire clip.
    assert total_covered_duration(chunks) == pytest.approx(total)


def test_fixed_windows_overlap_is_exact():
    chunks = fixed_windows(100.0, window=30.0, overlap=1.0)
    # Every neighbour pair except possibly the last shares exactly `overlap`.
    for a, b in zip(chunks, chunks[1:]):
        overlap = a.end - b.start
        # Last window may be clamped; only assert on full-length neighbours.
        if a.duration == pytest.approx(30.0):
            assert overlap == pytest.approx(1.0)


def test_fixed_windows_step_is_window_minus_overlap():
    chunks = fixed_windows(200.0, window=30.0, overlap=5.0)
    starts = [c.start for c in chunks]
    steps = np.diff(starts)
    # All steps (before the clamped final one) equal window - overlap = 25.
    assert np.allclose(steps[:-1], 25.0)


def test_fixed_windows_indices_are_sequential():
    chunks = fixed_windows(90.0, window=30.0, overlap=2.0)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_fixed_windows_short_clip_single_chunk():
    chunks = fixed_windows(10.0, window=30.0, overlap=1.0)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0
    assert chunks[0].end == pytest.approx(10.0)


def test_fixed_windows_empty_for_zero_duration():
    assert fixed_windows(0.0) == []


def test_fixed_windows_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed_windows(100.0, window=30.0, overlap=30.0)
    with pytest.raises(ValueError):
        fixed_windows(100.0, window=30.0, overlap=-1.0)


def test_total_covered_duration_counts_overlap_once():
    total = 60.0
    chunks = fixed_windows(total, window=30.0, overlap=10.0)
    # Even with heavy overlap the union equals the clip length.
    assert total_covered_duration(chunks) == pytest.approx(total)


def test_energy_vad_splits_on_silence():
    sr = 16000
    # Build: 1s tone, 0.6s silence, 1s tone.
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    silence = np.zeros(int(0.6 * sr), dtype=np.float32)
    waveform = np.concatenate([tone, silence, tone])

    chunks = energy_vad_windows(waveform, sr, min_silence=0.3, max_window=30.0)
    # The silent gap should produce two voiced segments.
    assert len(chunks) == 2
    assert chunks[0].index == 0 and chunks[1].index == 1
    # Second segment starts after the first ends (silence between them).
    assert chunks[1].start > chunks[0].end


def test_energy_vad_caps_long_segments():
    sr = 16000
    # 70s of continuous tone, no silence -> must be split by max_window.
    t = np.linspace(0, 70, 70 * sr, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    chunks = energy_vad_windows(
        waveform, sr, min_silence=0.3, max_window=30.0, overlap=1.0
    )
    # No chunk exceeds the max window length.
    assert all(c.duration <= 30.0 + 1e-6 for c in chunks)
    assert len(chunks) >= 3
