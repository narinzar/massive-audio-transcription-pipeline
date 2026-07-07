"""Checkpoint resume tests.

Simulates a crash by processing only some chunks, then verifies that a second
run skips the completed chunks and only processes the remaining ones.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checkpoint import (  # noqa: E402
    STATUS_DONE,
    STATUS_FAILED,
    CheckpointStore,
    ChunkResult,
)
from src.chunking import fixed_windows  # noqa: E402
from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402


def _fake_transcribe(waveform, sr, start, end, config):
    """Deterministic stand-in for the model: one segment per chunk."""
    return [{"start": 0.0, "end": end - start, "text": f"seg@{start:.1f}"}]


def _no_diarize(waveform, sr, num_speakers=2):
    return []


def test_store_save_and_load_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path)
    rec = ChunkResult(
        index=3,
        start=90.0,
        end=120.0,
        status=STATUS_DONE,
        segments=[{"start": 90.0, "end": 100.0, "text": "hi", "speaker": "SPEAKER_00"}],
        attempts=2,
    )
    store.save(rec)
    loaded = store.load(3)
    assert loaded is not None
    assert loaded.index == 3
    assert loaded.status == STATUS_DONE
    assert loaded.attempts == 2
    assert loaded.segments[0]["text"] == "hi"


def test_completed_indices_only_counts_done(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(ChunkResult(0, 0.0, 30.0, STATUS_DONE))
    store.save(ChunkResult(1, 30.0, 60.0, STATUS_FAILED))
    store.save(ChunkResult(2, 60.0, 90.0, STATUS_DONE))
    assert store.completed_indices() == {0, 2}
    assert store.pending([0, 1, 2, 3]) == [1, 3]


def test_corrupt_record_is_ignored(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save(ChunkResult(0, 0.0, 30.0, STATUS_DONE))
    # Write a garbage record file that must not crash load_all.
    (Path(tmp_path) / "chunk_1.json").write_text("{ not valid json", encoding="utf-8")
    assert store.completed_indices() == {0}
    assert store.load(1) is None


def _make_waveform(duration_s=100.0, sr=16000):
    n = int(duration_s * sr)
    return np.zeros(n, dtype=np.float32), sr


def test_resume_skips_completed_chunks(tmp_path):
    waveform, sr = _make_waveform(100.0)
    ckpt = tmp_path / "ckpt"
    chunks = fixed_windows(100.0, window=30.0, overlap=1.0)

    # Pre-seed the store as if chunks 0 and 1 finished before a crash.
    store = CheckpointStore(ckpt)
    for idx in (0, 1):
        c = chunks[idx]
        store.save(
            ChunkResult(
                index=c.index,
                start=c.start,
                end=c.end,
                status=STATUS_DONE,
                segments=[
                    {
                        "start": c.start,
                        "end": c.end,
                        "text": f"pre@{c.start:.1f}",
                        "speaker": "SPEAKER_00",
                    }
                ],
            )
        )

    # Track which chunks the transcribe function actually runs on this pass.
    ran_starts = []

    def tracking_transcribe(wf, s, start, end, config):
        ran_starts.append(round(start, 1))
        return [{"start": 0.0, "end": end - start, "text": f"new@{start:.1f}"}]

    cfg = PipelineConfig(
        window=30.0,
        overlap=1.0,
        workers=1,
        checkpoint_dir=str(ckpt),
    )
    result = run_pipeline(
        waveform,
        sr,
        cfg,
        transcribe_fn=tracking_transcribe,
        diarize_fn=_no_diarize,
    )

    # The two pre-seeded chunks must NOT be re-transcribed.
    preseeded_starts = {round(chunks[0].start, 1), round(chunks[1].start, 1)}
    assert preseeded_starts.isdisjoint(set(ran_starts))
    # Exactly the remaining chunks ran.
    assert len(ran_starts) == len(chunks) - 2
    assert result.resumed_chunks == 2
    assert result.completed_chunks == len(chunks)
    # Final transcript keeps the pre-seeded segments (resumed work preserved).
    texts = [s["text"] for s in result.segments]
    assert any(t.startswith("pre@") for t in texts)


def test_full_run_then_rerun_does_no_work(tmp_path):
    waveform, sr = _make_waveform(90.0)
    ckpt = tmp_path / "ckpt"
    cfg = PipelineConfig(window=30.0, overlap=1.0, workers=1, checkpoint_dir=str(ckpt))

    r1 = run_pipeline(
        waveform, sr, cfg, transcribe_fn=_fake_transcribe, diarize_fn=_no_diarize
    )
    assert r1.completed_chunks == r1.total_chunks
    assert r1.resumed_chunks == 0

    # Second run: everything is already done -> no transcribe calls at all.
    calls = {"n": 0}

    def counting_transcribe(wf, s, start, end, config):
        calls["n"] += 1
        return [{"start": 0.0, "end": end - start, "text": "x"}]

    r2 = run_pipeline(
        waveform, sr, cfg, transcribe_fn=counting_transcribe, diarize_fn=_no_diarize
    )
    assert calls["n"] == 0
    assert r2.resumed_chunks == r2.total_chunks
    assert r2.completed_chunks == r2.total_chunks


def test_failed_chunk_is_retried_on_next_run(tmp_path):
    waveform, sr = _make_waveform(60.0)
    ckpt = tmp_path / "ckpt"
    cfg = PipelineConfig(
        window=30.0,
        overlap=1.0,
        workers=1,
        max_attempts=2,
        base_delay=0.0,
        checkpoint_dir=str(ckpt),
    )

    # First pass: chunk starting at 0 always fails, the other succeeds.
    def failing_transcribe(wf, s, start, end, config):
        if start < 1.0:
            raise RuntimeError("boom")
        return [{"start": 0.0, "end": end - start, "text": "good"}]

    r1 = run_pipeline(
        waveform, sr, cfg, transcribe_fn=failing_transcribe, diarize_fn=_no_diarize
    )
    assert r1.failed_chunks == 1

    # Second pass: everything succeeds; the previously failed chunk must re-run.
    ran = []

    def healing_transcribe(wf, s, start, end, config):
        ran.append(round(start, 1))
        return [{"start": 0.0, "end": end - start, "text": "healed"}]

    r2 = run_pipeline(
        waveform, sr, cfg, transcribe_fn=healing_transcribe, diarize_fn=_no_diarize
    )
    # The failed chunk (start 0.0) is retried; the done one is skipped.
    assert 0.0 in ran
    assert r2.failed_chunks == 0
