"""Orchestrate the transcription pipeline over a worker pool.

Flow per chunk:
    slice window -> (retry-wrapped) transcribe -> diarize-merge -> checkpoint.

Completed chunks are skipped on restart via the CheckpointStore, so a crash
mid-run resumes from the last completed chunk instead of restarting. The final
merge orders segments by start time and the pipeline reports the real-time
factor = audio_seconds / wall_seconds (a value above 1 means faster than
realtime).

Parallelism: chunks are independent once time ranges are fixed, so they fan out
across a process pool. Each worker loads the model once (module-level cache in
transcribe.py). For CPU-bound faster-whisper this gives real throughput gains;
set workers=1 to run in-process (used by tests).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from . import diarize as diarize_mod
from .checkpoint import STATUS_DONE, STATUS_FAILED, CheckpointStore, ChunkResult
from .chunking import Chunk, fixed_windows
from .retry import retry_call
from .transcribe import TranscribeConfig, transcribe_chunk


@dataclass
class PipelineConfig:
    window: float = 30.0
    overlap: float = 1.0
    num_speakers: int = 2
    workers: int = 1
    max_attempts: int = 4
    base_delay: float = 0.5
    factor: float = 2.0
    max_delay: float = 30.0
    checkpoint_dir: str = "checkpoints"
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)


@dataclass
class PipelineResult:
    segments: List[Dict[str, Any]]
    audio_seconds: float
    wall_seconds: float
    real_time_factor: float
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    resumed_chunks: int


def _process_chunk(
    chunk: Chunk,
    waveform: np.ndarray,
    sample_rate: int,
    config: PipelineConfig,
    transcribe_fn: Callable[..., List[Dict[str, Any]]],
    diarize_fn: Callable[..., List[Any]],
) -> ChunkResult:
    """Transcribe + diarize-merge a single chunk, with retry around transcribe.

    Returns a ChunkResult (status done or failed). Raising is avoided so the
    pool always gets a record to checkpoint.
    """
    t0 = time.perf_counter()
    attempts_box = {"n": 0}

    def _do_transcribe() -> List[Dict[str, Any]]:
        attempts_box["n"] += 1
        return transcribe_fn(
            waveform, sample_rate, chunk.start, chunk.end, config.transcribe
        )

    try:
        segments = retry_call(
            _do_transcribe,
            max_attempts=config.max_attempts,
            base_delay=config.base_delay,
            factor=config.factor,
            max_delay=config.max_delay,
        )
        a = max(0, int(round(chunk.start * sample_rate)))
        b = min(len(waveform), int(round(chunk.end * sample_rate)))
        turns = diarize_fn(
            waveform[a:b], sample_rate, num_speakers=config.num_speakers
        )
        # Shift turn times back into the clip timeline before merging.
        shifted = [
            diarize_mod.SpeakerTurn(
                start=t.start + chunk.start,
                end=t.end + chunk.start,
                speaker=t.speaker,
            )
            for t in turns
        ]
        merged = diarize_mod.merge_speaker_labels(segments, shifted)
        return ChunkResult(
            index=chunk.index,
            start=chunk.start,
            end=chunk.end,
            status=STATUS_DONE,
            segments=merged,
            attempts=attempts_box["n"],
            wall_seconds=time.perf_counter() - t0,
        )
    except Exception as exc:  # record failure instead of crashing the pool
        return ChunkResult(
            index=chunk.index,
            start=chunk.start,
            end=chunk.end,
            status=STATUS_FAILED,
            segments=[],
            attempts=attempts_box["n"],
            error=repr(exc),
            wall_seconds=time.perf_counter() - t0,
        )


def _dedup_overlap(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop near-duplicate segments produced by window overlap.

    Segments are sorted by start; a segment is dropped if it starts before the
    previous kept segment ended and carries identical text (the overlap zone
    transcribed twice).
    """
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    kept: List[Dict[str, Any]] = []
    for seg in ordered:
        if kept:
            prev = kept[-1]
            same_text = seg["text"].strip() == prev["text"].strip()
            overlaps = seg["start"] < prev["end"] - 1e-6
            if same_text and overlaps:
                continue
        kept.append(seg)
    return kept


def run_pipeline(
    waveform: np.ndarray,
    sample_rate: int,
    config: Optional[PipelineConfig] = None,
    chunks: Optional[List[Chunk]] = None,
    transcribe_fn: Callable[..., List[Dict[str, Any]]] = transcribe_chunk,
    diarize_fn: Callable[..., List[Any]] = diarize_mod.diarize_waveform,
) -> PipelineResult:
    """Run the full pipeline over a loaded mono waveform.

    waveform:      1-D float array of samples (mono).
    sample_rate:   samples per second.
    chunks:        optional precomputed chunk list; defaults to fixed_windows.
    transcribe_fn: injected for tests (defaults to real transcribe_chunk).
    diarize_fn:    injected for tests (defaults to real diarizer).

    Skips chunks already marked done in the checkpoint store, so a re-run after
    a crash resumes instead of restarting.
    """
    config = config or PipelineConfig()
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    audio_seconds = len(waveform) / sample_rate

    if chunks is None:
        chunks = fixed_windows(
            audio_seconds, window=config.window, overlap=config.overlap
        )

    store = CheckpointStore(config.checkpoint_dir)
    done_before = store.completed_indices()
    resumed = sum(1 for c in chunks if c.index in done_before)
    pending = [c for c in chunks if c.index not in done_before]

    wall_start = time.perf_counter()

    def _run_one(chunk: Chunk) -> ChunkResult:
        result = _process_chunk(
            chunk, waveform, sample_rate, config, transcribe_fn, diarize_fn
        )
        store.save(result)
        return result

    if config.workers <= 1:
        iterator = tqdm(pending, desc="chunks", unit="chunk")
        for chunk in iterator:
            _run_one(chunk)
    else:
        # Note: the worker function and its args must be picklable. transcribe_fn
        # and diarize_fn default to module-level functions, which pickle fine.
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = {
                pool.submit(
                    _process_chunk,
                    chunk,
                    waveform,
                    sample_rate,
                    config,
                    transcribe_fn,
                    diarize_fn,
                ): chunk
                for chunk in pending
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="chunks",
                unit="chunk",
            ):
                result = fut.result()
                store.save(result)

    wall_seconds = time.perf_counter() - wall_start

    # Assemble final transcript from every done record (including resumed ones).
    all_records = store.load_all()
    done_records = [
        r for r in all_records.values() if r.status == STATUS_DONE
    ]
    failed = [r for r in all_records.values() if r.status == STATUS_FAILED]

    segments: List[Dict[str, Any]] = []
    for rec in sorted(done_records, key=lambda r: r.index):
        segments.extend(rec.segments)
    segments = _dedup_overlap(segments)

    rtf = audio_seconds / wall_seconds if wall_seconds > 0 else float("inf")

    return PipelineResult(
        segments=segments,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        real_time_factor=rtf,
        total_chunks=len(chunks),
        completed_chunks=len(done_records),
        failed_chunks=len(failed),
        resumed_chunks=resumed,
    )
