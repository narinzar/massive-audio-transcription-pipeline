"""Parallel audio transcription pipeline.

Modules:
    chunking    - split long audio into overlapping windows or VAD segments
    transcribe  - run Whisper inference on chunks
    diarize     - lightweight speaker segmentation and label merging
    checkpoint  - per-chunk on-disk checkpoint records for crash recovery
    retry       - exponential-backoff retry helper
    pipeline    - orchestration across a worker pool with real-time-factor stats
"""

__all__ = [
    "chunking",
    "transcribe",
    "diarize",
    "checkpoint",
    "retry",
    "pipeline",
]
