# massive-audio-transcription-pipeline

A parallel audio transcription pipeline: it chunks long audio into overlapping
windows, runs Whisper inference across a worker pool, merges lightweight speaker
diarization onto the transcript, and checkpoints every chunk so a crash resumes
from the last completed chunk instead of restarting.

## Problem

Transcribing many hours of audio with Whisper is slow and fragile. A single
long file does not fit the model's short input window, a serial run wastes idle
cores, and any crash (OOM, a transient CUDA error, a lost network mount) throws
away all completed work. This pipeline addresses all three: it splits audio into
model-sized windows with overlap for context continuity, fans the windows out
across processes for throughput, and persists a per-chunk checkpoint so an
interrupted job continues where it stopped. It also reports the real-time factor
(audio hours processed per wall-clock hour) so you can see whether you are
running faster than realtime.

## Approach

- Chunk long audio into fixed-length overlapping windows, or optionally split on
  low-energy (silence) gaps with an energy-based voice-activity detector, then
  cap any long voiced segment to the model window.
- Transcribe each chunk with faster-whisper (default model `base`, batched GPU
  inference via CTranslate2); openai-whisper is a selectable fallback backend.
- Merge speaker labels onto transcript segments by maximum time overlap. The
  built-in diarizer is a dependency-free energy/pause clustering stand-in and is
  documented as a drop-in swap for pyannote.audio (which needs `HF_TOKEN`).
- Checkpoint every chunk to a JSON record with an atomic write. On restart the
  pipeline skips chunks already marked done and re-runs only the rest.
- My addition: transcription is wrapped in exponential-backoff retry (capped
  attempts, growing delays). Combined with per-chunk checkpointing, a transient
  failure is absorbed in place, and a hard failure mid-run resumes from the last
  completed chunk on the next invocation.
- Fan chunks out across a process pool (`--workers N`); each worker loads the
  model once. The pipeline reports `real_time_factor = audio_seconds /
  wall_seconds`.

## Setup

```
# create and activate a virtual environment (Python 3.12)
uv venv --python 3.12 .venv          # or: python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# install torch from the CUDA 12.8 wheel index first (GPU, e.g. RTX 5090)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# then the rest
pip install -r requirements.txt

cp .env.example .env                 # no secrets required
```

CPU-only machines can skip the cu128 torch install; faster-whisper falls back to
`int8` compute on CPU automatically.

## How to run

```
# 1. fetch a couple of small public-domain speech clips into data/
python scripts/00_fetch_sample_audio.py

# 2. transcribe everything in data/ -> outputs/transcript.json + stats.json
python scripts/01_transcribe.py

# parallel run across 4 worker processes, larger model:
python scripts/01_transcribe.py --workers 4 --model small

# run the test suite (mocks the model; no download needed):
pytest -q
```

Key flags for `01_transcribe.py`: `--workers`, `--model`, `--backend`
(`faster-whisper` or `openai-whisper`), `--window`, `--overlap`, `--speakers`.

## Results

Numbers below are produced by running the commands above; this repo ships the
code, run it to populate them.

Reproduce and observe:

```
# baseline (serial) vs parallel throughput on the same data:
python scripts/01_transcribe.py --workers 1
python scripts/01_transcribe.py --workers 4
# compare real_time_factor in outputs/stats.json between the two runs.
```

| Metric | Where | Expected behavior |
| --- | --- | --- |
| real_time_factor | outputs/stats.json | TBD (run). Above 1 on GPU means faster than realtime. |
| workers 1 vs 4 | outputs/stats.json | TBD (run). RTF should rise with workers up to the core count, then plateau. |
| resume after crash | console + checkpoints/ | TBD (run). A re-run resumes; completed chunks are skipped. |

Expected qualitative behavior:

- Real-time factor above 1 on a GPU means the pipeline transcribes faster than
  realtime (more than one audio hour per wall-clock hour).
- Throughput should increase with `--workers` up to the number of physical CPU
  cores (or GPU saturation) and then plateau; past that, added workers contend
  for the same device.
- Checkpointing makes a re-run resume rather than restart. Interrupt a run with
  Ctrl-C partway through, then run the same command again: it skips completed
  chunks and processes only the remainder. `stats.json` reports `resumed_chunks`
  greater than 0 on the second run.
- Retry/backoff absorbs transient failures: a chunk that hits an intermittent
  error is retried with growing delays before it is recorded as failed. Failed
  chunks are re-attempted on the next run.

## What I'd do next at larger scale

Replace the stand-in diarizer with pyannote.audio for real speaker attribution,
and move checkpoints from per-chunk JSON files to a single append-only log or a
small database to cut filesystem overhead on jobs with millions of chunks. For
multi-node scale I would put chunk work on a task queue (Ray or Celery) with the
same checkpoint contract, so workers can be added or lost without losing
completed transcripts.
