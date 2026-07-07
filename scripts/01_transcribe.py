"""Run the transcription pipeline over data/ and write outputs.

For each audio file in data/, the script loads the waveform (resampled to
16 kHz mono via soundfile + a light resampler), runs the pipeline, and appends
its segments to a combined transcript. Per-file checkpoints live under
checkpoints/<filename>/ so a crash or Ctrl-C mid-run resumes on the next call.

Outputs:
    outputs/transcript.json - list of {file, start, end, speaker, text}
    outputs/stats.json      - per-file and aggregate real-time factor

Run:
    python scripts/01_transcribe.py
    python scripts/01_transcribe.py --workers 4 --model base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import PipelineConfig, run_pipeline  # noqa: E402
from src.transcribe import TranscribeConfig  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"
CKPT_DIR = REPO_ROOT / "checkpoints"
TARGET_SR = 16000
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _resample_linear(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resample to the target rate (adequate for Whisper input)."""
    if src_sr == dst_sr:
        return samples
    duration = len(samples) / src_sr
    n_dst = int(round(duration * dst_sr))
    if n_dst <= 1:
        return samples
    src_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    dst_times = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    return np.interp(dst_times, src_times, samples).astype(np.float32)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    data = _resample_linear(data, sr, TARGET_SR)
    return data, TARGET_SR


def find_audio_files() -> list[Path]:
    return sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS
    )


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Transcribe data/ audio files")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--window", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=1.0)
    parser.add_argument("--speakers", type=int, default=2)
    parser.add_argument(
        "--backend",
        type=str,
        default="faster-whisper",
        choices=["faster-whisper", "openai-whisper"],
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_audio_files()
    if not files:
        print(
            f"No audio files in {DATA_DIR}. Run "
            "scripts/00_fetch_sample_audio.py first.",
            file=sys.stderr,
        )
        return 1

    all_segments = []
    per_file_stats = []
    total_audio = 0.0
    total_wall = 0.0

    for path in files:
        print(f"\n=== {path.name} ===")
        waveform, sr = load_audio(path)
        cfg = PipelineConfig(
            window=args.window,
            overlap=args.overlap,
            num_speakers=args.speakers,
            workers=args.workers,
            checkpoint_dir=str(CKPT_DIR / path.stem),
            transcribe=TranscribeConfig(
                backend=args.backend, model_name=args.model
            ),
        )
        result = run_pipeline(waveform, sr, cfg)

        for seg in result.segments:
            all_segments.append({"file": path.name, **seg})

        per_file_stats.append(
            {
                "file": path.name,
                "audio_seconds": round(result.audio_seconds, 3),
                "wall_seconds": round(result.wall_seconds, 3),
                "real_time_factor": round(result.real_time_factor, 3),
                "total_chunks": result.total_chunks,
                "completed_chunks": result.completed_chunks,
                "failed_chunks": result.failed_chunks,
                "resumed_chunks": result.resumed_chunks,
            }
        )
        total_audio += result.audio_seconds
        total_wall += result.wall_seconds
        print(
            f"    audio={result.audio_seconds:.1f}s "
            f"wall={result.wall_seconds:.1f}s "
            f"RTF={result.real_time_factor:.2f} "
            f"(resumed {result.resumed_chunks}, failed {result.failed_chunks})"
        )

    aggregate_rtf = total_audio / total_wall if total_wall > 0 else float("inf")
    stats = {
        "aggregate": {
            "audio_seconds": round(total_audio, 3),
            "wall_seconds": round(total_wall, 3),
            "real_time_factor": round(aggregate_rtf, 3),
            "audio_hours_per_wall_hour": round(aggregate_rtf, 3),
            "files": len(files),
        },
        "per_file": per_file_stats,
    }

    (OUT_DIR / "transcript.json").write_text(
        json.dumps(all_segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nWrote {len(all_segments)} segments to outputs/transcript.json"
        f"\nAggregate RTF = {aggregate_rtf:.2f} (audio hours per wall hour)"
        f"\nStats in outputs/stats.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
