"""Download a couple of small public-domain speech clips into data/.

Sources are public-domain / permissively licensed short speech recordings.
The URLs below point at small clips; if any URL is unreachable the script
reports it and continues, so a partial fetch still leaves usable data.

Run:
    python scripts/00_fetch_sample_audio.py

Files land in data/ and are gitignored (only data/README.md is tracked).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Small public-domain / CC0 speech samples. These are widely mirrored test
# clips. If you prefer LibriVox chapters or a Common Voice sample, drop any
# .wav / .mp3 / .flac file into data/ and the pipeline will pick it up.
SAMPLES = [
    (
        "harvard.wav",
        "https://www2.cs.uic.edu/~i101/SoundFiles/harvard.wav",
    ),
    (
        "gettysburg.wav",
        "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav",
    ),
]


def fetch_one(name: str, url: str) -> bool:
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name} already present")
        return True
    try:
        print(f"[get ] {name} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "audio-pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"[ok  ] {name} ({len(data)} bytes)")
        return True
    except Exception as exc:  # network issues should not abort the whole run
        print(f"[fail] {name}: {exc}", file=sys.stderr)
        return False


def fetch_from_hf() -> int:
    """Fallback: build a longer clip from the public-domain LibriSpeech dummy
    set on the Hugging Face Hub. Used when the direct URLs are unreachable
    (for example behind a restrictive network). Concatenates the clips into one
    multi-minute file so the chunking path is exercised.
    """
    try:
        import io

        import numpy as np
        import soundfile as sf
        from datasets import Audio, load_dataset
    except Exception as exc:
        print(f"[hf  ] fallback unavailable ({exc})", file=sys.stderr)
        return 0
    try:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(decode=False))
        parts, sr = [], 16000
        for row in ds:
            arr, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            parts.append(arr)
            parts.append(np.zeros(int(0.5 * sr), dtype="float32"))
        long = np.concatenate(parts)
        dest = DATA_DIR / "librispeech_long.wav"
        sf.write(dest, long, sr)
        print(f"[hf  ] wrote {dest.name} ({len(long) / sr:.1f}s of speech)")
        return 1
    except Exception as exc:
        print(f"[hf  ] fallback failed: {exc}", file=sys.stderr)
        return 0


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, url in SAMPLES:
        if fetch_one(name, url):
            ok += 1
    if ok == 0:
        print("Direct URLs unreachable; trying the Hugging Face LibriSpeech fallback...")
        ok += fetch_from_hf()
    print(f"\nFetched audio into {DATA_DIR} (sources succeeded: {ok})")
    if ok == 0:
        print(
            "No samples fetched. Place any .wav/.mp3/.flac file in data/ "
            "manually and re-run the transcription script.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
