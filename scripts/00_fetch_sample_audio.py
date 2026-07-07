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


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for name, url in SAMPLES:
        if fetch_one(name, url):
            ok += 1
    print(f"\nFetched {ok}/{len(SAMPLES)} samples into {DATA_DIR}")
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
