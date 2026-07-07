# data/

Audio inputs for the pipeline live here. This directory is gitignored except
for this README.

## How to populate

Run the fetch script to download small public-domain speech clips:

```
python scripts/00_fetch_sample_audio.py
```

Or drop your own audio files here manually. Supported extensions:
`.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`. Files are downmixed to mono and
resampled to 16 kHz automatically at load time.

The transcription script (`scripts/01_transcribe.py`) processes every audio
file it finds in this directory.
