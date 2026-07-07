# outputs/

Pipeline results land here. This directory is gitignored except for this
README.

After running `python scripts/01_transcribe.py` you will find:

- `transcript.json` - a list of segments, each
  `{"file", "start", "end", "text", "speaker"}`, ordered by time per file.
- `stats.json` - per-file and aggregate timing, including the real-time factor
  (audio seconds processed per wall-clock second, equivalently audio hours per
  wall-clock hour).

Checkpoints are written under `checkpoints/<filename>/` (also gitignored) so a
crashed or interrupted run resumes from the last completed chunk.
