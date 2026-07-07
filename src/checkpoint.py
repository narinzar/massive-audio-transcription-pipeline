"""Per-chunk checkpoint records for crash recovery.

Each chunk gets one JSON file under the checkpoint directory named
`chunk_<index>.json`. A record stores the chunk's status ("done" or "failed"),
the merged segment payload, and metadata. On restart the pipeline reads every
record, skips chunks already marked "done", and re-runs the rest. This turns a
mid-run crash into a resume from the last completed chunk rather than a restart.

Writes are atomic: the record is written to a temp file in the same directory
and then os.replace'd into place, so a crash during write cannot leave a
half-written record that later parses as valid.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_DONE = "done"
STATUS_FAILED = "failed"


@dataclass
class ChunkResult:
    """Result payload persisted for a single chunk."""

    index: int
    start: float
    end: float
    status: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    attempts: int = 1
    error: Optional[str] = None
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkResult":
        return cls(
            index=d["index"],
            start=d["start"],
            end=d["end"],
            status=d["status"],
            segments=d.get("segments", []),
            attempts=d.get("attempts", 1),
            error=d.get("error"),
            wall_seconds=d.get("wall_seconds", 0.0),
        )


class CheckpointStore:
    """Filesystem-backed store of ChunkResult records."""

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, index: int) -> Path:
        return self.directory / f"chunk_{index}.json"

    def save(self, result: ChunkResult) -> None:
        """Atomically write a chunk record to disk."""
        target = self._path(result.index)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.directory), prefix=".tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, target)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load(self, index: int) -> Optional[ChunkResult]:
        path = self._path(index)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return ChunkResult.from_dict(json.load(fh))
        except (json.JSONDecodeError, KeyError):
            # Corrupt or partial record: treat as absent so the chunk re-runs.
            return None

    def load_all(self) -> Dict[int, ChunkResult]:
        """Return every readable record keyed by chunk index."""
        results: Dict[int, ChunkResult] = {}
        for path in sorted(self.directory.glob("chunk_*.json")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    rec = ChunkResult.from_dict(json.load(fh))
                results[rec.index] = rec
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return results

    def completed_indices(self) -> set[int]:
        """Indices whose record exists and is marked done."""
        return {
            idx
            for idx, rec in self.load_all().items()
            if rec.status == STATUS_DONE
        }

    def is_done(self, index: int) -> bool:
        rec = self.load(index)
        return rec is not None and rec.status == STATUS_DONE

    def pending(self, indices: List[int]) -> List[int]:
        """Filter `indices` down to those not yet completed."""
        done = self.completed_indices()
        return [i for i in indices if i not in done]

    def clear(self) -> None:
        for path in self.directory.glob("chunk_*.json"):
            path.unlink()
