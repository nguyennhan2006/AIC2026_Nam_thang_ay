"""Atomic checkpoints; a failed run can resume without corrupting completed work."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


class JobLedger:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def read(self, video_id: str) -> dict:
        path = self.directory / f"{video_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def write(self, video_id: str, stage: str, status: str, **details) -> None:
        target = self.directory / f"{video_id}.json"
        current = self.read(video_id)
        current.update({
            "video_id": video_id,
            "stage": stage,
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        })
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(current, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
