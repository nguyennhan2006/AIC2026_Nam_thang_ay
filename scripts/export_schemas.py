from __future__ import annotations

import json
from pathlib import Path

from datasection.schemas import ClipSegment, DatasetManifest, Event, Keyframe, Scene, Video


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "contracts"
    root.mkdir(parents=True, exist_ok=True)
    for name, model in (
        ("keyframe", Keyframe), ("scene", Scene), ("clip", ClipSegment), ("event", Event),
        ("video", Video), ("dataset_manifest", DatasetManifest),
    ):
        (root / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
