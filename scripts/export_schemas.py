"""Export portable JSON Schema contracts from the canonical Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = PROJECT_ROOT / "contracts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.keyframe import Keyframe  # noqa: E402


def export_keyframe_schema() -> Path:
    """Write the Keyframe JSON Schema deterministically and return its path."""

    CONTRACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CONTRACT_DIR / "keyframe.schema.json"
    output_path.write_text(
        json.dumps(Keyframe.model_json_schema(), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    print(export_keyframe_schema())
