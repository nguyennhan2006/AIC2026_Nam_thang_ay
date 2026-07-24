from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import EngineConfig
from .ingest import load_components
from .records import BuildReport
from .storage import create_database
from .vector_index import build_vector_index


def build_index(
    input_root: str | Path,
    index_dir: str | Path,
    config: EngineConfig | None = None,
) -> BuildReport:
    """Build a complete local lexical/vector/metadata index."""

    started = time.perf_counter()
    config = config or EngineConfig()
    config.validate()
    input_root = Path(input_root).expanduser().resolve()
    index_dir = Path(index_dir).expanduser().resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = index_dir.parent / "_aic_search_input_cache"

    loaded = load_components(
        input_root, staging_dir, needs_review_penalty=config.needs_review_penalty
    )
    scene_vector_manifest = build_vector_index(
        index_dir, loaded.scene_embeddings, config, name="scene"
    )
    frame_vector_manifest = None
    if loaded.keyframe_embeddings is not None and loaded.keyframes:
        frame_vector_manifest = build_vector_index(
            index_dir, loaded.keyframe_embeddings, config, name="frame"
        )
    meta = {
        "schema_version": 2,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_embedding_model": loaded.scene_embedding_model,
        "keyframe_embedding_model": loaded.keyframe_embedding_model,
        "embedding_dimension": loaded.embedding_dimension,
        "source_root": str(loaded.source_root),
        "stats": loaded.stats,
        "warnings": loaded.warnings,
        "config": config.to_dict(),
    }
    database_path = index_dir / "aic_search.db"
    create_database(database_path, loaded.scenes, loaded.keyframes, meta)

    manifest = {
        **meta,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "files": {
            "database": database_path.name,
        },
        "scene_vector_index": scene_vector_manifest,
        "frame_vector_index": frame_vector_manifest,
    }
    manifest_path = index_dir / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return BuildReport(
        index_dir=str(index_dir),
        database_path=str(database_path),
        vector_backend=(
            scene_vector_manifest["backend"]
            + (f"+{frame_vector_manifest['backend']}" if frame_vector_manifest else "")
        ),
        embedding_dimension=loaded.embedding_dimension,
        scene_count=len(loaded.scenes),
        keyframe_count=len(loaded.keyframes),
        video_count=len({scene.video_id for scene in loaded.scenes}),
        warnings=loaded.warnings,
        elapsed_sec=round(time.perf_counter() - started, 4),
    )
