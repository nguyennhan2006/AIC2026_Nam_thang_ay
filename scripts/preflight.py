"""Fail-fast deployment checks before starting Online."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from datasection.exporter import verify_export
from online.adapters.vector_stores import QdrantVectorStore
from online.config import Settings


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=Path("storage/exports"))
    args = parser.parse_args()
    manifest = verify_export(args.export_dir)
    settings = Settings.from_env()
    if settings.metadata_jsonl.resolve() != (args.export_dir / "scenes.jsonl").resolve():
        raise ValueError("AIC_METADATA_JSONL does not point at the verified export")
    if settings.backend == "qdrant":
        matches = [item for item in manifest.indexes if item.backend == "qdrant" and item.name == settings.qdrant_scene_collection and item.vector_name == settings.qdrant_vector_name]
        if not matches:
            raise ValueError("manifest does not publish the configured Qdrant index")
        store = QdrantVectorStore(settings.qdrant_url or "", settings.qdrant_scene_collection, settings.qdrant_vector_name, api_key=settings.qdrant_api_key, timeout_sec=settings.request_timeout_sec)
        if not await store.health():
            raise RuntimeError("Qdrant collection is not ready")
    print(json.dumps({"status":"ready", "backend":settings.backend, "build_id":manifest.build_id, "videos":manifest.video_count, "scenes":manifest.scene_count, "keyframes":manifest.keyframe_count}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
