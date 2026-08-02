from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
import sys
from pathlib import Path

from offline.assemble import assemble
from offline.config import OfflineSettings
from offline.stagepack import discover_packs
from offline.indexing import (
    FRAME_PAYLOAD_FIELDS,
    QdrantIndexer,
    build_local_index,
    frame_rows,
    scene_rows,
    scene_rows_remote,
)
from offline.providers import RemoteInferenceProvider
from offline.pipeline import OfflinePipeline
from datasection.exporter import sha256_file
from datasection.schemas import DatasetManifest, IndexArtifact


def _publish_indexes(manifest_path: Path, settings: OfflineSettings, output: Path, local_backend: str, dimension: int, qdrant: bool) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    portable = output.with_suffix(".json")
    try:
        location = portable.resolve().relative_to(settings.data_root).as_posix()
    except ValueError as exc:
        raise ValueError("index output must be below AIC_DATA_ROOT") from exc
    additions = [IndexArtifact(
        backend="faiss" if local_backend == "faiss" else "file", name="scenes_visual_v1",
        entity="scene", vector_name="visual", dimension=dimension, location=location,
        checksum=sha256_file(portable),
    )]
    if qdrant:
        collection = os.getenv("AIC_QDRANT_SCENE_COLLECTION", "aic_scenes_v1")
        additions.append(IndexArtifact(backend="qdrant", name=collection, entity="scene", vector_name="visual", dimension=dimension, location=f"qdrant://qdrant/{collection}"))
    keys = {(item.backend, item.name) for item in additions}
    indexes = [item for item in manifest.indexes if (item.backend, item.name) not in keys] + additions
    updated = manifest.model_copy(update={"indexes": indexes})
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(updated.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(manifest_path)


async def _main() -> None:
    parser = argparse.ArgumentParser(description="AIC V1 offline pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    assemble_cmd = sub.add_parser(
        "assemble", help="hợp nhất stage pack (notebook) -> canonical dataset"
    )
    assemble_cmd.add_argument("--packs", type=Path, default=Path("storage/packs"))
    assemble_cmd.add_argument("--out", type=Path, default=None)
    assemble_cmd.add_argument("--dataset-id", default="aic-video-v1")
    index = sub.add_parser("index")
    index.add_argument("--scenes", type=Path, default=Path("storage/exports/scenes.jsonl"))
    index.add_argument("--output", type=Path, default=Path("storage/indexes/scenes"))
    index.add_argument("--qdrant", action="store_true")
    index.add_argument("--encoder", choices=("local", "remote"), default=os.getenv("AIC_INDEX_ENCODER", "local"))
    index.add_argument(
        "--frames",
        action="store_true",
        help="build index mức frame (aic_frames_v2) thay vì mức scene",
    )
    index.add_argument("--embedding-name", default=os.getenv("AIC_FRAME_EMBEDDING_NAME"))
    args = parser.parse_args()
    if args.command == "run":
        run_settings = OfflineSettings.from_env()
        if run_settings.provider == "mock":
            print(
                "WARNING: AIC_OFFLINE_PROVIDER=mock — captions/OCR/objects are "
                "placeholders, not real model output. Set "
                "AIC_OFFLINE_PROVIDER=remote and AIC_GPU_URL for real data.",
                file=sys.stderr,
            )
        result = await OfflinePipeline(run_settings).run()
        print(result.model_dump_json(indent=2))
        return
    if args.command == "assemble":
        assemble_settings = OfflineSettings.from_env()
        if args.out is not None:
            assemble_settings = replace(assemble_settings, export_dir=args.out)
        packs = discover_packs(args.packs)
        manifest, report = await assemble(
            packs, settings=assemble_settings, dataset_id=args.dataset_id
        )
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        if report.quarantined:
            print(
                f"WARNING: {len(report.quarantined)} record bị cách ly -> "
                f"{assemble_settings.export_dir / 'quarantine.jsonl'}",
                file=sys.stderr,
            )
        print(f"manifest: {manifest.build_id} -> {assemble_settings.export_dir}")
        return
    settings = OfflineSettings.from_env()
    if args.frames:
        rows, degraded = frame_rows(
            args.scenes.parent, settings.data_root, embedding_name=args.embedding_name
        )
        if not rows:
            raise ValueError("cannot build an empty frame index")
        if degraded:
            print(
                "WARNING: một số frame chưa có embedding thật, đã dùng hashing text "
                "fallback — index này KHÔNG phải dense visual, đừng dùng để đo ablation.",
                file=sys.stderr,
            )
        local_backend = build_local_index(rows, args.output)
        print(f"local frame index: {local_backend} ({len(rows)} frames)")
        if args.qdrant:
            collection = os.getenv("AIC_QDRANT_FRAME_COLLECTION", "aic_frames_v2")
            client = QdrantIndexer(
                os.environ["AIC_QDRANT_URL"], collection, os.getenv("AIC_QDRANT_API_KEY"),
            )
            names = {str(row["vector_name"]): len(row["vector"]) for row in rows}
            await client.provision(vectors=names, payload_fields=FRAME_PAYLOAD_FIELDS)
            await client.upsert(rows)
            print(f"qdrant: {collection} ready ({', '.join(sorted(names))})")
        return
    if args.encoder == "remote":
        if not settings.gpu_url:
            raise ValueError("AIC_GPU_URL is required for --encoder remote")
        provider = RemoteInferenceProvider(settings.gpu_url, settings.gpu_api_key, settings.timeout_sec, settings.retries)
        rows = await scene_rows_remote(args.scenes, settings.data_root, provider)
    else:
        rows = scene_rows(args.scenes)
    if not rows:
        raise ValueError("cannot build an empty index")
    local_backend = build_local_index(rows, args.output)
    print(f"local index: {local_backend} ({len(rows)} rows)")
    if args.qdrant:
        client = QdrantIndexer(os.environ["AIC_QDRANT_URL"], os.getenv("AIC_QDRANT_SCENE_COLLECTION", "aic_scenes_v1"), os.getenv("AIC_QDRANT_API_KEY"))
        await client.provision(len(rows[0]["vector"]))
        await client.upsert(rows)
        print("qdrant: ready")
    _publish_indexes(args.scenes.parent / "dataset_manifest.json", settings, args.output, local_backend, len(rows[0]["vector"]), args.qdrant)
    print("manifest: index artifacts published")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
