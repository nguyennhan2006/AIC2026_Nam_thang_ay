"""Rebuild dense caption index từ export competition.

Vấn đề: Dense index cũ được build từ Kaggle artifacts (151,459 scenes)
nhưng export competition chỉ có 87,742 scenes. Điều này gây ra
scenes không tồn tại trong repository khi retrieval trả về kết quả.

Script này rebuild index từ captions trong export competition để đảm bảo
dense index và repository align với nhau.

Chạy:
    python -m scripts.rebuild_caption_dense_index \
        --metadata storage/exports_competition/scenes.jsonl \
        --model storage/models/jina-clip-v2 \
        --out storage/caption_embedding_jina_v2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from online.adapters.json_metadata import JsonlSceneRepository


def _extract_captions(scene: dict) -> str:
    """Extract caption text từ scene JSON (khác với SceneDocument.captions)."""
    parts = []

    # Keyframe captions
    for kf in scene.get("keyframes", []):
        for cap in kf.get("captions", []):
            if isinstance(cap, dict):
                text = cap.get("text", "")
            else:
                text = str(cap)
            if text:
                parts.append(text)

    # Tags/keywords
    for kf in scene.get("keyframes", []):
        for tag in kf.get("tags", []):
            if tag:
                parts.append(str(tag))

    # ASR segments
    for seg in scene.get("asr_segments", []):
        text = seg.get("text", "")
        if text:
            parts.append(text)

    return " | ".join(p.strip() for p in parts if p.strip())


async def main_async(args: argparse.Namespace) -> None:
    print(f"Loading metadata from {args.metadata}...")
    repository = await JsonlSceneRepository.load(args.metadata)
    scenes = await repository.all()

    print(f"Total scenes: {len(scenes)}")

    # Build scene_id -> text mapping
    scene_texts = []
    for scene in scenes:
        text = _extract_captions(scene.model_dump())
        if text.strip():
            scene_texts.append((scene.scene_id, text))

    print(f"Scenes with text: {len(scene_texts)}/{len(scenes)}")

    if not scene_texts:
        raise SystemExit("No scenes with caption text to index")

    # Import encoder
    from online.adapters.dense_text import JinaClipV2Encoder, build_text_encoder

    print(f"Building encoder ({args.encoder})...")
    encoder = build_text_encoder(
        args.encoder, str(args.model), device=args.device, for_passages=True
    )

    # Encode all documents
    print(f"Encoding {len(scene_texts)} documents...")
    scene_ids = [sid for sid, _ in scene_texts]
    texts = [text for _, text in scene_texts]

    # Encode in batches
    embeddings = []
    batch_size = args.batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_emb = encoder.encode(batch, batch_size)
        embeddings.append(batch_emb)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    matrix = np.concatenate(embeddings, axis=0)
    print(f"  Embedding matrix shape: {matrix.shape}")

    # Save
    args.out.mkdir(parents=True, exist_ok=True)

    np.save(args.out / "embeddings.npy", matrix.astype(np.float32))

    with open(args.out / "scene_ids.json", "w", encoding="utf-8") as f:
        json.dump(scene_ids, f, ensure_ascii=False)

    fingerprint = hashlib.sha256(matrix.tobytes()).hexdigest()[:16]
    manifest = {
        "schema_version": "aic2026-caption-embedding-v1",
        "status": "success",
        "metadata_source": str(args.metadata),
        "model_id": args.model_id or args.encoder,
        "encoder_kind": args.encoder,
        "query_prefix": "",
        "index_fingerprint": fingerprint,
        "dimension": int(matrix.shape[1]),
        "vector_count": len(scene_ids),
        "scene_count": len(scene_ids),
    }

    with open(args.out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {args.out}:")
    print(f"  - embeddings.npy: {matrix.shape}")
    print(f"  - scene_ids.json: {len(scene_ids)} scenes")
    print(f"  - manifest.json")
    print(f"\nManifest:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild caption dense index from competition export")
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_competition/scenes.jsonl"))
    parser.add_argument("--model", type=Path,
                        default=Path("storage/models/jina-clip-v2"))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--encoder", choices=["e5", "jina_v3", "jina_clip_v2"],
                        default="jina_clip_v2")
    parser.add_argument("--out", type=Path,
                        default=Path("storage/caption_embedding_jina_v2"))
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
