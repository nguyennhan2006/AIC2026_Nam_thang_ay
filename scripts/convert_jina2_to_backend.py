"""Chuyển jina v2 FAISS artifacts (Kaggle output) sang format CaptionDenseRetriever.

Chạy MỘT LẦN sau khi extract ZIP:
    python scripts/convert_jina2_to_backend.py

Tạo:
    storage/caption_embedding_jina_v2/embeddings.npy   (float32, (151459, 1024))
    storage/caption_embedding_jina_v2/scene_ids.json  (json list)
    storage/caption_embedding_jina_v2/manifest.json   (ghi đúng encoder_kind=jina_v3)

Lý do cần chuyển:
  - Kaggle sinh caption_faiss.index (FAISS IndexFlatIP), dùng được cho offline eval.
  - CaptionDenseRetriever online đọc embeddings.npy + scene_ids.json (numpy), KHÔNG
    đọc FAISS. Numpy matrix dùng chính vector từ Kaggle, chỉ đổi float16 -> float32.
  - Score trong CaptionDenseRetriever: numpy dot product thay vì FAISS.search.
    Hai cách cho KẾT QUẢ GIOỐNG NHAU vì cùng L2-normalized vector.

FAISS vẫn giữ nguyên trong thư mục — không xóa, chỉ bổ sung 3 file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# FAISS không cần import ở đây — chỉ dùng numpy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "storage" / "caption_embedding_jina_v2"


def load_vector_mapping():
    """Build: row_idx -> unique_caption_id và unique_caption_id -> representative (vid, frm)."""
    # caption_vector_mapping.jsonl: row -> unique_caption_id
    row_to_ucap: dict[int, str] = {}
    with open(SRC / "caption_vector_mapping.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            row_to_ucap[i] = d["unique_caption_id"]

    # caption_to_keyframes.jsonl: unique_caption_id -> first (vid, frm)
    # Nhiều keyframe cùng caption → lấy representative đầu tiên
    ucap_rep: dict[str, tuple[str, int]] = {}
    with open(SRC / "caption_to_keyframes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            kf = (d["video_id"], d["frame_idx"])
            if d["unique_caption_id"] not in ucap_rep:
                ucap_rep[d["unique_caption_id"]] = kf

    # Build scene_ids: deduplicate theo unique_caption_id
    # Row i = unique_caption_id[i] → (vid, frm) → scene_id = f"{vid}_S{frm//8:04d}"
    # Khớp với scene_id trong metadata export (scenes.jsonl).
    scene_ids: list[str] = []
    for i in range(len(row_to_ucap)):
        uc_id = row_to_ucap[i]
        vid, frm = ucap_rep.get(uc_id, ("UNKNOWN", 0))
        scene_ids.append(f"{vid}_S{frm//8:04d}")

    return scene_ids


def convert_embeddings():
    """Đọc caption_vectors.f16.npy → embeddings.npy (float32)."""
    src_path = SRC / "caption_vectors.f16.npy"
    dst_path = SRC / "embeddings.npy"

    print(f"  reading {src_path}...")
    mat_f16 = np.load(src_path)  # shape (151459, 1024), dtype float16
    print(f"    shape={mat_f16.shape}, dtype={mat_f16.dtype}")

    # Float16 → float32 (FAISS L2-normalize rồi, chỉ chuyển dtype)
    mat_f32 = mat_f16.astype(np.float32)
    print(f"    converted to float32")

    print(f"  writing {dst_path}...")
    np.save(dst_path, mat_f32)
    sz = dst_path.stat().st_size
    print(f"    saved: {sz / 1024 / 1024:.1f} MB")

    # Verify
    loaded = np.load(dst_path)
    print(f"    verified: shape={loaded.shape}, dtype={loaded.dtype}")


def write_scene_ids(scene_ids: list[str]):
    """Ghi scene_ids.json."""
    path = SRC / "scene_ids.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(scene_ids, fh, ensure_ascii=False)
    print(f"  scene_ids.json: {len(scene_ids):,} entries, {path.stat().st_size / 1024 / 1024:.1f} MB")


def write_manifest():
    """Ghi manifest.json đúng format CaptionDenseRetriever đọc."""
    path = SRC / "manifest.json"
    manifest = {
        "schema_version": "aic2026-caption-embedding-v1",
        "status": "success",
        "metadata_source": "jinaai/jina-clip-v2 Kaggle embedding (caption_embedding_jina_v2_artifacts)",
        "model_id": "jinaai/jina-clip-v2",
        "encoder_kind": "jina_v3",  # QUAN TRỌNG: phải khớp AIC_CAPTION_DENSE_ENCODER=jina_v3
        "query_prefix": "",           # jina dùng task= chứ không phải prefix
        "index_fingerprint": "jina_v2_faiss_artifacts_20250825",
        "dimension": 1024,
        "vector_count": 151459,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"  manifest.json: encoder_kind=jina_v3")


def verify_coverage():
    """Kiểm tra scene_ids khớp với metadata export."""
    # Load scene_ids từ index
    with open(SRC / "scene_ids.json", encoding="utf-8") as fh:
        index_ids = set(json.load(fh))

    # Load scene_ids từ metadata
    metadata_ids: set[str] = set()
    scenes_path = ROOT / "storage" / "exports_competition" / "scenes.jsonl"
    if scenes_path.exists():
        with open(scenes_path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                metadata_ids.add(d.get("scene_id", ""))
        covered = index_ids & metadata_ids
        print(f"\n  Coverage check:")
        print(f"    index keyframes: {len(index_ids):,}")
        print(f"    metadata scenes: {len(metadata_ids):,}")
        print(f"    overlap:         {len(covered):,}")
        if metadata_ids:
            pct = len(covered) / len(metadata_ids) * 100
            print(f"    coverage:        {pct:.1f}%")
            if pct < 98:
                missing = sorted(metadata_ids - index_ids)[:5]
                print(f"    WARNING: coverage < 98%! Missing sample: {missing}")
        else:
            print(f"    (scenes.jsonl not found, skip coverage check)")
    else:
        print(f"\n  (scenes.jsonl not at {scenes_path}, skip coverage check)")


def main():
    print("=" * 60)
    print("Convert jina v2 FAISS artifacts → backend format")
    print("=" * 60)

    for fname in ["caption_vectors.f16.npy", "caption_vector_mapping.jsonl",
                  "caption_to_keyframes.jsonl"]:
        p = SRC / fname
        if not p.exists():
            raise SystemExit(f"Missing: {p} — extract ZIP first")

    print("\n[1] Converting embeddings (float16 → float32)...")
    convert_embeddings()

    print("\n[2] Building scene_ids mapping...")
    scene_ids = load_vector_mapping()
    write_scene_ids(scene_ids)

    print("\n[3] Writing manifest...")
    write_manifest()

    print("\n[4] Coverage check...")
    verify_coverage()

    print(f"\n{'='*60}")
    print("Done. Backend format ready.")
    print("Bật jina v2 bằng cách thêm vào .env.fpt.local:")
    print("  AIC_CAPTION_DENSE_INDEX=storage/caption_embedding_jina_v2")
    print("  AIC_CAPTION_DENSE_ENCODER=jina_v3")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
