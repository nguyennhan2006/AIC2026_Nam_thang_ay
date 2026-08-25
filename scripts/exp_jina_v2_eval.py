"""Đánh giá jina v2 dense retrieval trên 25 câu P1.

    python scripts/exp_jina_v2_eval.py

Dùng jinaai/jina-clip-v2 (HuggingFace, CPU) để encode query,
rồi tìm kiếm trong FAISS index đã extract ở:
    storage/caption_embedding_jina_v2/
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYFRAMES_JSONL = ROOT / "storage/exports_competition/keyframes.jsonl"
JINA_V2_DIR = ROOT / "storage/caption_embedding_jina_v2"
# FAISS read_index doesn't support Unicode paths on Windows — use D:\aic2026_temp
JINA_V2_FAISS_ASCII = Path("D:/aic2026_temp/faiss.index")
P1_SUB_DIR = Path(r"C:\Users\ASUS\Downloads\AIC2026-SoTuyen1\submission")
# P1 query texts extracted from original PDFs, stored in scratchpad
P1_TEXT_DIR = Path(r"C:\Users\ASUS\AppData\Local\Temp\claude\d--Sinh-vi-n-CNhan-AIC-Data-AIC2026-Nam-thang-ay\3011bb7c-28fd-4234-a91c-c17dc2305e60\scratchpad\sotuyen1")


def load_gold() -> dict[str, tuple[str, int]]:
    gold: dict[str, tuple[str, int]] = {}
    for p in glob.glob(str(P1_SUB_DIR / "*.csv")):
        rows = list(csv.reader(open(p, encoding="utf8")))
        if not rows:
            continue
        m = re.search(r"query-(p1-\d+)-\w+\.csv", p)
        if not m:
            continue
        vid = rows[0][0].strip()
        frm = int(rows[0][1])
        gold[m.group(1)] = (vid, frm)
    return gold


def load_queries() -> dict[str, str]:
    queries: dict[str, str] = {}
    for p in sorted(P1_TEXT_DIR.glob("*.txt")):
        m = re.search(r"p1-(\d+)-(\w+)", p.name)
        if not m:
            continue
        qid = f"p1-{m.group(1)}"
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        # skip "Câu N:" / "Chủ đề:" / etc.
        text = " ".join(
            l for l in lines
            if not re.match(r"^[CQ]\d+[:\.]", l) and not re.match(r"^Chủ\s*đề", l)
        )
        queries[qid] = text
    return queries


def load_faiss_mappings():
    """Build row->(vid, frm) and (vid, frm)->row."""
    # caption_vector_mapping.jsonl: unique_caption_id -> row index
    # Use ASCII temp path for Windows FAISS compatibility
    ascii_dir = Path("D:/aic2026_temp")
    ucap_to_row: dict[str, int] = {}
    map_path = os.fspath(ascii_dir / "caption_vector_mapping.jsonl")
    with open(map_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            ucap_to_row[d["unique_caption_id"]] = i

    # caption_to_keyframes.jsonl: keyframe -> unique_caption_id
    row_to_kf: dict[int, tuple[str, int]] = {}
    kf_to_row: dict[tuple[str, int], int] = {}
    kf_path = os.fspath(ascii_dir / "caption_to_keyframes.jsonl")
    with open(kf_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            kf = (d["video_id"], d["frame_idx"])
            if kf in kf_to_row:
                continue
            uc_id = d["unique_caption_id"]
            row = ucap_to_row.get(uc_id)
            if row is not None:
                kf_to_row[kf] = row
                row_to_kf[row] = kf

    print(f"  row_to_kf: {len(row_to_kf):,} | kf_to_row: {len(kf_to_row):,}")
    return row_to_kf, kf_to_row


def load_jina_model():
    from transformers import AutoModel
    t0 = time.time()
    model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
    model.eval()
    print(f"  jina-clip-v2 loaded in {time.time()-t0:.1f}s")
    return model


def encode_queries(model, queries: dict[str, str], batch_size: int = 8) -> dict[str, np.ndarray]:
    """Encode all queries using jina-clip-v2 encode_text."""
    import torch
    texts = list(queries.values())
    qids = list(queries.keys())
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            result = model.encode_text(
                batch,
                task="retrieval.query",
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            # result is numpy array (batch, dim) when batch input
            for v in result:
                vectors.append(np.asarray(v, dtype=np.float32))
    return {qid: vectors[j] for j, qid in enumerate(qids)}


def recall_at_k(results: list[tuple[str, int, float]],
                gold_vid: str, gold_frm: int, k: int) -> tuple[bool, int]:
    """(found, best_rank) — found=True nếu có keyframe cùng video+frame trong top-k."""
    best = 999
    for rank, (vid, frm, _) in enumerate(results[:k], 1):
        if vid == gold_vid:
            best = min(best, rank)
            if frm == gold_frm:
                return True, rank
    return False, best


def recall_video_at_k(results: list[tuple[str, int, float]],
                      gold_vid: str, k: int) -> tuple[bool, int]:
    """(found, best_rank) — found=True nếu có keyframe cùng video trong top-k."""
    for rank, (vid, _, _) in enumerate(results[:k], 1):
        if vid == gold_vid:
            return True, rank
    return False, 999


def main():
    print("=" * 72)
    print("JINA V2 DENSE RETRIEVAL EVALUATION — 25 câu P1")
    print("=" * 72)

    gold = load_gold()
    queries = load_queries()
    print(f"\nGold: {len(gold)} | Queries loaded: {len(queries)}")
    for qid in sorted(queries.keys()):
        print(f"  {qid}: {queries[qid][:80]}")

    # Load FAISS
    print("\n[1] Loading FAISS index...")
    t0 = time.time()
    index = faiss.read_index(os.fspath(JINA_V2_FAISS_ASCII))
    print(f"    {index.ntotal:,} vectors, dim={index.d}")
    row_to_kf, kf_to_row = load_faiss_mappings()
    print(f"    loaded in {time.time()-t0:.1f}s")

    # Load jina model
    print("\n[2] Loading jina-clip-v2...")
    model = load_jina_model()

    # Encode queries
    print(f"\n[3] Encoding {len(queries)} queries...")
    t0 = time.time()
    q_vecs = encode_queries(model, queries)
    print(f"    done in {time.time()-t0:.1f}s")

    # Valid keyframes lookup
    valid_kf: dict[tuple[str, int], bool] = {}
    with KEYFRAMES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            valid_kf[(d["video_id"], d["frame_idx"])] = True

    # Search + evaluate
    print(f"\n[4] Searching & evaluating...")
    top_k = 50
    total = len(gold)

    at_k_frame = {1: 0, 5: 0, 10: 20, 30: 0, 50: 0}
    at_k_video = {1: 0, 5: 0, 10: 0, 30: 0, 50: 0}

    print(f"\n{'QID':8} {'gold_vid':12} {'gold_frm':9}  "
          f"{'valid':6} {'nearest':8}  "
          f"{'v2_rank_frm':12} {'v2_rank_vid':11}  "
          f"v2_score")
    print("-" * 90)

    by_vid: dict[str, list] = {}

    for qid, (gold_vid, gold_frm) in sorted(gold.items()):
        qtext = queries.get(qid, "")
        if qid not in q_vecs:
            print(f"  {qid}: NO QUERY VECTOR")
            continue

        q = q_vecs[qid].reshape(1, -1).astype(np.float32)
        D, I = index.search(q, top_k)

        results: list[tuple[str, int, float]] = []
        for d, i in zip(D[0], I[0]):
            if i < 0 or i not in row_to_kf:
                continue
            vid, frm = row_to_kf[i]
            results.append((vid, frm, float(d)))

        ok_frame, rank_frm = recall_at_k(results, gold_vid, gold_frm, top_k)
        ok_video, rank_vid = recall_video_at_k(results, gold_vid, top_k)

        valid = (gold_vid, gold_frm) in valid_kf
        if not valid:
            frames = [f for (v, f) in valid_kf if v == gold_vid]
            nearest = min(frames, key=lambda x: abs(x - gold_frm)) if frames else "?"
            dist = abs(gold_frm - nearest) if nearest != "?" else 0
        else:
            nearest = gold_frm
            dist = 0

        for k in at_k_frame:
            if ok_frame:
                at_k_frame[k] += 1
        for k in at_k_video:
            if ok_video:
                at_k_video[k] += 1

        rank_frm_str = f"#{rank_frm}" if rank_frm < 999 else "?"
        rank_vid_str = f"#{rank_vid}" if rank_vid < 999 else "?"
        score_str = f"{results[0][2]:.4f}" if results else "n/a"

        print(f"{qid:8} {gold_vid:12} {gold_frm:9}  "
              f"{str(valid):6} {f'+{dist}' if dist else str(nearest):8}  "
              f"{rank_frm_str:12} {rank_vid_str:11}  "
              f"{score_str}")

        by_vid.setdefault(gold_vid, []).append({
            "qid": qid, "valid": valid,
            "ok_frame": ok_frame, "ok_video": ok_video,
            "rank_frm": rank_frm, "rank_vid": rank_vid,
        })

    # Summary
    print(f"\n{'='*72}")
    print(f"RESULT — jina v2 dense (caption avg 41.9 words, jinaai/jina-clip-v2)")
    print(f"{'='*72}")
    print(f"\nRecall@K by EXACT FRAME (video+frame):")
    for k in [1, 5, 10, 20, 30, 50]:
        v = at_k_frame.get(k, 0)
        print(f"  @{k:3d}: {v:3d}/{total} = {v/total*100:5.1f}%")

    print(f"\nRecall@K by VIDEO (any frame in same video):")
    for k in [1, 5, 10, 20, 30, 50]:
        v = at_k_video.get(k, 0)
        print(f"  @{k:3d}: {v:3d}/{total} = {v/total*100:5.1f}%")

    print(f"\nPer-video breakdown:")
    for vid, infos in sorted(by_vid.items()):
        n = len(infos)
        ok1 = sum(1 for x in infos if x["ok_video"] and x["rank_vid"] <= 1)
        ok5 = sum(1 for x in infos if x["ok_video"] and x["rank_vid"] <= 5)
        ok10 = sum(1 for x in infos if x["ok_video"] and x["rank_vid"] <= 10)
        valid_n = sum(1 for x in infos if x["valid"])
        print(f"  {vid:12}  valid={valid_n}/{n}  vid@1={ok1}/{n} vid@5={ok5}/{n} vid@10={ok10}/{n}")

    # Top-10 queries by rank
    print(f"\nTop-10 video ranks:")
    ranked = [(info["rank_vid"], info["qid"], vid)
              for vid, infos in by_vid.items() for info in infos]
    for r, qid, vid in sorted(ranked)[:10]:
        print(f"  #{r:4d}: {qid} ({vid})")

    print(f"\n{'='*72}")
    print(f"Note: 6/25 frames are valid keyframes (others are arbitrary frame indices)")
    print(f"50 new videos (L26_V300-V349, 8,293 kf) NOT yet embedded in this index")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
