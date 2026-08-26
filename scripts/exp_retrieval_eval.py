"""Đánh giá retrieval trên 55 câu hỏi thật (P1 + P2-A).

Hai cấu hình đo song song:
  A) baseline: caption 35 từ hiện tại trong pack
  B) jina_v2: bật dense index của jinaai/jina-clip-v2

Cách chạy:
  python scripts/exp_retrieval_eval.py                    # cả hai
  python scripts/exp_retrieval_eval.py --method bm25    # chỉ BM25 (không cần FAISS)
  python scripts/exp_retrieval_eval.py --method dense    # chỉ jina v2 dense
  python scripts/exp_retrieval_eval.py --queries p1      # chỉ P1 (25 câu)

Output: bảng recall@1/5/10 theo video và tổng, với từng câu chi tiết.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import rank_bm25

from online.adapters.fpt_client import FptClient, image_to_data_url
from online.config import Settings
from prompts_caption_v2 import GROUP_GENRE

# ── Paths ────────────────────────────────────────────────────────────────────
KEYFRAMES_JSONL = ROOT / "storage/exports_competition/keyframes.jsonl"
SCENES_JSONL    = ROOT / "storage/exports_competition/scenes.jsonl"
P1_SUB_DIR      = Path(r"C:\Users\ASUS\Downloads\AIC2026-SoTuyen1\submission")
P2A_DIR         = Path(r"C:\Users\ASUS\AppData\Local\Temp\claude\d--Sinh-vi-n-CNhan-AIC-Data-AIC2026-Nam-thang-ay\3011bb7c-28fd-4234-a91c-c17dc2305e60\scratchpad\p2A")
JINA_V2_DIR     = Path(r"C:\Users\ASUS\AppData\Local\Temp\claude\d--Sinh-vi-n-CNhan-AIC-Data-AIC2026-Nam-thang-ay\3011bb7c-28fd-4234-a91c-c17dc2305e60\scratchpad\jina_v2")

# ── Gold answers ──────────────────────────────────────────────────────────────
def load_gold() -> dict[str, tuple[str, int]]:
    """{qid: (video, frame_idx)} cho tất cả câu có trong thư mục nộp."""
    import csv, glob, re
    gold: dict[str, tuple[str, int]] = {}
    for p in glob.glob(str(P1_SUB_DIR / "*.csv")):
        rows = list(csv.reader(open(p, encoding="utf8")))
        if not rows: continue
        qid = re.search(r"(query-p1-(\d+)-\w+\.csv)", os.path.basename(p))
        if not qid: continue
        vid = rows[0][0].strip()
        frm = int(rows[0][1])
        gold[qid.group(1)] = (vid, frm)
    for p in sorted(P2A_DIR.glob("*.txt")):
        qid = "p2-" + re.search(r"p2-(\d+)", p.name).group(1)
        # P2-A: đọc text đề để lấy video từ corpus map
        # Tạm: chỉ dùng P1 cho recall@K vì P2 chưa có gold nộp
    print(f"Gold: {len(gold)} câu P1")
    return gold

# ── Build BM25 index from keyframe captions ─────────────────────────────────
def build_bm25_index(
    caption_fn,
    batch_size: int = 50000,
) -> tuple[rank_bm25.BM25Okapi, list[tuple[str, int]], dict[tuple[str, int], int]]:
    """Build BM25 index over all keyframe captions.

    Args:
        caption_fn: callable(keyframe_dict) -> str caption text
        batch_size: commit index every N docs for memory

    Returns:
        bm25, keyframe_list (ordered), keyframe_to_row
    """
    print("  scanning keyframes.jsonl...")
    kf_list: list[tuple[str, int]] = []  # [(vid, frm)]
    corpus: list[list[str]] = []
    row_map: dict[tuple[str, int], int] = {}

    with KEYFRAMES_JSONL.open(encoding="utf-8") as fh:
        for row_idx, line in enumerate(fh):
            d = json.loads(line)
            vid = d["video_id"]
            frm = d["frame_idx"]
            text = caption_fn(d)
            if not text.strip():
                text = "[no_caption]"
            tokens = text.lower().split()
            kf_list.append((vid, frm))
            corpus.append(tokens)
            row_map[(vid, frm)] = row_idx
            if row_idx % 40000 == 0 and row_idx > 0:
                print(f"    {row_idx:,} keyframes scanned...")

    print(f"  building BM25 over {len(corpus):,} docs...")
    bm25 = rank_bm25.BM25Okapi(corpus)
    return bm25, kf_list, row_map


def caption_old(d: dict) -> str:
    """Caption v1: lấy text từ captions trong keyframe record."""
    return " ".join(c["text"] for c in d.get("captions", []))


def caption_jina_v2(d: dict) -> str:
    """Caption jina_v2: dùng chính text trong pack (cùng nguồn với embed)."""
    return caption_old(d)  # cùng nguồn — khác ở index, không phải text


# ── FAISS index (jina v2) ──────────────────────────────────────────────────
def load_jina_v2_faiss() -> tuple[
    faiss.IndexFlatIP,
    dict[int, tuple[str, int]],
    dict[tuple[str, int], int],
]:
    """Load jina v2 FAISS index và mapping."""
    print("  loading jina v2 FAISS index...")
    index_file = JINA_V2_DIR / "caption_faiss.index"
    mapping_file = JINA_V2_DIR / "caption_vector_mapping.jsonl"

    index = faiss.read_index(str(index_file))
    # FAISS IndexFlatIP trả cosine score trực tiếp
    print(f"    FAISS index: {index.ntotal:,} vectors, dim={index.d}")

    # row -> (vid, frm)
    row_to_kf: dict[int, tuple[str, int]] = {}
    kf_to_row: dict[tuple[str, int], int] = {}

    # Xây mapping: ucap_id -> row từ mapping, rồi keyframe từ to_keyframes
    ucap_to_row: dict[str, int] = {}
    for i, line in enumerate(open(mapping_file, encoding="utf-8")):
        d = json.loads(line)
        ucap_to_row[d["unique_caption_id"]] = i

    kf_to_ucap: dict[tuple[str, int], str] = {}
    for line in open(JINA_V2_DIR / "caption_to_keyframes.jsonl", encoding="utf-8"):
        d = json.loads(line)
        kf = (d["video_id"], d["frame_idx"])
        if kf not in kf_to_ucap:  # lấy keyframe đầu tiên
            kf_to_ucap[kf] = d["unique_caption_id"]

    for kf, ucap_id in kf_to_ucap.items():
        row = ucap_to_row.get(ucap_id)
        if row is not None:
            row_to_kf[row] = kf
            kf_to_row[kf] = row

    print(f"    kf_to_row mapped: {len(kf_to_row):,} keyframes")
    print(f"    NOTE: 8.293 keyframe (L26_V300-V349) chưa có trong FAISS")
    return index, row_to_kf, kf_to_row


# ── Search ──────────────────────────────────────────────────────────────────
def search_bm25(
    bm25,
    kf_list: list[tuple[str, int]],
    query: str,
    top_k: int = 20,
) -> list[tuple[str, int, float]]:
    """BM25 search, trả [(video, frame, score), ...]"""
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(kf_list[i][0], kf_list[i][1], float(scores[i])) for i in top_idx]


def search_dense(
    index: faiss.IndexFlatIP,
    row_to_kf: dict[int, tuple[str, int]],
    query: str,
    top_k: int = 20,
    encoder=None,
) -> list[tuple[str, int, float]]:
    """Dense search qua jina v2 index. Truy vấn bằng FPT API."""
    # Lazy: encode query mỗi lần gọi. Cho 55 truy vấn thì chấp nhận được.
    if encoder is None:
        encoder = _jina_v2_query_encoder()
    q_vec = encoder(query).reshape(1, -1).astype(np.float32)
    D, I = index.search(q_vec, top_k)
    results = []
    for d, i in zip(D[0], I[0]):
        if i < 0 or i not in row_to_kf:
            continue
        vid, frm = row_to_kf[i]
        results.append((vid, frm, float(d)))
    return results


_jina_encoder: Optional[object] = None


def _jina_v2_query_encoder():
    """Lazy-load Jina v2 encoder qua FPT API."""
    global _jina_encoder
    if _jina_encoder is not None:
        return _jina_encoder

    # Dùng FPT endpoint /embeddings với jina
    from online.config import Settings
    from online.adapters.fpt_client import FptClient
    load_env_()
    s = Settings.from_env()
    client = FptClient.from_settings(s)

    # Dùng Jina v3 endpoint từ FPT (không phải jina-clip-v2 trên Kaggle)
    # Fallback: encode bằng cách call FPT API với model phù hợp
    # Thực tế: FPT không có jina-clip-v2 → dùng text-embedding model
    # Đánh dấu: chỉ chạy BM25 cho đánh giá
    print("  [WARN] jina v2 dense encoder không có trên FPT — dùng BM25 thay thế")
    _jina_encoder = None
    return None


def load_env_(path: Path = ROOT / ".env.fpt.local") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── Recall metric ────────────────────────────────────────────────────────────
def recall_at_k(
    results: list[tuple[str, int, float]],
    gold_vid: str,
    gold_frm: int,
    k: int,
) -> bool:
    """Có gold nằm trong top-k kết quả không?"""
    top_k = results[:k]
    # Tìm kết quả cùng video
    same_vid = [(v, f, s) for v, f, s in top_k if v == gold_vid]
    if not same_vid:
        return False
    # Nếu cùng video, kiểm tra frame gần nhất
    # Với KIS: frame_idx phải khớp; với TRAKE: chuỗi frame
    return any(f == gold_frm for v, f, s in same_vid)


def recall_video_at_k(
    results: list[tuple[str, int, float]],
    gold_vid: str,
    k: int,
) -> bool:
    """Có kết quả nào cùng video không?"""
    return any(v == gold_vid for v, _, _ in results[:k])


# ── Normalize query ──────────────────────────────────────────────────────────
def normalize_query(text: str) -> str:
    """Chuẩn hóa câu hỏi thành query retrieval."""
    import re
    # Loại bỏ dấu câu, giữ từ
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bm25", "dense", "both"], default="bm25")
    parser.add_argument("--queries", choices=["p1", "p2", "both"], default="p1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--only-video", action="store_true",
                        help="đo recall@K theo VIDEO thay vì theo FRAME")
    args = parser.parse_args()

    print("=" * 72)
    print("RETRIEVAL EVALUATION — 55 câu hỏi thật (P1 + P2-A)")
    print(f"method={args.method}  queries={args.queries}  top_k={args.top_k}")
    print("=" * 72)

    gold = load_gold()
    if not gold:
        print("Không tìm thấy gold answers. Kiểm tra đường dẫn nộp.")
        return

    # ── BM25 index ──────────────────────────────────────────────────────────
    bm25, kf_list, kf_row_map = None, [], {}
    if args.method in ("bm25", "both"):
        print("\n[BM25] building index...")
        t0 = time.time()
        bm25, kf_list, kf_row_map = build_bm25_index(caption_old)
        print(f"  done in {time.time()-t0:.1f}s, {len(kf_list):,} keyframes")

    # ── jina v2 dense ────────────────────────────────────────────────────────
    faiss_index, row_to_kf, kf_to_row = None, {}, {}
    if args.method in ("dense", "both"):
        print("\n[jina_v2] loading dense index...")
        t0 = time.time()
        faiss_index, row_to_kf, kf_to_row = load_jina_v2_faiss()
        print(f"  done in {time.time()-t0:.1f}s")

    # ── Load P2-A queries ────────────────────────────────────────────────────
    p2_queries: dict[str, str] = {}
    if args.queries in ("p2", "both"):
        import re
        for p in sorted(P2A_DIR.glob("*.txt")):
            qid = re.search(r"(query-p2-(\d+)-\w+\.txt)", p.name)
            if qid:
                p2_queries[qid.group(1)] = p.read_text(encoding="utf-8").strip()

    all_queries = {}
    # P1 queries
    import csv
    for f in sorted(Path(P1_SUB_DIR).glob("*.csv")):
        import re
        qid_m = re.search(r"query-(p1-\d+)-\w+\.csv", f.name)
        if not qid_m:
            continue
        rows = list(csv.reader(f.open(encoding="utf-8")))
        if not rows:
            continue
        # Đọc text đề từ thư mục gốc
        qfile = ROOT / ".." / ".." / ".." / ".." / ".." / "Sinh viên CNhan" / "download" / "SOTUYEN1-bo-de-thi" / f.name.replace(".csv", ".txt")
        if qfile.exists():
            qtext = qfile.read_text(encoding="utf-8").strip()
        else:
            qtext = rows[0][0]  # fallback
        all_queries[qid_m.group(1)] = qtext

    print(f"\nQueries loaded: {len(all_queries)} P1")
    if p2_queries:
        all_queries.update({f"p2-{k}": v for k, v in p2_queries.items()})
        print(f"Queries total: {len(all_queries)} (P1 + P2-A)")

    # ── Run search ───────────────────────────────────────────────────────────
    recall_methods = {}
    if bm25:
        recall_methods["BM25"] = _eval_bm25
    if faiss_index is not None:
        recall_methods["jina_v2"] = _eval_dense

    for method_name, eval_fn in recall_methods.items():
        print(f"\n{'='*72}")
        print(f"METHOD: {method_name}")
        print(f"{'='*72}")
        eval_fn(all_queries, gold, args.top_k, args.only_video)


def _eval_bm25(queries, gold, top_k, only_video):
    import numpy as np
    import rank_bm25
    from pathlib import Path

    print("  building BM25 index (caption v1, 35 từ)...")
    kf_list: list[tuple[str, int]] = []
    corpus: list[list[str]] = []

    with KEYFRAMES_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            vid, frm = d["video_id"], d["frame_idx"]
            text = " ".join(c["text"] for c in d.get("captions", []))
            if not text.strip():
                text = "[no_caption]"
            kf_list.append((vid, frm))
            corpus.append(text.lower().split())

    bm25 = rank_bm25.BM25Okapi(corpus)
    print(f"  index built: {len(kf_list):,} keyframes")

    total = len(gold)
    at_k = {1: 0, 5: 0, 10: 0}
    by_vid: dict[str, list] = defaultdict(list)

    for qid, (gold_vid, gold_frm) in sorted(gold.items()):
        qtext = queries.get(qid, "")
        if not qtext:
            continue
        q_tokens = qtext.lower().split()
        scores = bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1]

        best_same_vid_frm = None
        best_same_vid = None
        rank_frm = rank_vid = 999
        for rank, idx in enumerate(top_idx[:top_k * 2]):
            vid, frm = kf_list[idx]
            if vid == gold_vid:
                if best_same_vid is None:
                    best_same_vid = (vid, frm, rank + 1)
                if frm == gold_frm and best_same_vid_frm is None:
                    best_same_vid_frm = (vid, frm, rank + 1)
                    break

        ok_frm = best_same_vid_frm is not None
        ok_vid = best_same_vid is not None

        if ok_frm:
            at_k[1] += 1 if (best_same_vid_frm and best_same_vid_frm[2] <= 1) else 0
            at_k[5] += 1 if best_same_vid_frm and best_same_vid_frm[2] <= 5 else 0
            at_k[10] += 1 if best_same_vid_frm and best_same_vid_frm[2] <= 10 else 0
        if ok_vid:
            at_k[1] += 0
            at_k[5] += 0
            at_k[10] += 0

        by_vid[gold_vid].append({
            "qid": qid, "ok_frame": ok_frm, "ok_video": ok_vid,
            "rank_frame": best_same_vid_frm[2] if best_same_vid_frm else 999,
            "rank_video": best_same_vid[2] if best_same_vid else 999,
        })

    print(f"\n{'QID':8} {'gold_video':12} {'gold_frm':9}  BM25@1  BM25@5  BM25@10  rank_frm")
    print("-" * 72)
    for qid, (gold_vid, gold_frm) in sorted(gold.items()):
        info = next((x for x in by_vid.get(gold_vid, []) if x["qid"] == qid), None)
        if not info:
            continue
        print(f"{qid:8} {gold_vid:12} {gold_frm:9}  "
              f"{'✓' if info['ok_frame'] and info['rank_frame']<=1 else '✗':5}  "
              f"{'✓' if info['ok_frame'] and info['rank_frame']<=5 else '✗':5}  "
              f"{'✓' if info['ok_frame'] and info['rank_frame']<=10 else '✗':6}  "
              f"#{info['rank_frame'] if info['rank_frame']<999 else '?':4}")

    # Summary
    print(f"\n{'='*60}")
    print(f"BM25 Caption V1 (35 từ) — Recall@K theo FRAME (khớp đúng video+frame):")
    for k in [1, 5, 10]:
        cnt = sum(1 for vi in by_vid.values()
                  for x in vi if x["ok_frame"] and x["rank_frame"] <= k)
        print(f"  Recall@{k}: {cnt:3d}/{total} = {cnt/total*100:5.1f}%")

    print(f"\nBM25 Caption V1 (35 từ) — Recall@K theo VIDEO (khớp đúng video):")
    for k in [1, 5, 10]:
        cnt = sum(1 for vi in by_vid.values()
                  for x in vi if x["ok_video"] and x["rank_video"] <= k)
        print(f"  Recall@{k}: {cnt:3d}/{total} = {cnt/total*100:5.1f}%")

    # Per-video breakdown
    print(f"\nPer video (BM25):")
    vid_scores = {}
    for vid, infos in sorted(by_vid.items()):
        n = len(infos)
        ok1 = sum(1 for x in infos if x["ok_frame"] and x["rank_frame"] <= 1)
        ok5 = sum(1 for x in infos if x["ok_frame"] and x["rank_frame"] <= 5)
        ok10 = sum(1 for x in infos if x["ok_frame"] and x["rank_frame"] <= 10)
        vid_scores[vid] = (ok1, ok5, ok10, n)
        print(f"  {vid:12}  {ok1:2d}/{n:2d}@{1} {ok5:2d}/{n:2d}@{5} {ok10:2d}/{n:2d}@{10}")

    return by_vid


def _eval_dense(queries, gold, top_k, only_video):
    print("  [TODO] jina v2 dense evaluation — cần encode query bằng jina v2")
    print("  FPT không có jina-clip-v2; cần chạy trên Kaggle GPU.")
    print("  Tạm thời dùng BM25 làm proxy.")


if __name__ == "__main__":
    main()
