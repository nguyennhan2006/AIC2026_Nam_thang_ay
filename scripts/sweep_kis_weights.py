"""Dò trọng số `KisConfig` trên bản dump của `dump_kis_features.py`.

Không gọi API, không chạy retrieval — đổi trọng số chỉ đổi thứ tự, nên xếp
hạng lại được offline. Một grid 72 cấu hình chạy dưới một giây, thay vì 52 giờ
nếu mỗi cấu hình một lượt `eval_tasks`.

**Kiểm chứng trước, kết luận sau.** Script tính lại thứ hạng ở đúng bộ trọng
số đang chạy rồi so với `online_rank` trong file dump. Không khớp 100% thì bản
dump thiếu thành phần nào đó và mọi con số phía dưới vô nghĩa — nên đó là thứ
đầu tiên nó in ra.

**Tách theo video là bắt buộc, không phải tùy chọn.** 36 truy vấn nghĩa là một
truy vấn = 0.028 R@1; một cấu hình "+1 truy vấn" là nhiễu. Tệ hơn, phần lớn
cấu hình thắng ở đây thắng TOÀN BỘ trên V001 và trả giá ở hai video kia — đo
được bằng `--holdout`. Đó chính là cách TRAKE từng khớp riêng V001
(`video_recall@1` 1.000 trên V001, 0.375 trên holdout).

    python -m scripts.sweep_kis_weights --features outputs/evaluation/kis_features.json
    python -m scripts.sweep_kis_weights --holdout
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

# Mặc định của `online.services.kis.KisConfig`.
BASE: dict[str, float] = {
    "retrieval": 1.0, "must": 0.6, "rare": 0.25, "nice": 0.1,
    "agreement": 0.15, "safe": 0.3, "contradiction": 0.8,
}


def gold_rank(query: dict, weights: dict[str, float]) -> int | None:
    scored = []
    for candidate in query["candidates"]:
        total = (
            weights["retrieval"] * candidate["retrieval"]
            + weights["must"] * candidate["must"]
            + weights["rare"] * candidate["rare"]
            + weights["nice"] * candidate["nice"]
            + weights["agreement"] * candidate["agreement"]
            + weights["safe"] * candidate["safe"]
        )
        if candidate["contradicted"]:
            total -= weights["contradiction"]
        scored.append((total, candidate["video_id"], candidate["frame_idx"], candidate["gold"]))
    # Cùng tie-break với `KisProcessor.rank`, nếu không thứ hạng sẽ lệch ở các
    # candidate bằng điểm và bản kiểm chứng bên dưới báo động giả.
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    return next((rank for rank, row in enumerate(scored, start=1) if row[3]), None)


def score(rows: list[dict], weights: dict[str, float]) -> tuple[int, float]:
    ranks = [gold_rank(query, weights) for query in rows]
    found = [rank for rank in ranks if rank]
    hits = sum(1 for rank in ranks if rank == 1)
    return hits, (sum(1.0 / rank for rank in found) / len(rows) if rows else 0.0)


def build_grid() -> list[dict[str, float]]:
    return [
        dict(BASE, must=must, rare=rare, safe=safe, agreement=agreement)
        for must, rare, safe, agreement in itertools.product(
            (0.2, 0.4, 0.6), (0.25, 0.6, 0.9, 1.3), (0.0, 0.15, 0.3), (0.0, 0.15)
        )
    ]


def _label(weights: dict[str, float]) -> str:
    return " ".join(
        f"{key}={weights[key]}" for key in ("must", "rare", "safe", "agreement")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="outputs/evaluation/kis_features.json")
    parser.add_argument("--holdout", action="store_true",
                        help="chọn trọng số trên một video, đo trên hai video còn lại")
    args = parser.parse_args()

    data = json.loads(Path(args.features).read_text(encoding="utf-8"))
    videos = sorted({item["query_id"][:4] for item in data})
    by_video = {v: [q for q in data if q["query_id"].startswith(v)] for v in videos}

    mismatched = [
        item["query_id"] for item in data
        if item.get("online_rank") != gold_rank(item, BASE)
    ]
    if mismatched:
        print(f"DỪNG: {len(mismatched)}/{len(data)} truy vấn lệch so với thứ hạng online "
              f"({', '.join(mismatched[:5])}…).")
        print("Bản dump thiếu thành phần điểm nào đó — mọi con số phía dưới sẽ sai.")
        return 1
    print(f"{len(data)} truy vấn, {sum(len(q['candidates']) for q in data)} candidate — "
          f"khớp 100% thứ hạng online\n")

    base_hits, base_mrr = score(data, BASE)
    print(f"{'BASE (đang chạy)':38s} R@1 {base_hits}/{len(data)}  MRR {base_mrr:.3f}")

    grid = build_grid()
    if args.holdout:
        print("\nchọn trên  →  đo trên hai video kia")
        for fit in videos:
            train, test = by_video[fit], [q for v in videos if v != fit for q in by_video[v]]
            picked = max(grid, key=lambda w: score(train, w))
            print(f"  {fit}  →  {score(test, BASE)[0]:2d}/{len(test)} → "
                  f"{score(test, picked)[0]:2d}/{len(test)}"
                  f"  ({score(test, picked)[0] - score(test, BASE)[0]:+d})"
                  f"   [{_label(picked)}]")

        print("\ncấu hình không làm tụt video nào:")
        survivors = []
        for weights in grid:
            deltas = [score(by_video[v], weights)[0] - score(by_video[v], BASE)[0] for v in videos]
            if all(d >= 0 for d in deltas) and sum(deltas) > 0:
                survivors.append((sum(deltas), deltas, weights))
        if not survivors:
            print("  KHÔNG CÓ — mọi cấu hình tăng điểm đều đánh đổi ở ít nhất một video.")
        for total, deltas, weights in sorted(survivors, key=lambda x: -x[0])[:8]:
            print(f"  +{total}  {dict(zip(videos, deltas))}   [{_label(weights)}]")
        return 0

    ranked = sorted(
        ((score(data, w), w) for w in grid),
        key=lambda item: (-item[0][0], -item[0][1]),
    )
    print("\n== 10 cấu hình tốt nhất ==")
    for (hits, mrr), weights in ranked[:10]:
        per = "  ".join(f"{v}:{score(by_video[v], weights)[0]}/{len(by_video[v])}" for v in videos)
        print(f"{_label(weights):38s} R@1 {hits}/{len(data)}  MRR {mrr:.3f}   {per}")
    print("\nChạy lại với --holdout trước khi tin bất kỳ dòng nào ở trên.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
