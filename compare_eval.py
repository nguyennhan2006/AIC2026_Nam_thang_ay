"""So hai lần chạy `test_10_queries.py` — baseline vs tier2.

Hai lần chạy khác nhau ĐÚNG một biến (`AIC_ENABLE_LLM_QUERY_BUNDLE`), nên mọi
chênh lệch ở đây quy được về Tier 2.

    python compare_eval.py                  # baseline vs tier2
    python compare_eval.py a.json b.json    # hai file bất kỳ
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"thiếu {path} — chạy `python test_10_queries.py <label>` trước")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["id"]: r for r in data["results"] if "id" in r}


def rank_str(rank) -> str:
    return str(rank) if rank else "-"


def main() -> None:
    a_path = Path(sys.argv[1] if len(sys.argv) > 2 else "eval_10q_baseline.json")
    b_path = Path(sys.argv[2] if len(sys.argv) > 2 else "eval_10q_tier2.json")

    a, b = load(a_path), load(b_path)

    print("=" * 92)
    print(f"{a_path.stem}  ->  {b_path.stem}")
    print("(khác nhau đúng một biến: AIC_ENABLE_LLM_QUERY_BUNDLE)")
    print("=" * 92)
    print(f"{'query':<26} {'rank':>13}   {'frame':>11}   {'top video':<24}")
    print("-" * 92)

    better = worse = same = 0
    frame_gain = frame_loss = 0

    for qid in sorted(set(a) | set(b)):
        ra, rb = a.get(qid, {}), b.get(qid, {})
        ka, kb = ra.get("video_rank"), rb.get("video_rank")
        fa, fb = ra.get("frame_hit"), rb.get("frame_hit")

        # Xếp hạng: None (miss) coi như tệ nhất.
        va = ka if ka else 10**6
        vb = kb if kb else 10**6
        if vb < va:
            mark, _ = "BETTER", better
            better += 1
        elif vb > va:
            mark = "worse"
            worse += 1
        else:
            mark = ""
            same += 1

        if fb and not fa:
            frame_gain += 1
        elif fa and not fb:
            frame_loss += 1

        frame_col = f"{'HIT' if fa else '-'} -> {'HIT' if fb else '-'}"
        print(
            f"{qid:<26} {rank_str(ka):>5} -> {rank_str(kb):<5}   {frame_col:>11}   "
            f"{str(rb.get('top_video')):<24} {mark}"
        )

    def score(d: dict) -> tuple[int, int]:
        return (
            sum(1 for r in d.values() if r.get("video_rank")),
            sum(1 for r in d.values() if r.get("frame_hit")),
        )

    va_, fa_ = score(a)
    vb_, fb_ = score(b)
    total = len(set(a) | set(b))

    print("-" * 92)
    print(f"video recall@20 : {va_}/{total}  ->  {vb_}/{total}   ({vb_ - va_:+d})")
    print(f"frame hit@20    : {fa_}/{total}  ->  {fb_}/{total}   ({fb_ - fa_:+d})")
    print(f"thứ hạng        : {better} tốt hơn, {worse} tệ hơn, {same} không đổi")
    print(f"frame           : {frame_gain} thêm, {frame_loss} mất")


if __name__ == "__main__":
    main()
