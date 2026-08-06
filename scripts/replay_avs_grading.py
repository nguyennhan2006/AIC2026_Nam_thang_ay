"""AVS-CRITERIA-01 — thử 5 cách chấm tiêu chí trên cùng một bộ candidate.

Chỉ thay `AvsCriteria.grade`; cổng grade, gom cụm, MMR và cap vẫn là code thật
của `AvsProcessor.rank`. Nhờ vậy chênh lệch giữa các biến thể quy được về đúng
một nguyên nhân.

    A  chuỗi con (hiện tại)
    B  độ phủ token, không trọng số
    C  độ phủ token có trọng số IDF
    D  C + thưởng cụm nguyên văn + thưởng khoảng cách gần
    E  D nhưng chấm THEO TỪNG TRƯỜNG rồi lấy max (caption / object-action / OCR / ASR)

Hai quy tắc gộp KHÁC NHAU và không được lẫn:

- Giữa các *option* trong cùng một nhóm (tách bởi "hoặc"/"hay") và giữa các
  *trường* của cùng một scene: dùng `max`. Chúng là những cách nói khác nhau
  của cùng một ý.
- Giữa các *nhóm* (tách bởi dấu phẩy/"và"): dùng trung bình. Truy vấn đòi cả
  "người cứu hộ" lẫn "đưa nạn nhân lên xe" thì khớp một nửa không phải khớp.

Vì sao dùng IDF: độ phủ token thô cho `người`, `hoạt động`, `đang`, `cảnh` cùng
trọng số với `thợ lặn` hay `rùa biển`, và bộ tiêu chí đầy những từ như thế.

Chỉ có 24 truy vấn AVS, nên KHÔNG dò ngưỡng ở vòng này — điểm giữ ở dạng liên
tục, ánh xạ 0–3 giữ nguyên ngưỡng cũ. Chọn ngưỡng chỉ sau khi nhìn phân phối
positive/hard-negative.

    python -m scripts.replay_avs_grading --candidates outputs/evaluation/avs_candidates.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.evidence import EvidencePack
from online.services import avs as avs_module
from online.services.avs import AvsConfig, AvsCriteria, AvsProcessor, extract_criteria

# ---------------------------------------------------------------- IDF corpus


class _Idf:
    """IDF trên chính văn bản scene mà cổng grade đang đọc."""

    def __init__(self, documents: list[str]) -> None:
        from collections import Counter

        self.total = max(len(documents), 1)
        counts: Counter[str] = Counter()
        for text in documents:
            counts.update(set(normalize_vi(text).split()))
        self.counts = counts

    def weight(self, token: str) -> float:
        return math.log((self.total + 1) / (self.counts.get(token, 0) + 1)) + 1.0


# ------------------------------------------------------------------ chấm điểm

_WORD = re.compile(r"[0-9a-zà-ỹ]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(normalize_vi(text))


def _coverage(option: str, words: set[str], idf: _Idf | None) -> float:
    toks = _tokens(option)
    if not toks:
        return 0.0
    if idf is None:
        return sum(1 for t in toks if t in words) / len(toks)
    total = sum(idf.weight(t) for t in toks)
    hit = sum(idf.weight(t) for t in toks if t in words)
    return hit / total if total else 0.0


def _proximity_bonus(option: str, text_tokens: list[str]) -> float:
    """Thưởng khi các token của tiêu chí nằm gần nhau trong văn bản.

    `lực lượng cứu hộ` rải rác khắp một caption dài là bằng chứng yếu hơn hẳn
    so với chúng đứng liền nhau, mà độ phủ thuần không phân biệt được.
    """

    toks = [t for t in _tokens(option)]
    if len(toks) < 2:
        return 0.0
    positions = [i for i, t in enumerate(text_tokens) if t in set(toks)]
    if len(positions) < 2:
        return 0.0
    span = positions[-1] - positions[0] + 1
    return len(toks) / span if span else 0.0


def make_grader(variant: str, idf: _Idf, fields_of):
    """Trả một hàm thay thế `AvsCriteria.grade`."""

    def score_text(option: str, text: str, use_idf: bool, bonus: bool) -> float:
        words = set(_tokens(text))
        value = _coverage(option, words, idf if use_idf else None)
        if bonus:
            if normalize_vi(option) in normalize_vi(text):
                value = min(1.0, value + 0.15)
            value = min(1.0, value + 0.10 * _proximity_bonus(option, _tokens(text)))
        return value

    def grade(self: AvsCriteria, text: str) -> int:
        if not self.inclusion:
            return 0
        words = set(_tokens(text))
        if any(_matches_exclusion(item, text, words) for item in self.exclusion):
            return 0

        texts = fields_of(text) if variant == "E" else [text]
        group_scores = []
        for group in self.inclusion:
            best = 0.0
            for option in group:
                # max giữa các option, và (biến thể E) giữa các trường.
                for field_text in texts:
                    best = max(best, score_text(
                        option, field_text,
                        use_idf=variant in {"C", "D", "E"},
                        bonus=variant in {"D", "E"},
                    ))
            group_scores.append(best)
        ratio = sum(group_scores) / len(group_scores)   # trung bình GIỮA CÁC NHÓM
        if ratio >= 0.999:
            return 3
        if ratio >= 0.6:
            return 2
        if ratio > 0.0:
            return 1
        return 0

    return grade


def _matches_exclusion(term: str, text: str, words: set[str]) -> bool:
    t = normalize_vi(term)
    if not t:
        return False
    return (t in normalize_vi(text)) if " " in t else (t in words)


# ------------------------------------------------------------------ chỉ số


def _dcg(values: list[int], k: int) -> float:
    return sum((2 ** v - 1) / math.log2(i + 1) for i, v in enumerate(values[:k], start=1))


def evaluate(data: list[dict], variant: str, idf: _Idf, config: AvsConfig) -> dict:
    original = AvsCriteria.grade

    def fields_of(text: str) -> list[str]:
        # `rerank_text` ghép bằng nhãn "Caption:", "OCR:", "ASR:", "Frames:".
        parts = re.split(r"\b(?:Caption|OCR|ASR|Frames?|Objects?):", text)
        return [p for p in parts if p.strip()] or [text]

    if variant != "A":
        AvsCriteria.grade = make_grader(variant, idf, fields_of)
    try:
        processor = AvsProcessor(config)
        per_query, per_video = [], {}
        for item in data:
            packs = [EvidencePack.model_validate(p) for p in item["packs"]]
            diagnostics: dict = {}
            results = processor.rank(
                item["query"], packs,
                retrieval_scores=item["retrieval_scores"],
                limit=item["limit"], normalizers=None, diagnostics=diagnostics,
            )
            ivs = item["relevant_intervals"]

            def hit(row):
                if row.video_id != item["target_video"] or row.best_frame_idx is None:
                    return 0, None
                m = [(iv["relevance_grade"], iv["event_id"]) for iv in ivs
                     if iv["start_frame"] <= row.best_frame_idx <= iv["end_frame"]]
                return max(m, default=(0, None))

            grades, seen = [], set()
            for row in results:
                g, event = hit(row)
                grades.append(0 if (event and event in seen) else g)
                if event:
                    seen.add(event)
            ideal = sorted((iv["relevance_grade"] for iv in ivs), reverse=True)
            ndcg = _dcg(grades, 100) / _dcg(ideal, 100) if _dcg(ideal, 100) else 0.0
            precision = (sum(1 for g in grades if g > 0) / len(grades)) if grades else 0.0
            events = {iv["event_id"] for iv in ivs}

            dropped = diagnostics.get("dropped", [])
            gold_dropped = sum(
                1 for d in dropped
                if d["video_id"] == item["target_video"] and d["best_frame_idx"] is not None
                and any(iv["start_frame"] <= d["best_frame_idx"] <= iv["end_frame"] for iv in ivs)
            )
            row = {
                "query_id": item["query_id"], "video": item["target_video"],
                "ndcg": ndcg, "precision": precision,
                "event_coverage": (len(seen) / len(events)) if events else 0.0,
                "results": len(results), "zero": int(not results),
                "post_grade": diagnostics.get("post_grade_candidate_count", 0),
                "pre_grade": diagnostics.get("pre_grade_candidate_count", 0),
                "gold_dropped_by_gate": gold_dropped,
                # hard-negative: candidate được GIỮ nhưng không chạm gold nào
                "hard_negative_accepted": sum(1 for g in grades if g == 0),
            }
            per_query.append(row)
            per_video.setdefault(item["target_video"], []).append(row)
    finally:
        AvsCriteria.grade = original

    def mean(key, rows=None):
        rows = rows or per_query
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "variant": variant,
        "nDCG@100": mean("ndcg"), "P@100": mean("precision"),
        "event_coverage": mean("event_coverage"),
        "zero_result_rate": mean("zero"),
        "gold_dropped_by_gate": mean("gold_dropped_by_gate"),
        "hard_negative_accepted": mean("hard_negative_accepted"),
        "post_grade": mean("post_grade"),
        "per_video": {v: round(mean("ndcg", rows), 4) for v, rows in sorted(per_video.items())},
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="outputs/evaluation/avs_candidates.json")
    parser.add_argument("--metadata", default="storage/exports_multivideo/scenes.jsonl")
    parser.add_argument("--variants", default="A,B,C,D,E")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    data = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    documents = []
    for line in Path(args.metadata).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scene = json.loads(line)
        parts = [c.get("text", "") for c in (scene.get("captions") or [])]
        parts += [k.get("normalized_text") or k.get("text") or "" for k in (scene.get("keywords") or [])]
        documents.append(" ".join(p for p in parts if p))
    idf = _Idf(documents)
    config = AvsConfig(
        max_per_video=int(__import__("os").environ.get("AIC_AVS_MAX_RESULTS_PER_VIDEO", 3)),
        grade_mode=__import__("os").environ.get("AIC_AVS_GRADE_MODE", "hard_gate"),
    )

    print(f"{len(data)} truy vấn AVS · cap={config.max_per_video} · "
          f"grade_mode={config.grade_mode}\n")
    header = (f"{'':2s} {'nDCG@100':>9} {'P@100':>7} {'event_cov':>10} {'zero':>6} "
              f"{'gold bị gate loại':>18} {'hard-neg giữ':>13}")
    print(header)
    print("-" * len(header))
    results = []
    for variant in args.variants.split(","):
        r = evaluate(data, variant.strip(), idf, config)
        results.append(r)
        print(f"{r['variant']:2s} {r['nDCG@100']:9.3f} {r['P@100']:7.3f} "
              f"{r['event_coverage']:10.3f} {r['zero_result_rate']:6.3f} "
              f"{r['gold_dropped_by_gate']:18.2f} {r['hard_negative_accepted']:13.1f}")

    print("\nnDCG theo video (ổn định hay chỉ thắng ở một video):")
    for r in results:
        print(f"  {r['variant']}  " + "  ".join(f"{v[-4:]}={s:.3f}" for v, s in r["per_video"].items()))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nđã ghi {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
