"""Bonus/penalty re-scoring sau weighted RRF (Phương án E).

Sau khi ``weighted_rrf`` trộn các nhánh, bước này cộng/trừ một lượng điểm
nhỏ dựa trên rule tường minh, dễ giải thích — rẻ hơn nhiều so với reranker
LLM/VLM và nhắm thẳng vào top-1/top-5:

    + ocr_exact_bonus    khi OCR của scene chứa nguyên cụm ngoặc kép
                         (so trên văn bản đã bỏ dấu),
    + ocr_partial_bonus  khi chứa >= 60% số từ của cụm đó,
    + object_bonus       cho mỗi object label trong query xuất hiện ở scene
                         (có trần ``object_bonus_cap``),
    + keyword_bonus      cho mỗi keyword của scene xuất hiện trong query,
    - must_match_penalty khi query có cụm ngoặc kép mà scene KHÔNG chứa
                         từ nào của cụm — tín hiệu "thiếu must-match".

THANG ĐIỂM QUAN TRỌNG: điểm RRF rất nhỏ (một nhánh weight 1.0 đóng góp
1/(60+rank) ≈ 0.016 ở rank 1). Bonus mặc định ở đây được chọn cùng thang
(0.003–0.02). Nếu chỉnh ``rrf_k`` hoặc modality weights thì chỉnh lại
``RuleConfig`` tương ứng. Mọi giá trị mặc định là điểm khởi đầu — chỉ giữ
rule nếu nó tăng Recall@K/MRR trên dev set (đo bằng ``scripts/eval_kis.py``,
so ``--use-rules`` với không dùng).

Cách sử dụng
------------
Áp dụng lên list ``Candidate`` đã fuse, TRƯỚC khi hydrate:

    from online.services.rules import RuleConfig, apply_bonus_penalty

    candidates = weighted_rrf(lists, weights)          # như hiện tại
    documents = {d.scene_id: d
                 for d in await repository.get_many(
                     [c.scene_id for c in candidates])}
    candidates = apply_bonus_penalty(
        candidates,
        documents,
        exact_phrases=plan.events[0].exact_phrases,    # cụm ngoặc kép
        query=plan.normalized_query,
        config=RuleConfig(),                           # hoặc tune riêng
    )

Hoặc để eval nhanh: ``python -m scripts.eval_kis --use-rules``.

Mỗi candidate được điều chỉnh sẽ có ``payload["rule_adjustments"]`` liệt kê
từng rule đã cộng/trừ bao nhiêu — dùng để debug tại sao một kết quả leo
lên/tụt xuống.
"""

from __future__ import annotations

from dataclasses import dataclass

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.models import Candidate, SceneDocument


@dataclass(frozen=True, slots=True)
class RuleConfig:
    """Trọng số rule, cùng thang với điểm RRF (~0.016/nhánh ở rank 1)."""

    ocr_exact_bonus: float = 0.020
    ocr_partial_bonus: float = 0.010
    ocr_partial_threshold: float = 0.6  # >= 60% số từ của cụm xuất hiện
    object_bonus: float = 0.004        # mỗi object label match
    object_bonus_cap: float = 0.012    # trần tổng object bonus
    keyword_bonus: float = 0.003       # mỗi keyword của scene có trong query
    keyword_bonus_cap: float = 0.009
    must_match_penalty: float = 0.015  # có ngoặc kép nhưng scene không dính từ nào


def _phrase_hit_level(phrase_norm: str, ocr_norm: str, threshold: float) -> str:
    """'exact' | 'partial' | 'miss' cho một cụm ngoặc kép trên OCR của scene."""

    if not phrase_norm:
        return "miss"
    if phrase_norm in ocr_norm:
        return "exact"
    words = phrase_norm.split()
    if not words:
        return "miss"
    ocr_words = set(ocr_norm.split())
    coverage = sum(1 for word in words if word in ocr_words) / len(words)
    if coverage >= threshold:
        return "partial"
    return "hit" if coverage > 0 else "miss"


def apply_bonus_penalty(
    candidates: list[Candidate],
    documents: dict[str, SceneDocument],
    *,
    exact_phrases: list[str],
    query: str,
    config: RuleConfig | None = None,
) -> list[Candidate]:
    """Trả về list candidate mới, đã cộng bonus/trừ penalty và sort lại.

    Không mutate input. Candidate thiếu document (chưa hydrate được) giữ
    nguyên điểm. So khớp chạy trên văn bản đã ``normalize_vi`` (bỏ dấu)
    để không nhạy với lỗi dấu của OCR.
    """

    config = config or RuleConfig()
    phrases_norm = [normalize_vi(item) for item in exact_phrases if item.strip()]
    query_norm = normalize_vi(query)
    query_words = set(query_norm.split())

    output: list[Candidate] = []
    for candidate in candidates:
        document = documents.get(candidate.scene_id)
        if document is None:
            output.append(candidate)
            continue

        adjustments: dict[str, float] = {}
        ocr_norm = normalize_vi(" ".join(document.ocr_texts))

        # --- OCR bonus / must-match penalty theo cụm ngoặc kép ---
        if phrases_norm:
            levels = [
                _phrase_hit_level(item, ocr_norm, config.ocr_partial_threshold)
                for item in phrases_norm
            ]
            if "exact" in levels:
                adjustments["ocr_exact"] = config.ocr_exact_bonus
            elif "partial" in levels:
                adjustments["ocr_partial"] = config.ocr_partial_bonus
            elif all(level == "miss" for level in levels):
                adjustments["must_match_miss"] = -config.must_match_penalty

        # --- Object bonus: label (đa số là tiếng Anh) xuất hiện trong query ---
        if document.object_labels and query_words:
            matched_labels = {
                label
                for label in document.object_labels
                if set(normalize_vi(label).split()) <= query_words
            }
            if matched_labels:
                bonus = min(
                    config.object_bonus * len(matched_labels),
                    config.object_bonus_cap,
                )
                adjustments["object"] = bonus

        # --- Keyword bonus: keyword của scene nằm gọn trong query ---
        if document.keywords and query_norm:
            matched_keywords = {
                keyword
                for keyword in document.keywords
                if normalize_vi(keyword) and normalize_vi(keyword) in query_norm
            }
            if matched_keywords:
                bonus = min(
                    config.keyword_bonus * len(matched_keywords),
                    config.keyword_bonus_cap,
                )
                adjustments["keyword"] = bonus

        if not adjustments:
            output.append(candidate)
            continue

        payload = dict(candidate.payload)
        payload["rule_adjustments"] = {
            name: round(value, 6) for name, value in adjustments.items()
        }
        output.append(
            candidate.model_copy(
                update={
                    "score": candidate.score + sum(adjustments.values()),
                    "payload": payload,
                }
            )
        )

    output.sort(key=lambda item: (-item.score, item.scene_id))
    return [
        candidate.model_copy(update={"rank": rank})
        for rank, candidate in enumerate(output, start=1)
    ]
