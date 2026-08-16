"""Fusion methods across incomparable retrieval scores — Search Mixing Console W5.

Có HAI họ method, khác nhau ở chỗ căn bản: cái gì được dùng làm đóng góp.

**Họ suy từ RANK** — `rrf`, `weighted_sum`, `max_score`, `intersection`, `union`.
Tất cả dùng chung `weight / (rrf_k + rank)`, không bao giờ đụng tới `raw_score`
giữa các branch (BM25 vs cosine vs color-overlap-ratio không so sánh được).
`weighted_sum`/`max_score` chỉ khác `rrf` ở cách GỘP (sum/max thay vì tổng kiểu
điều hoà) — chúng **không phải** weighted sum có chuẩn hoá điểm.

**Họ ĐỌC ĐIỂM THẬT** (Phase D, docs/31) — `norm_sum`, `norm_max`, `margin_sum`,
`entropy_sum`. Chuẩn hoá min-max trong phạm vi TỪNG branch rồi mới gộp.

Cơ chế thật là **ĐẬP ĐUÔI**, không phải "branch chắc chắn thắng branch đoán mò".
Tôi ban đầu giải thích theo hướng thứ hai và `tests/test_fusion_score_methods.py`
bác bỏ nó: ba branch cùng xếp một candidate hạng 1 thì `norm_max` vẫn chọn phía
đồng thuận, đúng như nó nên làm.

Con số nói rõ hơn::

    ti le dong gop hang-1 so voi hang-100 CUA CUNG MOT BRANCH
      RRF (k=60)     1/61 so voi 1/160   ->  2.62x
      min-max        1.00 so voi ~0.00   ->  vo han

RRF cho candidate hạng 100 tận **38%** số phiếu của hạng 1. Bảy branch × 100
candidate = 700 lá phiếu gần bằng nhau, và tín hiệu thật chìm trong đó. Chuẩn
hoá làm đuôi về ~0, nên chỉ phần đỉnh của mỗi branch còn bỏ phiếu.

`margin_sum`/`entropy_sum` đi xa hơn — nhân thêm hệ số đo độ tự tin của branch ở
lần truy vấn này. **Đo được là CẢ HAI đều kém hơn** bản chỉ chuẩn hoá (docs/31
§2.6): chuẩn hoá là thứ có ích, nhân thêm hệ số tự tin thì hại.

`intersection`/`union` khác ở chỗ candidate nào sống sót: `intersection` chỉ giữ
candidate được ít nhất `minimum_matching_branches` branch nhìn thấy.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Literal

from online.domain.models import Candidate, Modality
from online.domain.search_config import BranchRuntimeOptions

FusionMethod = Literal[
    "rrf", "weighted_sum", "max_score", "intersection", "union",
    # Phase D của docs/31 — nhóm ĐỌC ĐIỂM THẬT, không suy từ rank.
    "norm_sum", "norm_max", "margin_sum", "entropy_sum",
]

# Nhóm method dùng điểm đã chuẩn hoá thay cho `weight / (rrf_k + rank)`.
_SCORE_METHODS = frozenset({"norm_sum", "norm_max", "margin_sum", "entropy_sum"})


def _minmax(values: list[float]) -> list[float]:
    """Chuẩn hoá về [0, 1] TRONG PHẠM VI một branch.

    Min-max chứ không z-score: điểm giữa các branch không cùng thang (BM25 ~27,
    cosine ~0.8, overlap-ratio ~0.5) và cũng không cùng phân bố, nên thứ duy
    nhất so sánh được là "candidate này mạnh đến đâu SO VỚI phần còn lại mà
    chính branch đó trả về". Min-max cho đúng nghĩa đó và bị chặn, nên một
    branch có đuôi dài không tự thổi phồng mình.

    Mọi điểm bằng nhau -> trả 1.0: branch đó không phân biệt được gì, nhưng nó
    vẫn bỏ phiếu "tất cả đều liên quan". Trả 0.0 sẽ là xoá phiếu của nó.
    """

    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [1.0] * len(values)
    span = high - low
    return [(value - low) / span for value in values]


def _confidence(values: list[float]) -> float:
    """Độ chắc chắn của một branch = mức độ phân bố điểm của nó bị dồn về đỉnh.

    `1 - H/H_max` với `H` là entropy của phân bố điểm đã chuẩn hoá thành xác
    suất. Branch chỉ ra đúng một candidate -> H thấp -> confidence cao. Branch
    trả 100 candidate điểm ngang nhau -> H = H_max -> confidence 0.

    Chỉ dùng bởi `entropy_sum`. **Đo được là làm KÉM đi** so với bản chỉ chuẩn
    hoá (KIS R@1 0.611 so với 0.750) — giữ lại để không ai thử lại ý này mà
    không biết đã đo. Xem docs/31 §2.6.
    """

    total = sum(values)
    if total <= 0 or len(values) < 2:
        return 1.0
    entropy = 0.0
    for value in values:
        share = value / total
        if share > 0:
            entropy -= share * math.log(share)
    ceiling = math.log(len(values))
    if ceiling <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - entropy / ceiling))


def _margin(values: list[float]) -> float:
    """Khoảng cách tương đối giữa hạng 1 và hạng 2 của một branch, trong [0, 1].

    Cùng mục đích với `_confidence` nhưng chỉ nhìn hai vị trí đầu. Chỉ dùng bởi
    `margin_sum`, và cũng **đo được là kém hơn** bản chỉ chuẩn hoá (KIS R@1
    0.583 so với 0.750). Giữ lại làm ghi chép thí nghiệm.
    """

    if len(values) < 2:
        return 1.0
    ordered = sorted(values, reverse=True)
    top = ordered[0]
    if top <= 0:
        return 0.0
    return max(0.0, min(1.0, (top - ordered[1]) / top))


def _branch_weight(
    candidate: Candidate, modality_weights: dict[Modality, float], branches: dict[str, BranchRuntimeOptions],
) -> float:
    # `candidate.source` là execution_id (`bm25_caption.expanded`). Tra theo
    # execution trước, rồi tới branch — cấu hình đặt ở mức branch vẫn áp cho
    # mọi biến thể query của nó.
    override = branches.get(candidate.source)
    if override is None and "." in candidate.source:
        override = branches.get(candidate.source.rsplit(".", 1)[0])
    if override is not None:
        return override.weight if override.enabled else 0.0
    return modality_weights.get(candidate.modality, 0.0)


def fuse_candidates(
    ranked_lists: list[list[Candidate]],
    modality_weights: dict[Modality, float],
    *,
    method: FusionMethod = "rrf",
    rrf_k: int = 60,
    limit: int = 100,
    branches: dict[str, BranchRuntimeOptions] | None = None,
    minimum_matching_branches: int = 1,
) -> list[Candidate]:
    """Fuse `ranked_lists`, retain component scores, return deterministic ordering."""

    branches = branches or {}
    totals: defaultdict[str, float] = defaultdict(float)
    # Khoá phụ cho các method lấy MAX. Không có nó thì mọi candidate đứng hạng 1
    # của bất kỳ nhánh nào đều được đúng `weight * 1.0` và HOÀ nhau, rồi thứ
    # hạng bị quyết bằng `scene_id` — tức bảng chữ cái. Đo trên 8 truy vấn KIS
    # thật: 3/8 có hoà ở đỉnh. Tổng (chứ không phải max) là thứ tự nhiên để phá
    # hoà: cùng một đỉnh thì candidate được NHIỀU nhánh ủng hộ nên đứng trước.
    sums: defaultdict[str, float] = defaultdict(float)
    representatives: dict[str, Candidate] = {}
    components: defaultdict[str, dict[str, float]] = defaultdict(dict)
    contributions: defaultdict[str, dict[str, float]] = defaultdict(dict)
    modalities: defaultdict[str, set[str]] = defaultdict(set)
    evidence: defaultdict[str, list[dict]] = defaultdict(list)
    matching_branches: defaultdict[str, set[str]] = defaultdict(set)
    frame_hints: dict[str, Candidate] = {}

    for candidates in ranked_lists:
        # Chuẩn hoá TRONG PHẠM VI từng ranked list (= từng branch), tính một lần
        # trước vòng lặp: mọi candidate của cùng branch phải dùng chung min/max,
        # nếu tính lẻ thì mỗi candidate lại có một thang riêng.
        normalized: list[float] = []
        branch_scale = 1.0
        if method in _SCORE_METHODS:
            raw = [candidate.raw_score for candidate in candidates]
            normalized = _minmax(raw)
            if method == "margin_sum":
                branch_scale = _margin(raw)
            elif method == "entropy_sum":
                branch_scale = _confidence(raw)

        for fallback_rank, candidate in enumerate(candidates, start=1):
            rank = candidate.rank or fallback_rank
            weight = _branch_weight(candidate, modality_weights, branches)
            if method in _SCORE_METHODS:
                contribution = weight * branch_scale * normalized[fallback_rank - 1]
            else:
                contribution = weight / (rrf_k + rank)
            key = candidate.grouping_key
            if method in ("max_score", "norm_max"):
                totals[key] = max(totals[key], contribution)
                sums[key] += contribution
            else:
                totals[key] += contribution
            matching_branches[key].add(candidate.source)
            representatives.setdefault(key, candidate)
            components[key][candidate.source] = candidate.raw_score
            # Đóng góp thực tế vào điểm fusion — cùng thang đo giữa mọi branch,
            # khác với component_scores (raw, không so sánh được với nhau).
            contributions[key][candidate.source] = (
                contributions[key].get(candidate.source, 0.0) + contribution
            )
            modalities[key].add(candidate.modality.value)
            # Candidate neo frame mang tọa độ submission chính xác hơn candidate
            # neo scene; giữ lại cái có contribution cao nhất để tầng hydrate ưu
            # tiên frame do retrieval chỉ ra thay vì tự đoán lại.
            if candidate.frame_idx is not None:
                incumbent = frame_hints.get(key)
                if incumbent is None or contribution > contributions[key].get(
                    incumbent.source, 0.0
                ):
                    frame_hints[key] = candidate
            matched_text = candidate.payload.get("matched_text")
            if matched_text:
                evidence[key].append(
                    {
                        "modality": candidate.modality.value,
                        "text": matched_text,
                        "score": candidate.raw_score,
                    }
                )

    eligible = totals.keys()
    if method == "intersection":
        threshold = max(2, minimum_matching_branches)
        eligible = [key for key in totals if len(matching_branches[key]) >= threshold]

    ordered = sorted(eligible, key=lambda item: (-totals[item], -sums[item], item))[:limit]
    output: list[Candidate] = []
    for rank, key in enumerate(ordered, start=1):
        base = representatives[key]
        hint = frame_hints.get(key)
        payload = dict(base.payload)
        payload.update(
            {
                "component_scores": components[key],
                "branch_contributions": contributions[key],
                "matched_modalities": sorted(modalities[key]),
                "evidence": evidence[key],
                "matched_branches": sorted(matching_branches[key]),
            }
        )
        output.append(
            Candidate(
                candidate_id=key,
                entity_type=base.entity_type,
                scene_id=base.scene_id,
                clip_id=base.clip_id,
                event_id=base.event_id or (hint.event_id if hint else None),
                video_id=base.video_id,
                frame_idx=hint.frame_idx if hint else base.frame_idx,
                timestamp_sec=hint.timestamp_sec if hint else base.timestamp_sec,
                start_frame=base.start_frame,
                end_frame=base.end_frame,
                source=f"fusion_{method}",
                modality=base.modality,
                raw_score=totals[key],
                score_kind="fusion",
                rank=rank,
                payload=payload,
            )
        )
    return output


def weighted_rrf(
    ranked_lists: list[list[Candidate]],
    modality_weights: dict[Modality, float],
    *,
    rrf_k: int = 60,
    limit: int = 100,
    branches: dict[str, BranchRuntimeOptions] | None = None,
) -> list[Candidate]:
    """Backward-compatible alias for `fuse_candidates(..., method="rrf")`."""

    return fuse_candidates(
        ranked_lists, modality_weights, method="rrf", rrf_k=rrf_k, limit=limit, branches=branches,
    )


__all__ = ["FusionMethod", "fuse_candidates", "weighted_rrf"]
