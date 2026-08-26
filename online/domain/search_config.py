"""Search Mixing Console — configuration contracts (W0).

Domain contract only. KHÔNG đấu nối vào `SearchService`/`container.py` ở bước này
(đó là W3/W5 theo kế hoạch) — mọi field ở đây chỉ định nghĩa, validate và có default,
chưa có branch/service nào đọc `SearchOptions` thật. Xem plan gốc (mục 4-5, workstream
W0-W8) trong lịch sử phiên làm việc cho spec đầy đủ.

Toàn bộ field optional/có default để tương thích ngược: `SearchRequest.search_options
= None` phải giữ nguyên hành vi search hiện tại (không silent thay đổi mặc định).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from online.domain.base import StrictModel
from online.domain.candidate import Modality


class BranchRuntimeOptions(StrictModel):
    """Cấu hình runtime cho một retrieval branch (vd "dense_visual_frame", "ocr_fuzzy").

    `parameters` chứa tham số riêng của từng loại branch (vd fuzzy_ratio cho OCR) —
    được validate tiếp bởi adapter tương ứng khi branch đó được wire (W3), không
    validate ở tầng contract này.
    """

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    top_k: int = Field(default=300, ge=1, le=5000)
    min_score: float | None = None
    threshold_space: Literal["raw", "normalized", "percentile"] = "normalized"
    threshold_policy: Literal["hard", "soft"] = "soft"
    query_variant: Literal[
        "raw", "normalized", "english", "bilingual", "expanded", "hyde"
    ] = "raw"
    model_id: str | None = None
    index_id: str | None = None
    timeout_ms: int = Field(default=3000, ge=100, le=60000)
    field_weights: dict[str, float] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class QueryProcessingOptions(StrictModel):
    preserve_raw_query: bool = True
    normalize_query: bool = True
    generate_english_variant: bool = True
    generate_bilingual_variant: bool = True
    enable_decomposition: bool = True
    enable_expansion: bool = False
    enable_hyde: bool = False
    enable_negative_constraints: bool = True
    enable_temporal_parsing: bool = True


class FusionOptions(StrictModel):
    method: Literal[
        "rrf", "weighted_sum", "max_score", "intersection", "union",
        # Phase D docs/31 — doc diem that thay vi suy tu rank.
        "norm_sum", "norm_max", "margin_sum", "entropy_sum",
    ] = "rrf"
    fusion_top_k: int = Field(default=1000, ge=1, le=10000)
    rrf_k: int = Field(default=60, ge=1, le=500)
    minimum_matching_branches: int = Field(default=1, ge=1)
    normalized_score_method: Literal["minmax", "percentile", "calibrated"] = "percentile"
    dedup_scope: Literal["none", "frame", "scene", "event"] = "scene"
    dedup_similarity: float = Field(default=0.97, ge=0, le=1)
    max_results_per_video: int | None = None


class TextRerankOptions(StrictModel):
    enabled: bool = True
    model_id: str = "bge-reranker-v2-m3"
    input_top_k: int = 300
    output_top_k: int = 100  # Giữ đủ 100 candidates cho submission
    min_score: float | None = None
    weight: float = 1.0


class VlmRerankOptions(StrictModel):
    enabled: bool = True
    model_id: str = "qwen3-vl-32b"
    input_top_k: int = 20
    frames_per_candidate: int = 3
    output_top_k: int = 20
    timeout_ms: int = 30000
    weight: float = 1.0


class RerankOptions(StrictModel):
    enable_rules: bool = True
    text: TextRerankOptions = Field(default_factory=TextRerankOptions)
    vlm: VlmRerankOptions = Field(default_factory=VlmRerankOptions)
    temporal_verifier: bool = True


class TemporalOptions(StrictModel):
    enabled: bool = True
    same_video_required: bool = True
    ordered_steps_required: bool = True
    minimum_gap_sec: float = 0
    maximum_gap_sec: float = 300
    allow_missing_optional_step: bool = False
    neighbor_before_sec: float = 5
    neighbor_after_sec: float = 10
    # UI competition studio — expose đúng tham số ĐÃ CÓ SẴN trong
    # VideoRetrieverConfig/SequenceConfig (online/services/trake/) qua request
    # thay vì cố định lúc container build. KHÔNG đổi thuật toán: chỉ cho override
    # số, không thêm bước tính mới. None = giữ giá trị mặc định của deployment.
    order_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    # Phạt khoảng cách MỀM: min(lambda * max(0, gap - free_gap_sec), cap).
    # `free_gap_sec` là vùng miễn phạt (gold TRAKE cách nhau tối đa 36s, nên
    # mặc định 60s cho chuỗi đúng đi qua sạch), `gap_penalty_cap` là trần để
    # phạt không bao giờ lấn át độ liên quan. Xem online/services/temporal_gap.py.
    gap_penalty_per_sec: float | None = Field(default=None, ge=0.0, le=1.0)
    free_gap_sec: float | None = Field(default=None, ge=0.0, le=3600.0)
    gap_penalty_cap: float | None = Field(default=None, ge=0.0, le=5.0)
    missing_step_penalty: float | None = Field(default=None, ge=0.0, le=5.0)
    # "beam" | "dp" — cách ghép chuỗi. Xem search_sequences_dp.
    sequence_strategy: Literal["beam", "dp"] | None = None
    # PR-4B Stage A. Đo được: video ĐÚNG thắng áp đảo ở `context` (0.89 vs
    # 0.31) nhưng vẫn thua vì `ordering` và `duplicate_penalty`. Cho chỉnh
    # qua request để chạy ablation mà không sửa code.
    video_context_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    video_duplicate_penalty: float | None = Field(default=None, ge=0.0, le=5.0)
    video_coverage_weight: float | None = Field(default=None, ge=0.0, le=5.0)
    # Override trọng số modality riêng cho từng step TRAKE (thay compute_modality_
    # weights() tự động) — index theo thứ tự event 0-based; thiếu step nào thì
    # step đó vẫn dùng suy luận tự động từ nội dung như trước.
    step_modality_weights: list[dict[Modality, float]] = Field(default_factory=list)


class ResultOptions(StrictModel):
    display_top_k: int = 100
    display_min_score: float | None = None
    group_by: Literal["none", "video", "scene", "event", "cluster"] = "none"
    sort_by: Literal[
        "final_score", "visual_score", "caption_score", "ocr_score", "asr_score", "time"
    ] = "final_score"


class SearchOptions(StrictModel):
    """Root config object cho Search Mixing Console — mọi field optional/có default
    để `SearchRequest.search_options = None` giữ nguyên hành vi search hiện tại."""

    profile_id: str | None = None
    query: QueryProcessingOptions = Field(default_factory=QueryProcessingOptions)
    branches: dict[str, BranchRuntimeOptions] = Field(default_factory=dict)
    fusion: FusionOptions = Field(default_factory=FusionOptions)
    rerank: RerankOptions = Field(default_factory=RerankOptions)
    temporal: TemporalOptions = Field(default_factory=TemporalOptions)
    results: ResultOptions = Field(default_factory=ResultOptions)
