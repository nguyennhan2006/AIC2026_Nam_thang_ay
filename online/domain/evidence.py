"""Evidence pack — bằng chứng đầy đủ cho một candidate (PR-06).

`Evidence` cũ chỉ có `{modality, text, score}`: đủ để hiển thị một dòng gợi ý,
không đủ để (a) một reranker VLM phán đoán, (b) người dùng kiểm chứng trước
khi nộp, hay (c) tái lập lại kết quả sau khi đổi index.

Pack được dựng **lazy** cho top-N: hydrate hàng nghìn candidate với đầy đủ
neighbor context là lãng phí, và nhồi cả JSON dài vào prompt VLM còn làm giảm
chất lượng rerank.
"""

from __future__ import annotations

from pydantic import Field

from online.domain.base import StrictModel
from online.domain.candidate import FrameEvidence
from online.domain.scores import BranchScore


class RuleAdjustment(StrictModel):
    """Một rule đã cộng/trừ bao nhiêu điểm, để giải thích được thứ hạng."""

    rule: str
    delta: float
    detail: str | None = None


class NeighborContext(StrictModel):
    """Scene liền trước/liền sau — cần cho câu hỏi before/after và TRAKE."""

    scene_id: str
    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    caption: str | None = None
    ocr_text: str | None = None


class EvidencePack(StrictModel):
    """Toàn bộ bằng chứng của một candidate, đủ để verify và để tái lập."""

    candidate_id: str
    video_id: str
    scene_id: str | None = None
    event_id: str | None = None

    start_frame: int = Field(ge=0)
    end_frame_exclusive: int = Field(gt=0)
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)

    keyframes: list[FrameEvidence] = Field(default_factory=list)
    best_frame_idx: int | None = Field(default=None, ge=0)
    asr_window: str | None = None
    caption_text: str | None = None
    ocr_text: str | None = None

    previous_context: NeighborContext | None = None
    next_context: NeighborContext | None = None

    branch_scores: dict[str, BranchScore] = Field(default_factory=dict)
    branch_contributions: dict[str, float] = Field(default_factory=dict)
    rule_adjustments: list[RuleAdjustment] = Field(default_factory=list)

    model_versions: dict[str, str] = Field(default_factory=dict)
    index_versions: dict[str, str] = Field(default_factory=dict)
    dataset_version: str | None = None

    def rerank_text(self, *, max_chars: int = 1200) -> str:
        """Văn bản gọn đưa vào text reranker.

        Cố tình KHÔNG phải `model_dump_json()`: reranker đọc JSON thô sẽ tiêu
        phần lớn ngân sách token cho tên field và dấu ngoặc.
        """

        parts: list[str] = []
        if self.caption_text:
            parts.append(f"Caption: {self.caption_text}")
        if self.ocr_text:
            parts.append(f"OCR: {self.ocr_text}")
        if self.asr_window:
            parts.append(f"ASR: {self.asr_window}")
        objects = sorted({label for frame in self.keyframes for label in frame.object_labels})
        if objects:
            parts.append("Objects: " + ", ".join(objects[:20]))
        actions = sorted({tag for frame in self.keyframes for tag in frame.action_tags})
        if actions:
            parts.append("Actions: " + ", ".join(actions[:10]))
        return "\n".join(parts)[:max_chars]

    def representative_frames(self, limit: int = 3) -> list[FrameEvidence]:
        """Frame đại diện cho VLM: ưu tiên best frame, rồi trải đều phần còn lại."""

        if not self.keyframes:
            return []
        ordered = sorted(self.keyframes, key=lambda frame: frame.frame_idx)
        best = next(
            (frame for frame in ordered if frame.frame_idx == self.best_frame_idx), ordered[0]
        )
        rest = [frame for frame in ordered if frame is not best]
        if len(rest) <= limit - 1:
            return [best, *rest]
        step = len(rest) / (limit - 1)
        sampled = [rest[min(int(index * step), len(rest) - 1)] for index in range(limit - 1)]
        return [best, *sampled]


__all__ = ["EvidencePack", "NeighborContext", "RuleAdjustment"]
