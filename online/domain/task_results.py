"""Kết quả riêng của từng task, đúng đơn vị nộp bài (PR-07).

`SearchHit` là kết quả retrieval chung. Bốn task nộp bốn thứ khác nhau, và
gộp chúng vào một model duy nhất sẽ khiến QA thiếu answer còn TRAKE thiếu
danh sách frame — đúng thứ đã xảy ra trước PR-07:

    TEXTUAL_KIS  -> <video_id>, <frame_idx>
    QA           -> <video_id>, <frame_idx>, <answer>
    TRAKE        -> <video_id>, <frame_idx_1>, ..., <frame_idx_n>
    AVS          -> danh sách segment có relevance grade (task nội bộ)
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from online.domain.base import StrictModel
from online.domain.evidence import EvidencePack

VerifierStatus = Literal["SUPPORTED", "PARTIAL", "CONTRADICTED", "INSUFFICIENT"]

AnswerType = Literal[
    "count", "color", "ocr_text", "asr_text", "entity", "yes_no", "temporal", "other"
]


class PlaybackWindow(StrictModel):
    """Đoạn video cần phát cho một kết quả, đã nới thêm bối cảnh.

    Trước đây UI không phát được đoạn của kết quả vì không result nào mang
    thời gian LẪN đường dẫn video: KIS/QA chỉ có giây khi `evidence` đi kèm,
    TRAKE/AVS không có giây nào cả.

    `start_sec`/`end_sec` ĐÃ cộng `pad_sec` mỗi phía — scene p50 chỉ 4.1 giây,
    xem đúng 4 giây thì không đủ hiểu chuyện gì. `focus_sec` là thời điểm của
    frame được nộp; UI nên nhảy tới đó, phần nới chỉ là bối cảnh.
    """

    video_id: str
    #: Đường dẫn TƯƠNG ĐỐI so với `AIC_DATA_ROOT`, cùng quy ước với
    #: `video_path`/`image_path`. KHÔNG kèm prefix `/v1/media/` — client tự
    #: ghép, và bake sẵn prefix vào đây làm URL bị nhân đôi thành
    #: `/v1/media/%2Fv1%2Fmedia%2F...` rồi trả 400 "invalid media path".
    media_path: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)
    focus_sec: float = Field(ge=0.0)
    pad_sec: float = Field(ge=0.0)


class KisResultItem(StrictModel):
    rank: int = Field(ge=1)
    video_id: str
    frame_idx: int = Field(ge=0)
    scene_id: str | None = None
    event_id: str | None = None
    score: float
    safe_frame_score: float | None = None
    must_match_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: EvidencePack | None = None
    playback: PlaybackWindow | None = None


class AnswerCandidate(StrictModel):
    canonical: str
    surface: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    answer_type: AnswerType = "other"
    source: str = "unknown"


class QaResultItem(StrictModel):
    rank: int = Field(ge=1)
    video_id: str
    frame_idx: int = Field(ge=0)
    answer: str
    canonical_answer: str
    answer_type: AnswerType = "other"
    joint_score: float = Field(ge=0.0)
    verifier_status: VerifierStatus
    scene_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: EvidencePack | None = None
    playback: PlaybackWindow | None = None

    @model_validator(mode="after")
    def require_non_empty_answer(self) -> "QaResultItem":
        # QA sai một trong ba (video/frame/answer) là 0 điểm; answer rỗng nghĩa
        # là item này chắc chắn không ghi điểm, đừng để nó lọt vào submission.
        if not self.answer.strip():
            raise ValueError("QA result requires a non-empty answer")
        return self


class TrakeStep(StrictModel):
    step: int = Field(ge=1)
    frame_idx: int = Field(ge=0)
    scene_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    # Frame được tinh chỉnh tới mức nào — quan trọng vì cửa sổ GT chỉ ±4 frame.
    refinement: Literal["keyframe_only", "dense_window"] = "keyframe_only"
    # Đường dẫn ảnh keyframe thật đã chọn — None khi anchor nằm ngoài scene
    # (không có frame thật để trỏ tới, xem frame_refinement.py). UI dùng để
    # hiển thị thumbnail cho mỗi step card, không suy đoán theo path convention.
    image_path: str | None = None
    # Timestamp thật của frame (giây) — lấy trực tiếp từ FrameEvidence/SearchHit,
    # không suy ra từ fps (fps không có endpoint nào expose cho frontend).
    timestamp_sec: float | None = None


class TrakeResultItem(StrictModel):
    rank: int = Field(ge=1)
    video_id: str
    frame_ids: list[int] = Field(min_length=1)
    sequence_score: float
    steps: list[TrakeStep] = Field(default_factory=list)
    step_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    ordering_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Số thứ tự (1-based) của các step KHÔNG tìm được candidate nào trong
    # video này — rỗng nghĩa là chain đầy đủ. Không có field này thì output
    # không phân biệt được "chain đủ 5 step" với "chain 5 frame nhưng thật ra
    # chỉ có 3 step thật, 2 frame còn lại là step khác bị dồn vào" (PR-14A).
    missing_steps: list[int] = Field(default_factory=list)
    # TRAKE nộp NHIỀU frame; cửa sổ phát trải từ frame đầu tới frame cuối của
    # chuỗi, không phải một scene, để người chấm xem được cả chuỗi hành động.
    playback: PlaybackWindow | None = None
    # True khi video được chọn dù dưới min_step_coverage (không còn lựa chọn
    # nào khác) — kết quả suy yếu, tầng hiển thị nên cảnh báo rõ với người dùng.
    degraded: bool = False

    @model_validator(mode="after")
    def require_strictly_increasing_frames(self) -> "TrakeResultItem":
        if any(
            later <= earlier
            for earlier, later in zip(self.frame_ids, self.frame_ids[1:], strict=False)
        ):
            raise ValueError(
                f"TRAKE frame_ids must strictly increase in time, got {self.frame_ids}"
            )
        return self


class AvsResultItem(StrictModel):
    rank: int = Field(ge=1)
    video_id: str
    segment_id: str
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    relevance_grade: int = Field(ge=0, le=3)
    score: float
    cluster_id: str | None = None
    best_frame_idx: int | None = Field(default=None, ge=0)
    playback: PlaybackWindow | None = None


__all__ = [
    "PlaybackWindow",
    "AnswerCandidate",
    "AnswerType",
    "AvsResultItem",
    "KisResultItem",
    "QaResultItem",
    "TrakeResultItem",
    "TrakeStep",
    "VerifierStatus",
]
