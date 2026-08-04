"""TRAKE processor: video-first -> sequence -> frame refinement (PR-07).

Ba giai đoạn tách hẳn nhau vì chúng trả lời ba câu hỏi khác nhau và sai ở
giai đoạn nào thì hỏng theo kiểu khác nhau:

    Stage A  video nào?      sai -> 0 điểm, không cứu được
    Stage B  thứ tự nào?     sai -> lệch step, còn ăn điểm một phần
    Stage C  frame nào?      sai -> trượt cửa sổ 9 frame

Trước PR-07 toàn bộ TRAKE là 41 dòng trong `online/services/temporal.py`:
chỉ có một phần của Stage B, không hề khóa video và không tinh chỉnh frame.
"""

from __future__ import annotations

from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING

from online.domain.models import SceneDocument, SearchHit
from online.domain.task_results import TrakeResultItem, TrakeStep
from online.services.trake.frame_refinement import RefinementConfig, refine_step
from online.services.trake.sequence_search import (
    SequenceConfig,
    SequenceHypothesis,
    local_variants,
    search_sequences,
)
from online.services.trake.video_retriever import (
    VideoCandidate,
    VideoRetrieverConfig,
    rank_videos,
)

if TYPE_CHECKING:
    from online.domain.search_config import TemporalOptions


class TrakeProcessor:
    """Chạy đủ ba giai đoạn và trả về danh sách frame theo thứ tự."""

    def __init__(
        self,
        *,
        video_config: VideoRetrieverConfig | None = None,
        sequence_config: SequenceConfig | None = None,
        refinement_config: RefinementConfig | None = None,
    ) -> None:
        self.video_config = video_config or VideoRetrieverConfig()
        self.sequence_config = sequence_config or SequenceConfig()
        self.refinement_config = refinement_config or RefinementConfig()

    def run(
        self,
        step_queries: list[str],
        step_hits: list[list[SearchHit]],
        documents: dict[str, SceneDocument],
        *,
        limit: int = 100,
        include_variants: bool = True,
    ) -> list[TrakeResultItem]:
        videos = rank_videos(step_hits, self.video_config)
        if not videos:
            return []

        results: list[TrakeResultItem] = []
        for video in videos:
            hypotheses = search_sequences(
                video.video_id, step_hits, self.sequence_config, limit=limit
            )
            for hypothesis in hypotheses:
                item = self._to_result(
                    video, hypothesis, step_queries, documents, rank=len(results) + 1
                )
                if item is None:
                    continue
                results.append(item)
                if len(results) >= limit:
                    return results
            # Biến thể ±1 frame của giả thuyết tốt nhất: rẻ, và cửa sổ GT chỉ
            # rộng 9 frame nên lệch một frame vẫn có cơ hội ăn điểm.
            if include_variants and hypotheses and len(results) < limit:
                base = results[-1] if results else None
                if base is not None:
                    for frames in local_variants(hypotheses[0]):
                        if len(results) >= limit:
                            break
                        results.append(base.model_copy(update={
                            "rank": len(results) + 1,
                            "frame_ids": frames,
                            "sequence_score": base.sequence_score * 0.9,
                        }))
        return results[:limit]

    def _to_result(
        self,
        video: VideoCandidate,
        hypothesis: SequenceHypothesis,
        step_queries: list[str],
        documents: dict[str, SceneDocument],
        *,
        rank: int,
    ) -> TrakeResultItem | None:
        steps: list[TrakeStep] = []
        missing_steps: list[int] = []
        for index, hit in enumerate(hypothesis.hits):
            if hit is None:
                missing_steps.append(index + 1)
                continue
            document = documents.get(hit.scene_id)
            query = step_queries[index] if index < len(step_queries) else ""
            if document is None:
                steps.append(TrakeStep(
                    step=index + 1, frame_idx=hit.best_frame_idx,
                    scene_id=hit.scene_id, confidence=0.3,
                    image_path=hit.best_keyframe_path,
                    timestamp_sec=hit.best_timestamp_sec,
                ))
                continue
            steps.append(refine_step(
                index + 1, query, document, hit.best_frame_idx,
                config=self.refinement_config,
            ))
        if not steps:
            return None

        # Tinh chỉnh có thể phá thứ tự tăng dần (hai step trong cùng scene bị
        # kéo về cùng một frame). Sửa lại bằng cách đẩy step sau lên tối thiểu
        # 1 frame — chuỗi không tăng dần là submission không hợp lệ.
        frame_ids: list[int] = []
        for step in steps:
            frame_idx = step.frame_idx
            if frame_ids and frame_idx <= frame_ids[-1]:
                frame_idx = frame_ids[-1] + 1
            frame_ids.append(frame_idx)
        steps = [
            step.model_copy(update={"frame_idx": frame_ids[index]})
            for index, step in enumerate(steps)
        ]
        return TrakeResultItem(
            rank=rank,
            video_id=video.video_id,
            frame_ids=frame_ids,
            sequence_score=hypothesis.score + video.score,
            steps=steps,
            step_coverage=video.step_coverage,
            ordering_score=video.ordered_pair_coverage,
            missing_steps=missing_steps,
            degraded=video.below_min_coverage,
        )


def trake_processor_for_request(
    base: "TrakeProcessor", options: "TemporalOptions"
) -> "TrakeProcessor":
    """Override đúng tham số ĐÃ CÓ SẴN của `base` theo `options`, không đổi gì
    khác — dùng cho UI competition studio (TRAKE Alignment sliders) để tinh
    chỉnh theo từng request thay vì cố định lúc container build.

    Chỉ field caller ĐẶT TƯỜNG MINH mới ghi đè (`model_fields_set`), nên
    request không gửi `search_options` vẫn dùng đúng `base` (không đổi hành vi
    cũ) — cùng nguyên tắc `fusion.rrf_k` đã áp dụng ở `_retrieve()`.
    """

    fields_set = options.model_fields_set
    video_config = base.video_config
    sequence_config = base.sequence_config

    if "order_weight" in fields_set and options.order_weight is not None:
        video_config = _dataclass_replace(video_config, ordering_weight=options.order_weight)
    if "gap_penalty_per_sec" in fields_set and options.gap_penalty_per_sec is not None:
        sequence_config = _dataclass_replace(
            sequence_config, gap_penalty_per_sec=options.gap_penalty_per_sec
        )
    if "missing_step_penalty" in fields_set and options.missing_step_penalty is not None:
        sequence_config = _dataclass_replace(
            sequence_config, missing_step_penalty=options.missing_step_penalty
        )
    if "maximum_gap_sec" in fields_set:
        sequence_config = _dataclass_replace(sequence_config, max_gap_sec=options.maximum_gap_sec)
    if "allow_missing_optional_step" in fields_set:
        sequence_config = _dataclass_replace(
            sequence_config, allow_missing_steps=options.allow_missing_optional_step
        )

    if video_config is base.video_config and sequence_config is base.sequence_config:
        return base
    return TrakeProcessor(
        video_config=video_config,
        sequence_config=sequence_config,
        refinement_config=base.refinement_config,
    )


__all__ = [
    "RefinementConfig",
    "SequenceConfig",
    "SequenceHypothesis",
    "TrakeProcessor",
    "VideoCandidate",
    "VideoRetrieverConfig",
    "local_variants",
    "rank_videos",
    "refine_step",
    "trake_processor_for_request",
    "search_sequences",
]
