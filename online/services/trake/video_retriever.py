"""TRAKE Stage A — khóa đúng video trước khi căn chỉnh (PR-07).

Luật TRAKE: **sai video là 0 điểm**, đúng video thì điểm là tỉ lệ step rơi
đúng cửa sổ GT. Vì vậy quyết định đắt nhất là chọn video, và nó phải được ra
bằng bằng chứng gộp của TẤT CẢ các step — không phải bằng scene tốt nhất của
một step như `link_event_hits` trước PR-07 vẫn làm.

Điểm của một video::

    step_coverage         bao nhiêu step tìm thấy bằng chứng trong video này
  + ordered_pair_coverage bao nhiêu cặp step liên tiếp đúng thứ tự thời gian
  + context               độ mạnh trung bình của bằng chứng
  - duplicate_penalty     nhiều step dồn vào đúng một scene (teaser/nhạc hiệu)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from online.domain.models import SearchHit


@dataclass(frozen=True, slots=True)
class VideoCandidate:
    video_id: str
    score: float
    step_coverage: float
    ordered_pair_coverage: float
    context_score: float
    duplicate_penalty: float
    # Hit tốt nhất của từng step trong video này, index theo thứ tự step.
    best_per_step: tuple[SearchHit | None, ...] = ()
    # True khi video này lọt qua chỉ vì KHÔNG còn ứng viên nào khác đạt
    # min_step_coverage — kết quả cho video này nên bị coi là suy yếu (partial),
    # không phải một match đáng tin như bình thường (xem rank_videos()).
    below_min_coverage: bool = False

    @property
    def covered_steps(self) -> int:
        return sum(1 for item in self.best_per_step if item is not None)


@dataclass(frozen=True, slots=True)
class VideoRetrieverConfig:
    coverage_weight: float = 1.0
    ordering_weight: float = 0.6
    context_weight: float = 0.4
    duplicate_penalty: float = 0.5
    # Video thiếu quá nửa số step gần như chắc chắn sai; loại sớm cho rẻ.
    min_step_coverage: float = 0.5
    top_videos: int = 5


def _best_per_step(
    step_hits: list[list[SearchHit]], video_id: str
) -> list[SearchHit | None]:
    """Hit điểm cao nhất của từng step trong một video."""

    result: list[SearchHit | None] = []
    for hits in step_hits:
        in_video = [hit for hit in hits if hit.video_id == video_id]
        result.append(max(in_video, key=lambda hit: hit.score) if in_video else None)
    return result


def rank_videos(
    step_hits: list[list[SearchHit]], config: VideoRetrieverConfig | None = None
) -> list[VideoCandidate]:
    """Xếp hạng video theo bằng chứng gộp của mọi step.

    KHÔNG bao giờ trả rỗng chỉ vì mọi video đều dưới `min_step_coverage`, MIỄN
    có ít nhất một video có bằng chứng thật (step_coverage > 0): giữ lại ứng
    viên tốt nhất, đánh dấu `below_min_coverage=True` để tầng trên biết đây là
    kết quả suy yếu — thay vì trả `[]` khiến toàn bộ query TRAKE mất trắng chỉ
    vì thiếu 1 step trong lúc chỉ có đúng một video ứng viên (xem PR-14A: đây
    là nguyên nhân TRAKE có thể trả rỗng dù rank_videos tìm được bằng chứng).
    """

    config = config or VideoRetrieverConfig()
    if not step_hits:
        return []
    step_count = len(step_hits)
    videos = {hit.video_id for hits in step_hits for hit in hits}
    best_overall = max(
        (hit.score for hits in step_hits for hit in hits), default=0.0
    ) or 1.0

    all_candidates: list[VideoCandidate] = []
    for video_id in videos:
        per_step = _best_per_step(step_hits, video_id)
        covered = [item for item in per_step if item is not None]
        step_coverage = len(covered) / step_count
        if not covered:
            continue  # không một step nào có bằng chứng: không đáng làm fallback

        # Cặp liên tiếp đúng thứ tự: cả hai step đều có bằng chứng và step sau
        # nằm sau step trước trên trục frame.
        pairs = list(zip(per_step, per_step[1:], strict=False))
        ordered = sum(
            1
            for earlier, later in pairs
            if earlier is not None and later is not None
            and earlier.best_frame_idx < later.best_frame_idx
        )
        ordered_coverage = ordered / len(pairs) if pairs else 1.0

        context = sum(item.score for item in covered) / (len(covered) * best_overall) if covered else 0.0

        # Nhiều step trỏ về cùng một scene => nhiều khả năng đó là đoạn tóm tắt
        # đầu bản tin chứ không phải diễn biến thật.
        scene_ids = [item.scene_id for item in covered]
        distinct = len(set(scene_ids))
        duplicate = 1.0 - (distinct / len(scene_ids)) if scene_ids else 0.0

        score = (
            config.coverage_weight * step_coverage
            + config.ordering_weight * ordered_coverage
            + config.context_weight * context
            - config.duplicate_penalty * duplicate
        )
        all_candidates.append(VideoCandidate(
            video_id=video_id,
            score=score,
            step_coverage=step_coverage,
            ordered_pair_coverage=ordered_coverage,
            context_score=context,
            duplicate_penalty=duplicate,
            best_per_step=tuple(per_step),
        ))

    passing = [item for item in all_candidates if item.step_coverage >= config.min_step_coverage]
    if passing:
        candidates = passing
    elif all_candidates:
        best = max(all_candidates, key=lambda item: (item.score, item.video_id))
        candidates = [replace(best, below_min_coverage=True)]
    else:
        candidates = []

    candidates.sort(key=lambda item: (-item.score, item.video_id))
    return candidates[: config.top_videos]


__all__ = ["VideoCandidate", "VideoRetrieverConfig", "rank_videos"]
