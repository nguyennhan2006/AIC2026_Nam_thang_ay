"""TRAKE Stage B — beam search chuỗi step trong MỘT video (PR-07).

Khác `online/services/temporal.py` (bản trước PR-07) ở ba điểm:

1. Chạy trong phạm vi một video đã khóa ở Stage A, nên không phải cân nhắc
   giả thuyết xuyên video ở mỗi bước.
2. So sánh theo `frame_idx` chứ không `scene_idx`: hai step hoàn toàn có thể
   nằm trong cùng một scene (scene 10 giây chứa nhiều khoảnh khắc).
3. Có ràng buộc khoảng cách: hai step cách nhau 20 phút gần như chắc chắn
   không thuộc cùng một diễn biến.
"""

from __future__ import annotations

from dataclasses import dataclass

from online.domain.models import SearchHit


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    beam_size: int = 50
    min_gap_frames: int = 1
    max_gap_sec: float = 300.0
    gap_penalty_per_sec: float = 0.002
    # Cho phép bỏ qua một step không tìm được bằng chứng thay vì vứt cả chuỗi:
    # thiếu 1/4 step vẫn được 0.75 điểm, còn không có chuỗi nào thì được 0.
    allow_missing_steps: bool = True
    missing_step_penalty: float = 0.5


@dataclass(frozen=True, slots=True)
class SequenceHypothesis:
    video_id: str
    hits: tuple[SearchHit | None, ...]
    score: float

    @property
    def frame_ids(self) -> list[int]:
        return [hit.best_frame_idx for hit in self.hits if hit is not None]

    @property
    def covered(self) -> int:
        return sum(1 for hit in self.hits if hit is not None)


def _timestamp(hit: SearchHit) -> float:
    return hit.best_timestamp_sec if hit.best_timestamp_sec is not None else hit.start_sec


def search_sequences(
    video_id: str,
    step_hits: list[list[SearchHit]],
    config: SequenceConfig | None = None,
    *,
    limit: int = 100,
) -> list[SequenceHypothesis]:
    """Beam search các chuỗi frame tăng dần trong `video_id`."""

    config = config or SequenceConfig()
    if not step_hits:
        return []
    in_video = [
        sorted(
            (hit for hit in hits if hit.video_id == video_id),
            key=lambda hit: hit.best_frame_idx,
        )
        for hits in step_hits
    ]
    if all(not hits for hits in in_video):
        return []

    best_score = max((hit.score for hits in in_video for hit in hits), default=0.0) or 1.0
    # Beam khởi tạo: mỗi ứng viên của step 1, cộng nhánh "bỏ qua step 1".
    beams: list[tuple[tuple[SearchHit | None, ...], float, int]] = [
        ((hit,), hit.score / best_score, hit.best_frame_idx) for hit in in_video[0]
    ]
    if config.allow_missing_steps:
        beams.append(((None,), -config.missing_step_penalty, -1))
    if not beams:
        return []

    for hits in in_video[1:]:
        expanded: list[tuple[tuple[SearchHit | None, ...], float, int]] = []
        for sequence, score, last_frame in beams:
            for hit in hits:
                if hit.best_frame_idx < last_frame + config.min_gap_frames:
                    continue
                gap_sec = 0.0
                previous = next(
                    (item for item in reversed(sequence) if item is not None), None
                )
                if previous is not None:
                    gap_sec = max(0.0, _timestamp(hit) - _timestamp(previous))
                    if gap_sec > config.max_gap_sec:
                        continue
                expanded.append((
                    (*sequence, hit),
                    score + hit.score / best_score - config.gap_penalty_per_sec * gap_sec,
                    hit.best_frame_idx,
                ))
            if config.allow_missing_steps:
                expanded.append((
                    (*sequence, None), score - config.missing_step_penalty, last_frame
                ))
        if not expanded:
            return []
        expanded.sort(key=lambda item: (-item[1], [
            hit.best_frame_idx if hit is not None else -1 for hit in item[0]
        ]))
        beams = expanded[: config.beam_size]

    hypotheses = [
        SequenceHypothesis(video_id=video_id, hits=sequence, score=score)
        for sequence, score, _last in beams
        if any(hit is not None for hit in sequence)
    ]
    hypotheses.sort(key=lambda item: (-item.score, item.frame_ids))
    return hypotheses[:limit]


def local_variants(
    hypothesis: SequenceHypothesis, *, offsets: tuple[int, ...] = (-1, 1)
) -> list[list[int]]:
    """Sinh biến thể ±1 frame quanh chuỗi tốt nhất.

    Cửa sổ GT của TRAKE chỉ rộng 9 frame (±4), nên lệch một frame vẫn có thể
    ăn điểm. Đây là cách rẻ để lấp các mốc 2–5 của submission mà không cần
    thêm một giả thuyết video mới.
    """

    base = hypothesis.frame_ids
    variants: list[list[int]] = []
    for index in range(len(base)):
        for offset in offsets:
            shifted = list(base)
            shifted[index] = max(0, shifted[index] + offset)
            if all(
                later > earlier
                for earlier, later in zip(shifted, shifted[1:], strict=False)
            ):
                variants.append(shifted)
    return variants


__all__ = ["SequenceConfig", "SequenceHypothesis", "local_variants", "search_sequences"]
