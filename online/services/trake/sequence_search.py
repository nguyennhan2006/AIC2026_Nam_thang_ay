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
    # "beam" (mặc định, giữ nguyên hành vi cũ) hoặc "dp" (chính xác, xem
    # search_sequences_dp). Cờ này để ĐO xem beam có bỏ sót nghiệm không.
    strategy: str = "beam"


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


def search_sequences_dp(
    video_id: str,
    step_hits: list[list[SearchHit]],
    config: SequenceConfig | None = None,
    *,
    limit: int = 100,
) -> list[SequenceHypothesis]:
    """Quy hoạch động CHÍNH XÁC, cùng hàm mục tiêu với `search_sequences`.

    Tồn tại để tách bạch hai câu hỏi vẫn hay bị gộp làm một:

    1. *Có phải beam search bỏ sót chuỗi tốt hơn không?*  -> so DP với beam.
    2. *Có phải `s_i` là tín hiệu sai không?*             -> đổi `s_i`, giữ nguyên
       cách tìm.

    Beam giữ top-`beam_size` chuỗi dở dang ở mỗi bước, nên về lý thuyết có thể
    cắt nhầm một chuỗi đang kém nhưng về sau bù lại. DP không cắt gì.

    Trạng thái: `dp[f]` = tổng điểm tốt nhất của các bước đã xét, với `f` là
    frame của bước ĐƯỢC CHỌN gần nhất. Bỏ qua một bước không làm `f` tiến lên,
    nên ràng buộc tăng dần vẫn đúng. Với prefix-max thì mỗi bước là O(|F|),
    tổng **O(n · |F|)**.

    Lưu ý về `gap_penalty_per_sec`: DP bỏ qua nó. Phạt theo khoảng cách làm
    điểm phụ thuộc frame TRƯỚC ĐÓ chứ không chỉ frame hiện tại, tức hàm mục
    tiêu không còn cộng tính theo bước và DP mất tính chính xác. TRAKE-CONSTRAINT-01
    đã đo được tham số này không ảnh hưởng gì (0.263 ở cả bật lẫn tắt), nên bỏ
    nó là cái giá rẻ để đổi lấy một lời giải chính xác.
    """

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
    frames = sorted({hit.best_frame_idx for hits in in_video for hit in hits})
    # Vị trí 0 là mốc "chưa chọn bước nào" (frame = -vô cùng).
    position = {frame: index + 1 for index, frame in enumerate(frames)}
    size = len(frames) + 1
    # Mốc thời gian theo frame, để áp `max_gap_sec` giống hệt beam. Bỏ ràng
    # buộc này ra khỏi DP là sai lầm đã đo được: chuỗi trải tới 980 giây, nhảy
    # sang tận cuối video, và mean_r_score rơi 0.263 -> 0.094. `max_gap_sec` là
    # chặn CỨNG và nó đang làm việc thật — khác hẳn `gap_penalty_per_sec` (phạt
    # mềm), thứ mà TRAKE-CONSTRAINT-01 đo được là không ảnh hưởng gì.
    timestamp_at = [0.0] * size
    for hits in in_video:
        for hit in hits:
            timestamp_at[position[hit.best_frame_idx]] = _timestamp(hit)

    neg = float("-inf")
    dp: list[float] = [neg] * size
    dp[0] = 0.0
    # parent[i][p] = (vị trí trước đó, hit đã chọn ở bước i hoặc None)
    parents: list[list[tuple[int, SearchHit | None] | None]] = []

    for hits in in_video:
        nxt: list[float] = [neg] * size
        parent: list[tuple[int, SearchHit | None] | None] = [None] * size

        if config.allow_missing_steps:
            # Bỏ qua bước này: giữ nguyên frame cuối, chịu phạt.
            for index in range(size):
                if dp[index] > neg:
                    nxt[index] = dp[index] - config.missing_step_penalty
                    parent[index] = (index, None)

        for hit in hits:
            slot = position[hit.best_frame_idx]
            moment = _timestamp(hit)
            # Trạng thái trước đó hợp lệ khi: (a) là mốc "chưa chọn gì" — luôn
            # được, hoặc (b) frame nhỏ hơn ít nhất `min_gap_frames` VÀ cách
            # không quá `max_gap_sec`. Vế sau chính là ràng buộc mà beam áp ở
            # `sequence_search` dòng 98; thiếu nó thì DP giải một bài toán khác.
            #
            # Quét thẳng thay vì dùng deque đơn điệu: |F| là số frame ứng viên
            # của một video (hàng chục tới vài trăm) và n ≤ 6, nên O(n·|F|²)
            # vẫn không đáng kể so với chi phí retrieval.
            best_previous, best_from = dp[0], 0
            for index in range(1, slot):
                if dp[index] <= neg:
                    continue
                if frames[index - 1] > hit.best_frame_idx - config.min_gap_frames:
                    continue
                if moment - timestamp_at[index] > config.max_gap_sec:
                    continue
                if dp[index] > best_previous:
                    best_previous, best_from = dp[index], index
            if best_previous <= neg:
                continue
            candidate = best_previous + hit.score / best_score
            if candidate > nxt[slot]:
                nxt[slot] = candidate
                parent[slot] = (best_from, hit)

        dp = nxt
        parents.append(parent)
        if all(value <= neg for value in dp):
            return []

    end = max(range(size), key=lambda index: dp[index])
    if dp[end] <= neg:
        return []

    chain: list[SearchHit | None] = []
    cursor = end
    for parent in reversed(parents):
        entry = parent[cursor]
        if entry is None:
            return []
        cursor, hit = entry
        chain.append(hit)
    chain.reverse()

    if not any(hit is not None for hit in chain):
        return []
    return [SequenceHypothesis(video_id=video_id, hits=tuple(chain), score=dp[end])][:limit]


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
