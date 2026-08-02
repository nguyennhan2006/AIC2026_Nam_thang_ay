"""Gom và khử trùng lặp candidate sau fusion (PR-05).

Trước PR-05 việc "dedup" chỉ là hệ quả phụ của việc `fuse_candidates` gom
theo `scene_id`; `dedup_scope`/`dedup_similarity`/`max_results_per_video`
không có consumer nào. Hệ quả thực tế: top-100 chứa hàng chục scene liền kề
trong cùng một sự kiện, ăn hết các mốc chấm điểm 1/5/20 mà không thêm được
thông tin gì.

Chính sách theo task (khác nhau thật, không phải một hàm dùng chung):

* ``TEXTUAL_KIS`` — dedup theo event nhưng vẫn giữ vài video thay thế, vì
  sai video là mất trắng.
* ``QA`` — KHÔNG dedup mạnh: nhiều frame trong cùng event có thể là bằng
  chứng khác nhau cho cùng câu trả lời.
* ``TRAKE`` — không dedup ở đây; mỗi step là một truy vấn riêng và hai step
  hoàn toàn có thể ở cùng một scene.
* ``AVS`` — dedup event mạnh nhất, vì điểm phụ thuộc độ đa dạng.
"""

from __future__ import annotations

from online.domain.candidate import Candidate
from online.domain.tasks import TaskType

DedupScope = str


def _window_key(candidate: Candidate, window_sec: float) -> object:
    if candidate.timestamp_sec is None:
        return candidate.scene_id or candidate.candidate_id
    return int(candidate.timestamp_sec // window_sec)


def _group_key(candidate: Candidate, scope: DedupScope, window_sec: float) -> object:
    """Khóa gom theo scope. Trả về khóa duy nhất khi không gom được.

    Candidate thiếu thông tin của scope (vd không có `event_id`) KHÔNG bị gom
    bừa vào một nhóm chung — nếu không, mọi candidate chưa gán event sẽ bị
    dồn thành một và chỉ còn lại đúng một kết quả.
    """

    if scope == "frame":
        if candidate.frame_idx is None:
            return ("candidate", candidate.candidate_id)
        return ("frame", candidate.video_id, candidate.frame_idx)
    if scope == "scene":
        return ("scene", candidate.scene_id or candidate.candidate_id)
    if scope == "event":
        if not candidate.event_id:
            return ("scene", candidate.scene_id or candidate.candidate_id)
        return ("event", candidate.video_id, candidate.event_id)
    if scope == "video_window":
        return ("window", candidate.video_id, _window_key(candidate, window_sec))
    return ("candidate", candidate.candidate_id)


def deduplicate(
    candidates: list[Candidate],
    *,
    scope: DedupScope = "scene",
    max_per_video: int | None = None,
    max_per_event: int | None = None,
    window_sec: float = 5.0,
) -> list[Candidate]:
    """Giữ candidate tốt nhất mỗi nhóm, tôn trọng trần theo video/event.

    Đầu vào phải đã sắp theo điểm giảm dần (fusion đảm bảo điều đó), vì hàm
    này giữ phần tử ĐẦU TIÊN của mỗi nhóm và chỉ ghi lại dấu vết những cái bị
    gộp — không tự sắp lại để khỏi phá thứ tự mà rerank phía trước đã tạo ra.
    """

    if scope == "none" or not candidates:
        return candidates

    kept: list[Candidate] = []
    seen: dict[object, int] = {}
    absorbed: dict[int, list[str]] = {}
    per_video: dict[str, int] = {}
    per_event: dict[tuple[str, str], int] = {}

    for candidate in candidates:
        key = _group_key(candidate, scope, window_sec)
        index = seen.get(key)
        if index is not None:
            absorbed.setdefault(index, []).append(candidate.candidate_id)
            continue
        if max_per_video is not None and per_video.get(candidate.video_id, 0) >= max_per_video:
            continue
        event_key = (candidate.video_id, candidate.event_id or "")
        if (
            max_per_event is not None
            and candidate.event_id
            and per_event.get(event_key, 0) >= max_per_event
        ):
            continue
        seen[key] = len(kept)
        per_video[candidate.video_id] = per_video.get(candidate.video_id, 0) + 1
        if candidate.event_id:
            per_event[event_key] = per_event.get(event_key, 0) + 1
        kept.append(candidate)

    output: list[Candidate] = []
    for rank, candidate in enumerate(kept, start=1):
        merged = absorbed.get(rank - 1)
        payload = dict(candidate.payload)
        if merged:
            # Giữ dấu vết cái gì bị gộp: nếu không, "vì sao scene này biến mất"
            # trở thành câu hỏi không trả lời được lúc debug.
            payload["absorbed_candidates"] = merged
        output.append(candidate.model_copy(update={"rank": rank, "payload": payload}))
    return output


# Chính sách mặc định theo task. `max_per_video=None` nghĩa là không giới hạn.
TASK_POLICIES: dict[TaskType, dict[str, object]] = {
    TaskType.TEXTUAL_KIS: {"scope": "event", "max_per_video": 5, "max_per_event": 1},
    TaskType.QA: {"scope": "scene", "max_per_video": None, "max_per_event": 3},
    TaskType.TRAKE: {"scope": "none", "max_per_video": None, "max_per_event": None},
    TaskType.AVS: {"scope": "event", "max_per_video": 3, "max_per_event": 1},
}


def deduplicate_for_task(
    candidates: list[Candidate],
    task: TaskType,
    *,
    scope_override: str | None = None,
    max_per_video_override: int | None = None,
) -> list[Candidate]:
    """Áp chính sách của task, cho phép request ghi đè scope/trần video."""

    policy = dict(TASK_POLICIES[task])
    if scope_override is not None:
        policy["scope"] = scope_override
    if max_per_video_override is not None:
        policy["max_per_video"] = max_per_video_override
    return deduplicate(
        candidates,
        scope=str(policy["scope"]),
        max_per_video=policy["max_per_video"],  # type: ignore[arg-type]
        max_per_event=policy["max_per_event"],  # type: ignore[arg-type]
    )


__all__ = ["TASK_POLICIES", "deduplicate", "deduplicate_for_task"]
