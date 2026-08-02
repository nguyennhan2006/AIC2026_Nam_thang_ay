"""Chấm điểm submission cục bộ theo đúng luật thi (PR-08).

Tách khỏi `scripts/eval_tasks.py` (chấm trên `SearchResponse` để đo chất
lượng retrieval trong lúc phát triển) — module này chấm trên *submission đã
build*, để phục vụ `POST /v1/submissions/evaluate-local`: người dùng dán vào
một CSV/JSON gold nhỏ và xem trước điểm trước khi nộp thật.

Luật (docs/09_RESEARCH_ALIGNMENT.md, xác nhận lại ở
`examples/AIC2026_L21_V001_query_schema.json`)::

    KIS    đúng video AND frame nằm trong interval  -> 1.0, ngược lại 0.0
    QA     đúng video AND frame trong interval AND answer đúng -> 1.0
    TRAKE  sai video -> 0.0; đúng video -> mean(step frame rơi đúng cửa sổ GT)
"""

from __future__ import annotations

from dataclasses import dataclass

from online.domain.submission import KisSubmissionItem, QaSubmissionItem, TrakeSubmissionItem
from online.services.qa import answer_matches


@dataclass(frozen=True, slots=True)
class GoldInterval:
    start_frame: int
    end_frame: int  # inclusive

    def contains(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx <= self.end_frame


@dataclass(frozen=True, slots=True)
class KisGold:
    video_id: str
    intervals: tuple[GoldInterval, ...]


@dataclass(frozen=True, slots=True)
class QaGold:
    video_id: str
    intervals: tuple[GoldInterval, ...]
    accepted_answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrakeGold:
    video_id: str
    step_windows: tuple[GoldInterval, ...]


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    best_rank: int | None
    detail: str


def score_kis(items: list[KisSubmissionItem], gold: KisGold) -> ScoreResult:
    for rank, item in enumerate(items, start=1):
        if item.video_id != gold.video_id:
            continue
        if any(interval.contains(item.frame_idx) for interval in gold.intervals):
            return ScoreResult(1.0, rank, f"hit tại rank {rank}")
    return ScoreResult(0.0, None, "không có dòng nào đúng video + trong interval")


def score_qa(items: list[QaSubmissionItem], gold: QaGold) -> ScoreResult:
    for rank, item in enumerate(items, start=1):
        if item.video_id != gold.video_id:
            continue
        if not any(interval.contains(item.frame_idx) for interval in gold.intervals):
            continue
        if answer_matches(item.answer, gold.accepted_answers):
            return ScoreResult(1.0, rank, f"hit đủ ba điều kiện tại rank {rank}")
    return ScoreResult(0.0, None, "không có dòng nào đúng cả video + frame + answer")


def score_trake(item: TrakeSubmissionItem, gold: TrakeGold) -> ScoreResult:
    """Chấm MỘT dòng TRAKE (thường là dòng rank 1) — luật thi chấm mean R-Score.

    Không lặp qua danh sách như KIS/QA: TRAKE không có khái niệm "dòng nào
    trong top-K đúng", nộp bài chỉ có một chuỗi được chấm thật.
    """

    if item.video_id != gold.video_id:
        return ScoreResult(0.0, None, "sai video — theo luật, mean R-Score = 0")
    hits = sum(
        1
        for frame_idx, window in zip(item.frame_ids, gold.step_windows, strict=False)
        if window.contains(frame_idx)
    )
    total = len(gold.step_windows) or 1
    r_score = hits / total
    return ScoreResult(r_score, 1, f"{hits}/{total} step đúng cửa sổ GT")


__all__ = [
    "GoldInterval",
    "KisGold",
    "QaGold",
    "ScoreResult",
    "TrakeGold",
    "score_kis",
    "score_qa",
    "score_trake",
]
