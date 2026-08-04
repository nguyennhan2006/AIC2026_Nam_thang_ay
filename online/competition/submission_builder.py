"""Dựng submission item + CSV từ kết quả task processor (PR-08).

Format CSV đã xác nhận ở `docs/12_USER_GUIDE.md` §6: không header, tối đa
`MAX_SUBMISSION_ITEMS` dòng, `frame_idx` là true frame index.

    KIS    <video_id>, <frame_id>
    QA     <video_id>, <frame_id>, "<answer>"
    TRAKE  <video_id>, <frame_id_1>, ..., <frame_id_n>
"""

from __future__ import annotations

import csv
import io

from online.domain.submission import KisSubmissionItem, QaSubmissionItem, TrakeSubmissionItem
from online.domain.task_results import KisResultItem, QaResultItem, TrakeResultItem
from online.competition.rules import MAX_SUBMISSION_ITEMS


def build_kis_submission(results: list[KisResultItem]) -> list[KisSubmissionItem]:
    ordered = sorted(results, key=lambda item: item.rank)
    return [
        KisSubmissionItem(video_id=item.video_id, frame_idx=item.frame_idx)
        for item in ordered[:MAX_SUBMISSION_ITEMS]
    ]


def build_qa_submission(results: list[QaResultItem]) -> list[QaSubmissionItem]:
    ordered = sorted(results, key=lambda item: item.rank)
    return [
        QaSubmissionItem(video_id=item.video_id, frame_idx=item.frame_idx, answer=item.answer)
        for item in ordered[:MAX_SUBMISSION_ITEMS]
    ]


def build_trake_submission(results: list[TrakeResultItem]) -> list[TrakeSubmissionItem]:
    ordered = sorted(results, key=lambda item: item.rank)
    return [
        TrakeSubmissionItem(video_id=item.video_id, frame_ids=item.frame_ids)
        for item in ordered[:MAX_SUBMISSION_ITEMS]
    ]


def _writer():
    # \r\n mặc định của csv module; BTC nhận CSV nên giữ theo chuẩn RFC4180.
    # quoting=QUOTE_MINIMAL để "<answer>" chỉ được bọc ngoặc kép khi cần
    # (chứa dấu phẩy/ngoặc kép/xuống dòng) — không phải mọi answer.
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    return buffer, writer


def kis_to_csv(items: list[KisSubmissionItem]) -> str:
    buffer, writer = _writer()
    for item in items:
        writer.writerow([item.video_id, item.frame_idx])
    return buffer.getvalue()


def qa_to_csv(items: list[QaSubmissionItem]) -> str:
    buffer, writer = _writer()
    for item in items:
        writer.writerow([item.video_id, item.frame_idx, item.answer])
    return buffer.getvalue()


def trake_to_csv(items: list[TrakeSubmissionItem]) -> str:
    buffer, writer = _writer()
    for item in items:
        writer.writerow([item.video_id, *item.frame_ids])
    return buffer.getvalue()


__all__ = [
    "build_kis_submission",
    "build_qa_submission",
    "build_trake_submission",
    "kis_to_csv",
    "qa_to_csv",
    "trake_to_csv",
]
