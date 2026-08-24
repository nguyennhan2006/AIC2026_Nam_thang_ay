"""Kho bản nháp bài nộp, ghi ra JSONL trên đĩa (FB-003).

Vì sao trên đĩa chứ không trong RAM: giá trị của bản nháp nằm ở chỗ nó sống
qua một lần restart server. Trong buổi thi, restart xảy ra (đổi env, sập,
nạp lại pack) và mất hết công soát của cả đội thì tính năng này vô nghĩa.

Vì sao JSONL chứ không SQLite: cùng định dạng với mọi thứ khác trong repo
(scenes/keyframes/videos), đọc được bằng mắt, và mở được bằng chính công cụ
đội đang dùng khi cần cứu dữ liệu bằng tay.

Ghi lại TOÀN BỘ file mỗi lần thay đổi (không append) vì cần sửa/xoá được bản
cũ. Với vài chục bản nháp thì chi phí không đáng kể, và tránh hẳn việc phải
nén file định kỳ.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from online.domain.drafts import DraftSaveRequest, SubmissionDraft


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JsonlDraftStore:
    """Kho nháp dùng chung. An toàn với nhiều request cùng lúc trong MỘT tiến trình.

    Khoá là `asyncio.Lock`, tức là chỉ bảo vệ trong phạm vi một tiến trình —
    đủ vì server chạy một worker (ma trận vector 700 MB không cho phép chạy
    nhiều worker). Chạy nhiều tiến trình cùng trỏ vào một file thì hai lần lưu
    sát nhau có thể đè nhau; ghi bằng file tạm + `os.replace` nên ít nhất file
    không bao giờ ở trạng thái vỡ dở.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    # ---- đĩa ------------------------------------------------------------
    def _read_sync(self) -> list[SubmissionDraft]:
        if not self.path.exists():
            return []
        drafts: list[SubmissionDraft] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    drafts.append(SubmissionDraft.model_validate_json(line))
                except Exception:  # noqa: BLE001
                    # Một dòng hỏng KHÔNG được làm mất những bản còn lại: đây
                    # là công soát tay của người khác, thà bỏ một dòng còn hơn
                    # trả về danh sách rỗng và làm cả đội tưởng mất sạch.
                    continue
        return drafts

    def _write_sync(self, drafts: list[SubmissionDraft]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for draft in drafts:
                handle.write(draft.model_dump_json() + "\n")
        os.replace(temporary, self.path)

    # ---- API ------------------------------------------------------------
    async def list(self) -> list[SubmissionDraft]:
        """Mới nhất trước — người soát quan tâm bản vừa lưu, không phải bản đầu."""

        async with self._lock:
            drafts = await asyncio.to_thread(self._read_sync)
        # File giữ nguyên thứ tự ghi (bản lưu sau nằm cuối). Đảo trước rồi mới
        # sort: `sorted` ổn định, nên hai bản lưu trong CÙNG một mốc thời gian
        # vẫn ra đúng thứ tự lưu thay vì tuỳ ý — chuyện xảy ra thật khi hai
        # người bấm Lưu cách nhau vài chục mili giây.
        return sorted(reversed(drafts), key=lambda draft: draft.updated_at, reverse=True)

    async def get(self, draft_id: str) -> SubmissionDraft | None:
        for draft in await self.list():
            if draft.draft_id == draft_id:
                return draft
        return None

    async def save(self, request: DraftSaveRequest) -> SubmissionDraft:
        async with self._lock:
            drafts = await asyncio.to_thread(self._read_sync)
            existing = next(
                (item for item in drafts if item.draft_id == request.draft_id), None
            ) if request.draft_id else None
            draft = SubmissionDraft(
                draft_id=existing.draft_id if existing else uuid4().hex[:12],
                name=request.name,
                author=request.author,
                task=request.task,
                query=request.query,
                rows=request.rows,
                created_at=existing.created_at if existing else _now(),
                updated_at=_now(),
            )
            remaining = [item for item in drafts if item.draft_id != draft.draft_id]
            await asyncio.to_thread(self._write_sync, [*remaining, draft])
        return draft

    async def delete(self, draft_id: str) -> bool:
        async with self._lock:
            drafts = await asyncio.to_thread(self._read_sync)
            remaining = [item for item in drafts if item.draft_id != draft_id]
            if len(remaining) == len(drafts):
                return False
            await asyncio.to_thread(self._write_sync, remaining)
        return True


__all__ = ["JsonlDraftStore"]
