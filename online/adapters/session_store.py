"""Lưu trữ search session — mặc định in-memory (PR-09).

In-memory là baseline an toàn cho một tiến trình backend duy nhất (đúng
topology hiện tại — xem `docs/11_SERVER_IMPLEMENTATION.md`, backend là một
FastAPI process). Có giới hạn kích thước (LRU-ish theo thời gian chèn) để
không phình vô hạn trong một phiên thi kéo dài nhiều giờ. Khi cần chia sẻ
giữa nhiều worker/replica thì thay bằng Redis — cùng interface `SessionStore`,
không đổi chỗ gọi.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from online.domain.session import SearchExecutionTrace

# Protocol (dependency-inversion port) sống ở online.ports.interfaces.SessionStore,
# cùng chỗ với SceneRepository/VectorStore/... — đây chỉ là implementation.


class InMemorySessionStore:
    """OrderedDict + khóa async; đơn giản, đủ cho một process backend."""

    def __init__(self, max_size: int = 2000) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._traces: OrderedDict[str, SearchExecutionTrace] = OrderedDict()
        self._lock = asyncio.Lock()

    async def put(self, trace: SearchExecutionTrace) -> None:
        async with self._lock:
            self._traces[trace.session_id] = trace
            self._traces.move_to_end(trace.session_id)
            while len(self._traces) > self.max_size:
                self._traces.popitem(last=False)

    async def get(self, session_id: str) -> SearchExecutionTrace | None:
        async with self._lock:
            return self._traces.get(session_id)

    async def update(self, session_id: str, **changes: object) -> SearchExecutionTrace | None:
        async with self._lock:
            trace = self._traces.get(session_id)
            if trace is None:
                return None
            updated = trace.model_copy(update=changes)
            self._traces[session_id] = updated
            return updated


__all__ = ["InMemorySessionStore"]
