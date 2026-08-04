"""Chạy các branch song song, có deadline và trạng thái (PR-03).

Thay cho `asyncio.gather(*(item.search(...) for item in retrievers))` trần
trong `SearchService._retrieve`. Ba khác biệt:

1. Mỗi branch có deadline riêng (`BranchRuntimeOptions.timeout_ms`).
2. Exception được bắt tại chính branch đó và chuyển thành `BranchStatus` có
   kiểu, nên một branch chết không kéo đổ cả request.
3. Mọi branch đều xuất hiện trong `branch_status`, kể cả khi bị tắt hoặc
   không trả kết quả — UI thấy được, không phải đoán.
"""

from __future__ import annotations

import asyncio
from time import perf_counter
from typing import AsyncIterator

from online.domain.candidate import Candidate
from online.domain.execution import BranchStatus
from online.domain.models import QueryPlan
from online.errors import DependencyUnavailableError

DEFAULT_TIMEOUT_MS = 3000


def _branch_identity(retriever: object) -> tuple[str, str]:
    """Trả `(branch_id, execution_id)` của một retriever.

    Retriever cũ chỉ có `.name`; coi đó vừa là branch_id vừa là execution_id
    để adapter chưa migrate vẫn chạy.
    """

    branch_id = getattr(retriever, "branch_id", None) or getattr(retriever, "name", "unknown")
    execution_id = getattr(retriever, "execution_id", None) or branch_id
    return str(branch_id), str(execution_id)


def resolve_timeout_ms(plan: QueryPlan, execution_id: str, branch_id: str) -> int:
    override = plan.search_options.branches.get(execution_id) or plan.search_options.branches.get(
        branch_id
    )
    return override.timeout_ms if override is not None else DEFAULT_TIMEOUT_MS


class RetrievalOrchestrator:
    """Chạy toàn bộ retriever cho một plan và thu thập trạng thái."""

    def __init__(self, retrievers: list, *, default_timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.retrievers = retrievers
        self.default_timeout_ms = default_timeout_ms

    async def _run_one(
        self, retriever, plan: QueryPlan, limit: int
    ) -> tuple[list[Candidate], BranchStatus]:
        branch_id, execution_id = _branch_identity(retriever)
        timeout_ms = resolve_timeout_ms(plan, execution_id, branch_id) or self.default_timeout_ms
        started = perf_counter()

        def elapsed() -> int:
            return int((perf_counter() - started) * 1000)

        try:
            candidates = await asyncio.wait_for(
                retriever.search(plan, limit=limit), timeout=timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            return [], BranchStatus(
                execution_id=execution_id, branch_id=branch_id, state="timeout",
                latency_ms=elapsed(),
                warning=f"branch vượt quá deadline {timeout_ms}ms và đã bị bỏ qua",
            )
        except DependencyUnavailableError as exc:
            return [], BranchStatus(
                execution_id=execution_id, branch_id=branch_id, state="unavailable",
                latency_ms=elapsed(), warning=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 - phải giữ request sống
            return [], BranchStatus(
                execution_id=execution_id, branch_id=branch_id, state="failed",
                latency_ms=elapsed(), warning=f"{type(exc).__name__}: {exc}",
            )

        if not candidates:
            # Rỗng có hai nghĩa rất khác nhau: branch bị tắt (weight 0) hay
            # branch chạy nhưng không khớp gì. Tách ra để đọc log không nhầm.
            override = plan.search_options.branches.get(execution_id) or (
                plan.search_options.branches.get(branch_id)
            )
            disabled = override is not None and not override.enabled
            return [], BranchStatus(
                execution_id=execution_id, branch_id=branch_id,
                state="disabled" if disabled else "empty", latency_ms=elapsed(),
            )
        return candidates, BranchStatus(
            execution_id=execution_id, branch_id=branch_id, state="success",
            latency_ms=elapsed(), candidate_count=len(candidates),
        )

    async def execute(
        self, plan: QueryPlan, limit: int
    ) -> tuple[list[list[Candidate]], list[BranchStatus]]:
        """Chạy mọi branch; ném lỗi chỉ khi KHÔNG branch nào chạy nổi."""

        outcomes = await asyncio.gather(
            *(self._run_one(retriever, plan, limit) for retriever in self.retrievers)
        )
        lists = [candidates for candidates, _status in outcomes]
        statuses = [status for _candidates, status in outcomes]
        self._raise_if_all_degraded(statuses)
        return lists, statuses

    async def stream(
        self, plan: QueryPlan, limit: int
    ) -> AsyncIterator[tuple[list[Candidate], BranchStatus]]:
        """Như `execute`, nhưng yield từng branch NGAY khi nó xong (PR-09).

        Dùng cho `/v1/search/stream`: UI thấy `branch_completed`/`branch_failed`
        thật theo thời gian branch đó chạy xong, không phải một loạt sự kiện
        giả lập sau khi mọi thứ đã xong từ lâu. Thứ tự yield không xác định
        trước (branch nhanh xong trước), khác với `execute` luôn trả theo thứ
        tự `self.retrievers`.
        """

        tasks = [
            asyncio.ensure_future(self._run_one(retriever, plan, limit))
            for retriever in self.retrievers
        ]
        statuses: list[BranchStatus] = []
        for finished in asyncio.as_completed(tasks):
            candidates, status = await finished
            statuses.append(status)
            yield candidates, status
        self._raise_if_all_degraded(statuses)

    @staticmethod
    def _raise_if_all_degraded(statuses: list[BranchStatus]) -> None:
        if all(status.is_degraded for status in statuses):
            detail = "; ".join(f"{status.execution_id}={status.state}" for status in statuses)
            raise DependencyUnavailableError(
                f"mọi retrieval branch đều hỏng ({detail}) — không có kết quả nào đáng tin"
            )


__all__ = ["DEFAULT_TIMEOUT_MS", "RetrievalOrchestrator", "resolve_timeout_ms"]
