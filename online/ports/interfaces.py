"""Dependency-inversion ports for infrastructure and model providers."""

from __future__ import annotations

from typing import Protocol, Sequence

from online.domain.models import Candidate, QueryPlan, SceneDocument, SearchFilters
from online.domain.session import SearchExecutionTrace


class SceneRepository(Protocol):
    async def get(self, scene_id: str) -> SceneDocument | None: ...
    async def get_many(self, scene_ids: Sequence[str]) -> list[SceneDocument]: ...
    async def all(self) -> list[SceneDocument]: ...
    async def video_frame_count(self, video_id: str) -> int | None: ...


class TextEncoder(Protocol):
    async def encode(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    async def health(self) -> bool: ...
    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: SearchFilters,
    ) -> list[Candidate]: ...


class Retriever(Protocol):
    name: str
    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: list[Candidate]
    ) -> list[Candidate]: ...


class AnswerGenerator(Protocol):
    async def answer(self, question: str, contexts: list[SceneDocument]) -> tuple[str, float | None]: ...


class SessionStore(Protocol):
    """Lưu trace một lần search để replay/audit (PR-09)."""

    async def put(self, trace: SearchExecutionTrace) -> None: ...
    async def get(self, session_id: str) -> SearchExecutionTrace | None: ...
