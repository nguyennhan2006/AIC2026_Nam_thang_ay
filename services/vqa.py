"""Retrieval-augmented VQA with an explicit answer-generator port."""

from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from online.domain.models import SearchRequest, TaskType, VQARequest, VQAResponse
from online.ports.interfaces import AnswerGenerator, SceneRepository
from online.services.search import SearchService


class EvidenceOnlyAnswerGenerator:
    """Safe baseline that exposes evidence instead of inventing an answer."""

    async def answer(self, question: str, contexts: list) -> tuple[str, float | None]:
        snippets: list[str] = []
        for scene in contexts:
            text = " ".join(scene.captions + scene.ocr_texts + scene.asr_texts).strip()
            if text:
                snippets.append(f"[{scene.scene_id}] {text[:500]}")
        if not snippets:
            return "Không tìm thấy đủ bằng chứng để trả lời.", 0.0
        return "Bằng chứng truy xuất được:\n" + "\n".join(snippets), None


class VQAService:
    def __init__(
        self,
        search_service: SearchService,
        repository: SceneRepository,
        answer_generator: AnswerGenerator | None = None,
    ) -> None:
        self.search_service = search_service
        self.repository = repository
        self.answer_generator = answer_generator or EvidenceOnlyAnswerGenerator()

    async def answer(self, request: VQARequest) -> VQAResponse:
        started = perf_counter()
        search_response = await self.search_service.search(
            SearchRequest(
                query=request.question,
                task=TaskType.VQA,
                top_k=request.top_k_evidence,
                filters=request.filters,
                debug=request.debug,
            )
        )
        contexts = await self.repository.get_many(
            [item.scene_id for item in search_response.results]
        )
        answer, confidence = await self.answer_generator.answer(request.question, contexts)
        return VQAResponse(
            query_id=str(uuid4()),
            answer=answer,
            confidence=confidence,
            evidence=search_response.results,
            took_ms=(perf_counter() - started) * 1000,
        )

