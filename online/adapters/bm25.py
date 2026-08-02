"""Small in-memory BM25 baseline used for local demos and regression tests."""

from __future__ import annotations

from collections import Counter
import math
import re

from online.domain.models import Candidate, Modality, QueryPlan, SceneDocument
from online.ports.interfaces import SceneRepository
from online.services.branch_options import effective_limit, effective_weight


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _document_id(document) -> str:
    """Stable tie-break id. BM25Index is generic over any document exposing
    `field_text()` — SceneDocument (`scene_id`) and EventDocument (`event_id`,
    see online/adapters/event_search.py) are both used with it today."""

    return getattr(document, "scene_id", None) or document.event_id


class BM25Index:
    def __init__(
        self,
        documents: list[SceneDocument],
        field: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.field = field
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(item.field_text(field)) for item in documents]
        self.lengths = [len(item) for item in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        frequencies: Counter[str] = Counter()
        for tokens in self.tokens:
            frequencies.update(set(tokens))
        count = len(documents)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in frequencies.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[SceneDocument, float]]:
        query_tokens = tokenize(query)
        scored: list[tuple[SceneDocument, float]] = []
        for document, tokens, length in zip(
            self.documents, self.tokens, self.lengths, strict=True
        ):
            tf = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = tf[token]
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_length, 1e-9)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((document, score))
        scored.sort(key=lambda item: (-item[1], _document_id(item[0])))
        return scored[:limit]


class LexicalRetriever:
    def __init__(self, field: str, index: BM25Index) -> None:
        self.field = field
        self.index = index
        self.name = f"bm25_{field}"
        self.modality = Modality(field)

    @classmethod
    async def build(
        cls, field: str, repository: SceneRepository
    ) -> "LexicalRetriever":
        return cls(field, BM25Index(await repository.all(), field))

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.name, self.modality) <= 0:
            return []
        limit = effective_limit(plan, self.name, limit)
        query = plan.events[0].text if len(plan.events) == 1 else plan.normalized_query
        results = self.index.search(query, limit * 2)
        filtered = []
        for scene, score in results:
            if plan.filters.video_ids and scene.video_id not in plan.filters.video_ids:
                continue
            if plan.filters.scene_ids and scene.scene_id not in plan.filters.scene_ids:
                continue
            if plan.filters.has_ocr is not None and bool(scene.ocr_texts) != plan.filters.has_ocr:
                continue
            if plan.filters.has_asr is not None and bool(scene.asr_texts) != plan.filters.has_asr:
                continue
            if (
                plan.filters.start_sec_gte is not None
                and scene.start_sec < plan.filters.start_sec_gte
            ):
                continue
            if (
                plan.filters.end_sec_lte is not None
                and scene.end_sec > plan.filters.end_sec_lte
            ):
                continue
            text = scene.field_text(self.field)
            filtered.append(
                Candidate(
                    candidate_id=scene.scene_id,
                    entity_type="scene",
                    scene_id=scene.scene_id,
                    video_id=scene.video_id,
                    start_frame=scene.start_frame,
                    end_frame=scene.end_frame_exclusive - 1,
                    source=self.name,
                    modality=self.modality,
                    raw_score=score,
                    score_kind="bm25",
                    rank=len(filtered) + 1,
                    index_id=f"bm25_{self.field}_inmemory",
                    payload={"matched_text": text[:1000]},
                )
            )
            if len(filtered) >= limit:
                break
        return filtered
