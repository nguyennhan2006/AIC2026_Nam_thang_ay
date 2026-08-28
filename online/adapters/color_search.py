"""Color search branch — Search Mixing Console W3 (`color_search`).

Baseline: match a curated Vietnamese/English color-name lexicon against the
query, then score a scene by what fraction of the query's color tags are
present in `SceneDocument.color_names` (aggregated from
`Keyframe.color.dominant_colors`, see offline/gpu_engine.py's hue-bucket
naming — the canonical names this lexicon maps to: red/orange/yellow/green/
cyan/blue/purple/pink/black/white/gray).

No color-region reasoning (query "áo đỏ" vs "nền đỏ" are indistinguishable
today) — see docs/15_RESEARCH_AGENDA.md before assuming this is enough.
"""

from __future__ import annotations

import asyncio

import re

from online.domain.models import Candidate, Modality, QueryPlan, SceneDocument
from online.ports.interfaces import SceneRepository
from online.services.branch_options import effective_limit, effective_weight
from online.services.branch_query import get_branch_query

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+")

COLOR_LEXICON: dict[str, list[str]] = {
    "red": ["đỏ", "red"],
    "orange": ["cam", "orange"],
    "yellow": ["vàng", "yellow"],
    "green": ["xanh lá", "xanh lục", "green"],
    "cyan": ["xanh ngọc", "cyan", "turquoise"],
    "blue": ["xanh dương", "xanh biển", "xanh da trời", "blue"],
    "purple": ["tím", "purple", "violet"],
    "pink": ["hồng", "pink"],
    "black": ["đen", "black"],
    "white": ["trắng", "white"],
    "gray": ["xám", "gray", "grey"],
}


def extract_color_tags(query: str) -> list[str]:
    """Return the canonical color names mentioned in `query`, sorted+deduped."""

    if not query:
        return []
    text = query.casefold()
    tokens = set(TOKEN_RE.findall(text))
    tags: set[str] = set()
    for name, surface_forms in COLOR_LEXICON.items():
        for form in surface_forms:
            form_cf = form.casefold()
            matched = form_cf in tokens if " " not in form_cf else form_cf in text
            if matched:
                tags.add(name)
                break
    return sorted(tags)


class ColorSearchRetriever:
    """Exact color-tag overlap between the query and a scene's dominant colors."""

    branch_id = "color_search"
    execution_id = "color_search.raw"
    name = branch_id
    modality = Modality.COLOR
    backend_kind = "metadata"
    supported_controls = ("enabled", "weight", "top_k", "timeout_ms")

    def __init__(self, documents: list[SceneDocument]) -> None:
        self.documents = [doc for doc in documents if doc.color_names]

    @classmethod
    async def build(cls, repository: SceneRepository) -> "ColorSearchRetriever":
        return cls(await repository.all())

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.execution_id, self.modality, self.branch_id) <= 0:
            return []
        limit = effective_limit(plan, self.execution_id, limit, self.branch_id)
        query = get_branch_query(
            plan,
            self.branch_id,
            self.modality,
            plan.normalized_query,
            execution_id=self.execution_id,
        )
        if query is None:
            return []
        query_tags = extract_color_tags(query)
        if not query_tags:
            return []
        # Quét toàn corpus là CPU-bound thuần: chạy thẳng trên event loop thì
        # cả server đứng hình suốt quãng đó — không /v1/health, không phục vụ
        # ảnh keyframe, không nhận nổi truy vấn của người thứ hai. Tiền lệ đã
        # đo được ghi ngay trong module này: `ocr_fuzzy` chạy đồng bộ kéo
        # `dense_visual` từ p50 224ms lên 8,7s và timeout 40/84 truy vấn.
        return await asyncio.to_thread(self._search_sync, plan, limit, query_tags)

    def _search_sync(
        self, plan: QueryPlan, limit: int, query_tags: list[str]
    ) -> list[Candidate]:
        query_set = set(query_tags)
        scored: list[tuple[float, SceneDocument]] = []
        for doc in self.documents:
            if plan.filters.video_ids and doc.video_id not in plan.filters.video_ids:
                continue
            if plan.filters.scene_ids and doc.scene_id not in plan.filters.scene_ids:
                continue
            matched = query_set & set(doc.color_names)
            if matched:
                scored.append((len(matched) / len(query_set), doc))
        scored.sort(key=lambda item: (-item[0], item[1].scene_id))
        return [
            Candidate(
                candidate_id=doc.scene_id, entity_type="scene", scene_id=doc.scene_id,
                video_id=doc.video_id, start_frame=doc.start_frame,
                end_frame=doc.end_frame_exclusive - 1,
                source=self.execution_id, modality=self.modality,
                raw_score=score, score_kind="overlap_ratio", rank=rank,
                payload={"matched_colors": sorted(query_set & set(doc.color_names))},
            )
            for rank, (score, doc) in enumerate(scored[:limit], start=1)
        ]


__all__ = ["COLOR_LEXICON", "ColorSearchRetriever", "extract_color_tags"]
