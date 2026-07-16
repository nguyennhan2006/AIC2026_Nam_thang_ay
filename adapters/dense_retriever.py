"""Dense semantic retriever composed from an encoder and vector-store port."""

from online.domain.models import Candidate, Modality, QueryPlan
from online.ports.interfaces import TextEncoder, VectorStore


class DenseRetriever:
    name = "dense_visual"

    def __init__(self, encoder: TextEncoder, store: VectorStore) -> None:
        self.encoder = encoder
        self.store = store

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if plan.modality_weights.get(Modality.VISUAL, 0) <= 0:
            return []
        vector = await self.encoder.encode(plan.normalized_query)
        return await self.store.search(vector, limit=limit, filters=plan.filters)

