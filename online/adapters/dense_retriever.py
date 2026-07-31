"""Dense semantic retriever composed from an encoder and vector-store port."""

from online.domain.models import Candidate, Modality, QueryPlan
from online.ports.interfaces import TextEncoder, VectorStore
from online.services.branch_options import effective_limit, effective_weight


class DenseRetriever:
    name = "dense_visual"

    def __init__(self, encoder: TextEncoder, store: VectorStore) -> None:
        self.encoder = encoder
        self.store = store

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.name, Modality.VISUAL) <= 0:
            return []
        vector = await self.encoder.encode(plan.normalized_query)
        return await self.store.search(
            vector, limit=effective_limit(plan, self.name, limit), filters=plan.filters
        )

