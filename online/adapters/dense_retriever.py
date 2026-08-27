"""Dense semantic retriever composed from an encoder and vector-store port."""

from online.domain.execution import BackendKind
from online.domain.models import Candidate, Modality, QueryPlan
from online.ports.interfaces import TextEncoder, VectorStore
from online.services.branch_options import effective_limit, effective_weight


class DenseRetriever:
    """Vector search qua một `VectorStore` port.

    `branch_id`/`backend_kind` là tham số chứ không cố định: backend local
    dùng `HashingTextEncoder` + `InMemoryVectorStore`, tức BM25/hash trên
    caption chứ KHÔNG phải dense visual thật. Container đăng ký nó dưới tên
    `lexical_hash_fallback` với `backend_kind="lexical_fallback"` để
    `/capabilities` không quảng cáo nhầm và số liệu ablation không bị sai.
    """

    supported_controls = ("enabled", "weight", "top_k", "timeout_ms")

    def __init__(
        self,
        encoder: TextEncoder,
        store: VectorStore,
        *,
        branch_id: str = "dense_visual",
        backend_kind: BackendKind = "vector",
        modality: Modality = Modality.VISUAL,
        query_variant: str = "raw",
    ) -> None:
        self.encoder = encoder
        self.store = store
        self.branch_id = branch_id
        self.execution_id = f"{branch_id}.{query_variant}"
        self.name = branch_id
        self.modality = modality
        self.backend_kind = backend_kind

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.execution_id, self.modality, self.branch_id) <= 0:
            return []

        # Query Routing V2: use specialized visual query if available
        query_text = plan.normalized_query
        if plan.visual_query:
            # Prefer visual query (entity-focused, no abstract questions)
            query_text = plan.visual_query

        vector = await self.encoder.encode(query_text)
        candidates = await self.store.search(
            vector,
            limit=effective_limit(plan, self.execution_id, limit, self.branch_id),
            filters=plan.filters,
        )
        # Vector store không biết nó đang phục vụ execution nào; gắn nhãn ở đây
        # để `source` của candidate luôn khớp id mà /capabilities công bố.
        return [item.model_copy(update={"source": self.execution_id}) for item in candidates]
