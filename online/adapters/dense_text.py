"""Nhánh retrieval dense trên TEXT của caption/tag — DENSE-TEXT-01.

Khác `DenseRetriever` (online/adapters/dense_retriever.py): nhánh đó khớp
truy vấn với embedding **ảnh** (CLIP/SigLIP). Nhánh này khớp truy vấn với
embedding **văn bản** của caption/object/action đã trích offline. Hai nhánh
bổ sung cho nhau: CLIP mạnh ở hình thức thị giác, dense text mạnh ở diễn đạt
khác từ nhưng cùng nghĩa.

E5 dùng prefix bất đối xứng (`query: ` / `passage: `) — thiếu prefix là lỗi
IM LẶNG: model vẫn chạy, vẫn trả số, chỉ kém hẳn. Prefix đọc từ manifest của
index để encoder online và offline không bao giờ lệch nhau.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from online.domain.models import Candidate, Modality, QueryPlan
from online.services.branch_options import effective_limit, effective_weight


class CaptionDenseRetriever:
    backend_kind = "vector"
    branch_id = "caption_dense"
    execution_id = "caption_dense.raw"
    name = branch_id
    modality = Modality.CAPTION

    def __init__(self, index_dir: Path, encoder) -> None:
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"thiếu {manifest_path} — chạy scripts/build_caption_dense_index.py trước"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.matrix: np.ndarray = np.load(self.index_dir / "embeddings.npy")
        self.scene_ids: list[str] = json.loads(
            (self.index_dir / "scene_ids.json").read_text(encoding="utf-8")
        )
        if len(self.scene_ids) != self.matrix.shape[0]:
            raise ValueError(
                f"index hỏng: {len(self.scene_ids)} scene_id nhưng {self.matrix.shape[0]} vector"
            )
        self.encoder = encoder
        self.query_prefix: str = self.manifest.get("query_prefix", "query: ")
        self.model_id: str = self.manifest.get("model_id", "unknown")
        self.index_id: str = self.manifest.get("index_fingerprint", "unknown")

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.execution_id, self.modality, self.branch_id) <= 0:
            return []
        limit = effective_limit(plan, self.execution_id, limit, self.branch_id)
        query = self.query_prefix + plan.normalized_query
        vector = self.encoder.encode([query])[0]

        # Vector đã L2-normalize cả hai phía nên dot product = cosine.
        scores = self.matrix @ vector
        top = np.argsort(-scores)[:limit]
        return [
            Candidate(
                candidate_id=f"{self.execution_id}:{self.scene_ids[position]}",
                video_id=self.scene_ids[position].rsplit("_S", 1)[0],
                scene_id=self.scene_ids[position],
                source=self.execution_id,
                modality=self.modality,
                raw_score=float(scores[position]),
                score_kind="cosine",
                rank=rank,
                model_id=self.model_id,
                index_id=self.index_id,
            )
            for rank, position in enumerate(top, start=1)
        ]


__all__ = ["CaptionDenseRetriever"]
