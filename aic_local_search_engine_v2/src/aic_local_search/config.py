from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(slots=True)
class EngineConfig:
    """Search and index settings kept in ``index_manifest.json``."""

    vector_backend: Literal["auto", "faiss", "numpy"] = "auto"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    rrf_k: int = 60
    lexical_weight: float = 1.0
    vector_weight: float = 1.0
    lexical_candidates: int = 100
    vector_candidates: int = 100
    semantic_weight: float = 1.35
    ocr_weight: float = 1.0
    speech_weight: float = 1.0
    tags_weight: float = 1.1
    event_weight: float = 0.9
    scene_vector_weight: float = 1.35
    frame_vector_weight: float = 1.0
    needs_review_penalty: float = 0.75
    exclude_invalid: bool = True
    keep_numpy_fallback: bool = False

    def validate(self) -> None:
        if self.vector_backend not in {"auto", "faiss", "numpy"}:
            raise ValueError(f"Unsupported vector backend: {self.vector_backend}")
        if self.hnsw_m <= 0 or self.hnsw_ef_construction <= 0 or self.hnsw_ef_search <= 0:
            raise ValueError("HNSW parameters must be positive")
        if self.rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        if self.lexical_candidates <= 0 or self.vector_candidates <= 0:
            raise ValueError("Candidate counts must be positive")
        if not 0.0 <= self.needs_review_penalty <= 1.0:
            raise ValueError("needs_review_penalty must be in [0, 1]")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "EngineConfig":
        known = cls.__dataclass_fields__
        return cls(**{key: val for key, val in value.items() if key in known})
