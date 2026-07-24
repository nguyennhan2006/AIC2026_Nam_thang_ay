from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import EngineConfig


def _try_import_faiss() -> Any | None:
    try:
        import faiss

        return faiss
    except ImportError:
        return None


def normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("Vector contains NaN, infinity, or has zero norm")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def build_vector_index(
    index_dir: Path,
    embeddings: np.ndarray,
    config: EngineConfig,
    name: str,
) -> dict:
    config.validate()
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = normalize_vectors(embeddings)
    numpy_path = index_dir / f"{name}_embeddings.npy"

    faiss = _try_import_faiss()
    requested = config.vector_backend
    backend = "numpy"
    faiss_path = index_dir / f"{name}_hnsw.faiss"
    if requested == "faiss" and faiss is None:
        raise RuntimeError(
            "vector_backend='faiss' was requested but faiss is not installed. "
            "Install faiss-cpu with Conda or use vector_backend='numpy'."
        )
    if faiss is not None and requested in {"auto", "faiss"}:
        dimension = int(embeddings.shape[1])
        index = faiss.IndexHNSWFlat(dimension, config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.hnsw_ef_construction
        index.hnsw.efSearch = config.hnsw_ef_search
        index.add(embeddings)
        faiss.write_index(index, str(faiss_path))
        backend = "faiss_hnsw"
    elif faiss_path.exists():
        faiss_path.unlink()

    save_numpy = backend == "numpy" or config.keep_numpy_fallback
    if save_numpy:
        np.save(numpy_path, embeddings)
    elif numpy_path.exists():
        numpy_path.unlink()

    return {
        "name": name,
        "backend": backend,
        "count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "metric": "cosine_via_normalized_inner_product",
        "numpy_file": numpy_path.name if save_numpy else None,
        "faiss_file": faiss_path.name if backend == "faiss_hnsw" else None,
        "hnsw": {
            "m": config.hnsw_m,
            "ef_construction": config.hnsw_ef_construction,
            "ef_search": config.hnsw_ef_search,
        },
    }


class VectorIndex:
    def __init__(self, index_dir: Path, manifest: dict):
        self.index_dir = index_dir
        self.manifest = manifest
        self.dimension = int(manifest["dimension"])
        self.backend = str(manifest["backend"])
        self._faiss = None
        self._index = None
        self._matrix = None

        if self.backend == "faiss_hnsw":
            self._faiss = _try_import_faiss()
            if self._faiss is not None:
                self._index = self._faiss.read_index(str(index_dir / manifest["faiss_file"]))
        if self._index is None:
            self.backend = "numpy"
            if not manifest.get("numpy_file"):
                raise RuntimeError(
                    f"FAISS index {manifest.get('faiss_file')} exists but faiss is not installed, "
                    "and no Numpy fallback was retained."
                )
            self._matrix = np.load(index_dir / manifest["numpy_file"], mmap_mode="r")

    @classmethod
    def from_directory(cls, index_dir: Path, name: str = "scene") -> "VectorIndex":
        manifest = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        return cls(index_dir, manifest[f"{name}_vector_index"])

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        query = normalize_vectors(np.asarray(query_vector, dtype=np.float32))
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query.shape[1]} != index dimension {self.dimension}"
            )
        k = max(0, min(int(k), int(self.manifest["count"])))
        if k == 0:
            return []
        if self._index is not None:
            scores, indices = self._index.search(query, k)
            return [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0])
                if int(index) >= 0
            ]

        assert self._matrix is not None
        scores = np.asarray(self._matrix @ query[0], dtype=np.float32)
        if k == len(scores):
            indices = np.argsort(-scores)
        else:
            candidates = np.argpartition(-scores, k - 1)[:k]
            indices = candidates[np.argsort(-scores[candidates])]
        return [(int(index), float(scores[index])) for index in indices]


class OpenClipTextEncoder:
    """Lazy query encoder matching ``open_clip:ViT-B-32:openai`` embeddings."""

    def __init__(self, model_spec: str, device: str | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Visual text search needs PyTorch and open_clip_torch. "
                "Install a CUDA-enabled PyTorch build, then `pip install open_clip_torch`."
            ) from exc

        parts = model_spec.split(":")
        if len(parts) != 3 or parts[0] != "open_clip":
            raise ValueError(f"Unsupported visual embedding model: {model_spec}")
        _, architecture, pretrained = parts
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(
            architecture, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(architecture)

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with self.torch.inference_mode():
            vector = self.model.encode_text(tokens)
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.detach().cpu().numpy()[0].astype(np.float32)
