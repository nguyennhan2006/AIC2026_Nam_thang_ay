"""Frame embedding retrieval for scene/clip pooling.

The current implementation retrieves embeddings by calling the inference
provider's ``embedding`` task on every read. ``MemoizedEmbeddingReader`` only
caches vectors for the lifetime of one pipeline run (one ``process_video``
call) so that overlapping clip windows do not re-encode the same keyframe —
it is not a persistent cache. Persistent frame-level embedding storage
(populating ``Keyframe.embedding_refs``) is tracked as tech debt, see
docs/14_TECHNICAL_PREPARATION.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from datasection.schemas import Keyframe


class EmbeddingReader(Protocol):
    async def read(self, keyframe: Keyframe) -> list[float]: ...


class ProviderEmbeddingReader:
    """Reads a keyframe's visual embedding through the inference provider."""

    def __init__(self, provider, data_root: Path) -> None:
        self._provider = provider
        self._data_root = data_root

    async def read(self, keyframe: Keyframe) -> list[float]:
        response = await self._provider.image("embedding", self._data_root / keyframe.image_path)
        return [float(value) for value in response["vector"]]


class MemoizedEmbeddingReader:
    """Avoids re-encoding the same keyframe across overlapping clip windows."""

    def __init__(self, inner: EmbeddingReader) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    async def read(self, keyframe: Keyframe) -> list[float]:
        cached = self._cache.get(keyframe.keyframe_id)
        if cached is not None:
            return cached
        vector = await self._inner.read(keyframe)
        self._cache[keyframe.keyframe_id] = vector
        return vector


__all__ = ["EmbeddingReader", "ProviderEmbeddingReader", "MemoizedEmbeddingReader"]
