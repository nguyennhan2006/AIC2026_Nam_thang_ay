"""Deterministic mapping between business IDs and storage-specific IDs."""

from uuid import NAMESPACE_URL, uuid5


def qdrant_point_id(entity_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"aic2026:v1:{entity_id}"))
