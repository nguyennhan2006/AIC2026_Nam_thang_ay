"""Query Routing V2 — Modality-Aware Query Preparation.

Architecture:
    RAW QUERY
        │
        ▼
    Normalize + Parse (task/intent/events)
        │
        ▼
    ┌─────┴─────┐
    │           │
  VISUAL    LEXICAL/EVIDENCE
    │           │
    ▼           ▼
Jina CLIP   Caption/OCR/ASR
    │           │
    └─────┬─────┘
          ▼
     Weighted RRF
          │
          ▼
     Scene/Frame

Usage:
    from online.services.query import QueryRouter

    router = QueryRouter()
    bundle = await router.prepare(request)
    results = await dense.search(bundle.visual_query)
"""

from online.services.query.router import QueryRouter

__all__ = ["QueryRouter"]
