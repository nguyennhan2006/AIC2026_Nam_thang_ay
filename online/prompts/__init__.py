"""Prompt registry — xem `online/prompts/registry.py`."""

from online.prompts.registry import (
    EXPAND_QUERY,
    PREPARE_QUERY_BUNDLE,
    PROMPTS,
    QA_ANSWER,
    RECOMMEND_WEIGHTS,
    SELECT_EVIDENCE,
    TRANSLATE_QUERY,
    VLM_RERANK,
    PromptSpec,
    prompts_by_role,
)

__all__ = [
    "EXPAND_QUERY",
    "PREPARE_QUERY_BUNDLE",
    "PROMPTS",
    "PromptSpec",
    "QA_ANSWER",
    "RECOMMEND_WEIGHTS",
    "SELECT_EVIDENCE",
    "TRANSLATE_QUERY",
    "VLM_RERANK",
    "prompts_by_role",
]
