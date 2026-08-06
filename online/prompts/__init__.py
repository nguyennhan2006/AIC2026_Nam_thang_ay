"""Prompt registry — xem `online/prompts/registry.py`."""

from online.prompts.registry import (
    EXPAND_QUERY,
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
    "PROMPTS",
    "PromptSpec",
    "QA_ANSWER",
    "RECOMMEND_WEIGHTS",
    "SELECT_EVIDENCE",
    "TRANSLATE_QUERY",
    "VLM_RERANK",
    "prompts_by_role",
]
