"""Public metadata contracts for AIC 2026 Version 1."""

from .common import BoundingBox, EmbeddingReference, ModelProvenance, VectorLocation
from .keyframe import (
    CaptionRecord,
    ColorFeature,
    Keyframe,
    KeyframeRole,
    ObjectInstance,
    OCRInstance,
    QualitySignals,
)

__all__ = [
    "BoundingBox",
    "CaptionRecord",
    "ColorFeature",
    "EmbeddingReference",
    "Keyframe",
    "KeyframeRole",
    "ModelProvenance",
    "ObjectInstance",
    "OCRInstance",
    "QualitySignals",
    "VectorLocation",
]
