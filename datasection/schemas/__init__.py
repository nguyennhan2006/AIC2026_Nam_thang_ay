"""Public metadata contracts for AIC 2026 Version 1."""

from .common import BoundingBox, EmbeddingReference, ModelProvenance, VectorLocation
from .keyframe import (
    CaptionRecord,
    ColorFeature,
    Keyframe,
    KeyframeRole,
    NamedColorRatio,
    ObjectInstance,
    OCRInstance,
    QualitySignals,
)
from .scene import ASRSegment, Scene, SceneCaptionRecord, SceneKeyword, TransitionType
from .video import Video
from .dataset import DatasetManifest, IndexArtifact, ModelArtifact

__all__ = [
    "BoundingBox",
    "ASRSegment",
    "CaptionRecord",
    "ColorFeature",
    "EmbeddingReference",
    "Keyframe",
    "KeyframeRole",
    "ModelProvenance",
    "NamedColorRatio",
    "ObjectInstance",
    "OCRInstance",
    "QualitySignals",
    "Scene",
    "SceneCaptionRecord",
    "SceneKeyword",
    "TransitionType",
    "VectorLocation",
    "Video",
    "DatasetManifest",
    "IndexArtifact",
    "ModelArtifact",
]
