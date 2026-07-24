"""A lightweight local replacement for the AIC Elasticsearch search layer."""

from .builder import build_index
from .config import EngineConfig
from .engine import LocalHybridSearchEngine

__all__ = ["EngineConfig", "LocalHybridSearchEngine", "build_index"]
__version__ = "0.2.0"
