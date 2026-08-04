"""Shared Pydantic base — tách riêng để tránh circular import giữa
`online/domain/models.py` và `online/domain/search_config.py` (models.py cần
`SearchOptions` từ search_config.py, search_config.py cần cùng convention
StrictModel với models.py)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
