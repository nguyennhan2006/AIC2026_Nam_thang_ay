"""Compatibility façade — nguồn sự thật đã chuyển sang `datasection.schemas.keyframe`.
Xem `schemas/common.py` cho lý do và điều kiện xoá hẳn package này.
"""

from __future__ import annotations

from datasection.schemas.keyframe import *  # noqa: F401,F403
from datasection.schemas.keyframe import __all__  # noqa: F401
