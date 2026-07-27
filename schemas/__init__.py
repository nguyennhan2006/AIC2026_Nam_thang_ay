"""Compatibility façade — nguồn sự thật đã chuyển sang `datasection.schemas`.
Xem `schemas/common.py` cho lý do và điều kiện xoá hẳn package này.
"""

from datasection.schemas import *  # noqa: F401,F403
from datasection.schemas import __all__  # noqa: F401
