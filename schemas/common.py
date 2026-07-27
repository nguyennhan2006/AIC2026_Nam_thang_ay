"""Compatibility façade — nguồn sự thật đã chuyển sang `datasection.schemas.common`.

TECH-DEBT: package `schemas/` ở gốc repo từng là bản trùng lặp độc lập của
`datasection/schemas/` (phát hiện khi 2 bản lệch nhau sau khi mở rộng `ColorFeature`
cho Search Mixing Console W1 — xem lịch sử commit "feat(offline): add CPU color
feature extraction"). Không còn chỗ nào trong repo import trực tiếp từ `schemas.*`
tại thời điểm chuyển sang façade này (xác nhận bằng `rg "from schemas\\.|import
schemas\\b"`), nhưng vẫn giữ package thay vì xoá hẳn — có thể có script/notebook
ngoài kiểm soát version control đang trỏ vào đây. Xoá hẳn `schemas/` chỉ khi rg
trên vẫn rỗng NGOÀI chính façade này, sau một khoảng thời gian đủ dài.
"""

from __future__ import annotations

from datasection.schemas.common import *  # noqa: F401,F403
from datasection.schemas.common import __all__  # noqa: F401
