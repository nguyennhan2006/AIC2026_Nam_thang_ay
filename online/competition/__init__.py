"""Tầng nộp bài — tách hẳn khỏi SearchService (PR-08).

`SearchService` trả kết quả *retrieval*; đúng/sai của một submission là luật
của cuộc thi, không phải logic tìm kiếm. Gộp hai việc vào một chỗ (như từng
làm với UI cũ) sẽ khiến thay đổi luật chấm kéo theo sửa cả pipeline search.
"""

from __future__ import annotations
