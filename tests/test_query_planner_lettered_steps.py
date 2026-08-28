"""TRAKE thật dùng format "E1 ... E2 ... E3 ..." — không có ngoặc.

Trước bản sửa này, `RuleBasedQueryPlanner` chỉ nhận dạng "(1)...(2)..."
(NUMBERED_STEP_RE) và các từ nối "sau đó"/"tiếp theo" (TEMPORAL_RE). Đề sơ
tuyển thật (`Example_and_practical_data/AIC2026-SoTuyen1/.../query-p1-16-trake.txt`)
dùng "E1 ...\nE2 ...\nE3 ..." — khớp với KHÔNG pattern nào trong hai cái trên,
nên `plan.events` dừng ở 1 phần tử (nguyên đoạn văn) thay vì 4. `search.py`
chỉ bật chế độ TRAKE khi `len(plan.events) >= 2`, nên câu TRAKE thật không
bao giờ chạy TRAKE — rơi về tìm 1 câu duy nhất trên toàn đoạn văn.

Bộ test này khoá lại: format "E<số>" phải tách đúng, và hai format cũ
("(1)(2)(3)", "sau đó/tiếp theo") không bị phá.
"""

from __future__ import annotations

import asyncio
import unittest

from online.domain.models import SearchRequest, TaskType
from online.services.query_planner import RuleBasedQueryPlanner


def run(coro):
    return asyncio.run(coro)


# Nguyên văn query-p1-16-trake.txt (đề sơ tuyển thật).
P1_16_TEXT = (
    "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh "
    "lá cờ trắng viền đỏ.\n"
    "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay "
    "vòng.\n"
    "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh "
    "trụ (thời điểm đâu tiên các chân của lân đặt trên trụ sau khi xoay).\n"
    "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
)


class LetteredStepPlanningTests(unittest.TestCase):
    def _plan(self, query: str, task: TaskType = TaskType.TRAKE):
        planner = RuleBasedQueryPlanner()
        return run(planner.plan(SearchRequest(query=query, task=task)))

    def test_lettered_e_format_splits_into_three_events(self) -> None:
        plan = self._plan(P1_16_TEXT)
        # Đoạn dẫn trước "E1" bị bỏ, giống hệt cách "(1)(2)(3)" bỏ phần dẫn
        # trước "(1)" — chỉ giữ 3 khoảnh khắc thật sự cần khớp theo thứ tự.
        self.assertEqual(len(plan.events), 3)
        self.assertIn("hai con rồng vàng", plan.events[0].text)
        self.assertIn("xoay người trên các thanh trụ", plan.events[1].text)
        self.assertIn("dùi chạm vào kẻng đồng", plan.events[2].text)

    def test_parenthesized_numbering_still_works(self) -> None:
        plan = self._plan(
            "Người đàn ông bước vào: (1) cào muối trên sân; "
            "(2) vẫy tay chào; (3) đứng trước căn nhà.",
        )
        self.assertEqual(len(plan.events), 3)
        self.assertIn("cào muối", plan.events[0].text)
        self.assertIn("vẫy tay", plan.events[1].text)
        self.assertIn("đứng trước căn nhà", plan.events[2].text)

    def test_sau_do_marker_still_works_without_numbering(self) -> None:
        plan = self._plan(
            "Người đàn ông cào muối trên sân, sau đó vẫy tay chào, "
            "cuối cùng đứng trước căn nhà.",
        )
        self.assertEqual(len(plan.events), 3)

    def test_plain_single_scene_query_stays_one_event(self) -> None:
        plan = self._plan("Một người phụ nữ mặc áo dài màu hồng đang đứng")
        self.assertEqual(len(plan.events), 1)

    def test_lettered_marker_does_not_fire_on_non_trake_task(self) -> None:
        # Chỉ TRAKE mới thử tách "(1)(2)(3)"/"E1E2E3" — KIS giữ nguyên 1
        # event như trước (rule-tier planner không đổi hành vi cho KIS).
        plan = self._plan(P1_16_TEXT, task=TaskType.TEXTUAL_KIS)
        self.assertEqual(len(plan.events), 1)


if __name__ == "__main__":
    unittest.main()
