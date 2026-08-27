"""Hai cố vấn LLM: đề xuất trọng số, và lọc bằng chứng.

Cả hai chỉ đọc kết quả rồi nói lại — không đụng retrieval. Nên nguyên tắc
xuyên suốt ở đây là: hỏng thì mất LỜI KHUYÊN, không được mất KẾT QUẢ.
"""

from __future__ import annotations

import asyncio
import unittest

from online.adapters.fpt_advisor import FptEvidenceSelector, FptWeightRecommender
from online.adapters.provider_errors import ProviderError
from online.domain.evidence import EvidencePack
from online.prompts import PROMPTS, prompts_by_role


def run(coro):
    return asyncio.run(coro)


class _FakeChat:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return type("R", (), {"text": reply})()


BRANCHES = ["dense_visual", "bm25_caption", "bm25_ocr", "ocr_fuzzy"]


def _pack(text_caption: str, ocr: str) -> EvidencePack:
    return EvidencePack(
        candidate_id="L21_V001_S0001",
        video_id="L21_V001",
        scene_id="L21_V001_S0001",
        start_frame=0,
        end_frame_exclusive=100,
        start_sec=0.0,
        end_sec=4.0,
        caption_text=text_caption,
        ocr_text=ocr,
    )


class WeightRecommenderTests(unittest.TestCase):
    def test_recommends_only_branches_that_actually_exist(self) -> None:
        """LLM bịa thêm `branch_id` thì phần bịa phải bị bỏ.

        Đề xuất trọng số cho nhánh không tồn tại sẽ đi thẳng xuống tầng cấu
        hình và bị từ chối ở đó bằng 422 — người dùng nhận lỗi cho thứ họ không
        hề gõ ra.
        """

        client = _FakeChat(
            ['{"weights":{"bm25_ocr":3.0,"khong_co_that":2.0},"reason":"r","disabled":[]}']
        )
        out = run(
            FptWeightRecommender(client, model_id="m").recommend(
                "q", task="TEXTUAL_KIS", branch_ids=BRANCHES
            )
        )
        self.assertEqual(set(out["weights"]), {"bm25_ocr"})

    def test_weights_are_clamped_to_the_allowed_range(self) -> None:
        client = _FakeChat(
            ['{"weights":{"bm25_ocr":99,"dense_visual":-5},"reason":"r","disabled":[]}']
        )
        out = run(
            FptWeightRecommender(client, model_id="m").recommend(
                "q", task="TEXTUAL_KIS", branch_ids=BRANCHES
            )
        )
        self.assertEqual(out["weights"], {"bm25_ocr": 3.0, "dense_visual": 0.0})

    def test_zero_weight_survives_because_it_means_turn_the_branch_off(self) -> None:
        """0.0 là một QUYẾT ĐỊNH ("tắt nhánh này"), không phải giá trị thiếu.

        Lọc bỏ 0.0 như thể là falsy sẽ biến "tắt hẳn OCR" thành "không nói gì
        về OCR" — mất đúng phần có giá trị nhất của đề xuất.
        """

        client = _FakeChat(['{"weights":{"dense_visual":0.0},"reason":"r","disabled":["dense_visual"]}'])
        out = run(
            FptWeightRecommender(client, model_id="m").recommend(
                "q", task="TEXTUAL_KIS", branch_ids=BRANCHES
            )
        )
        self.assertIn("dense_visual", out["weights"])
        self.assertEqual(out["weights"]["dense_visual"], 0.0)

    def test_provider_failure_returns_none_instead_of_raising(self) -> None:
        client = _FakeChat([ProviderError("503")])
        out = run(
            FptWeightRecommender(client, model_id="m").recommend(
                "q", task="TEXTUAL_KIS", branch_ids=BRANCHES
            )
        )
        self.assertIsNone(out)

    def test_prompt_version_is_reported_so_two_runs_can_be_compared(self) -> None:
        client = _FakeChat(['{"weights":{"bm25_ocr":1.0},"reason":"r","disabled":[]}'])
        out = run(
            FptWeightRecommender(client, model_id="m").recommend(
                "q", task="TEXTUAL_KIS", branch_ids=BRANCHES
            )
        )
        self.assertEqual(out["prompt_version"], "search.recommend_weights@1")


class EvidenceSelectorTests(unittest.TestCase):
    def test_drops_broadcast_overlay_and_keeps_scene_content(self) -> None:
        """Đây là lý do module tồn tại.

        `EvidencePack.rerank_text()` gộp máy móc caption + OCR + ASR, nên logo
        đài và đồng hồ trên màn hình đứng ngang hàng với nội dung cảnh. Quan
        sát thật trước khi có bước lọc: bằng chứng trả về là "HTV9 HD" và
        "06:33:29" cho một truy vấn không hỏi kênh nào cũng chẳng hỏi mấy giờ.
        Những chuỗi đó có ở MỌI khung hình nên không chứng minh được gì.
        """

        client = _FakeChat(
            [
                '{"supports":true,'
                '"evidence":["Một cột nước cao vút phun lên từ giếng"],'
                '"reason":"khớp trực tiếp",'
                '"dropped_as_overlay":["HTV9 HD","06:33:29"]}'
            ]
        )
        out = run(
            FptEvidenceSelector(client, model_id="m").select(
                "cột nước phun lên",
                _pack("Một cột nước cao vút phun lên từ giếng", "HTV9 HD 06:33:29"),
            )
        )
        self.assertTrue(out["supports"])
        self.assertEqual(out["evidence"], ["Một cột nước cao vút phun lên từ giếng"])
        self.assertIn("HTV9 HD", out["dropped_as_overlay"])

    def test_says_it_does_not_support_rather_than_inventing_evidence(self) -> None:
        client = _FakeChat(
            [
                '{"supports":false,"evidence":[],"reason":"không có gì khớp",'
                '"dropped_as_overlay":[]}'
            ]
        )
        out = run(
            FptEvidenceSelector(client, model_id="m").select(
                "bảng hiệu có chữ", _pack("tin tức vụ án", "ĐỒNG THÁP")
            )
        )
        self.assertFalse(out["supports"])
        self.assertEqual(out["evidence"], [])

    def test_select_many_keeps_the_order_of_the_input_packs(self) -> None:
        """Caller ghép kết quả với `results` theo VỊ TRÍ."""

        client = _FakeChat(
            [
                '{"supports":true,"evidence":["A"],"reason":"","dropped_as_overlay":[]}',
                '{"supports":true,"evidence":["B"],"reason":"","dropped_as_overlay":[]}',
            ]
        )
        packs = [_pack("A", ""), _pack("B", "")]
        out = run(FptEvidenceSelector(client, model_id="m").select_many("q", packs))
        self.assertEqual([item["evidence"] for item in out], [["A"], ["B"]])

    def test_empty_pack_is_skipped_without_calling_the_model(self) -> None:
        client = _FakeChat([])
        out = run(FptEvidenceSelector(client, model_id="m").select("q", _pack("", "")))
        self.assertIsNone(out)
        self.assertEqual(client.calls, [])

    def test_failure_returns_none_so_the_search_result_survives(self) -> None:
        client = _FakeChat([ProviderError("503")])
        out = run(FptEvidenceSelector(client, model_id="m").select("q", _pack("caption", "")))
        self.assertIsNone(out)


class PromptRegistryTests(unittest.TestCase):
    def test_every_prompt_declares_a_model_role_not_a_model_name(self) -> None:
        """Tên model là chuyện cấu hình theo môi trường; "việc này có cần suy
        luận nhiều bước không" là thuộc tính của chính việc đó."""

        for spec in PROMPTS.values():
            self.assertIn(spec.model_role, {"fast", "reasoning", "vlm"}, spec.prompt_id)

    def test_reasoning_prompts_get_a_budget_large_enough_to_answer(self) -> None:
        """Model reasoning tiêu `max_tokens` cho phần suy luận TRƯỚC rồi mới
        sinh câu trả lời. Ngân sách hẹp làm hỏng 100% lệnh gọi mà không có dấu
        hiệu gì — đã xảy ra với QA (`max_tokens=200` hard-code)."""

        for spec in PROMPTS.values():
            if spec.model_role == "reasoning":
                self.assertGreaterEqual(spec.max_tokens, 1000, spec.prompt_id)

    def test_inventory_groups_prompts_by_the_model_role_they_need(self) -> None:
        grouped = prompts_by_role()
        self.assertEqual(set(grouped), {"fast", "reasoning", "vlm"})
        self.assertIn("query.translate_vi_en@1", grouped["fast"])
        self.assertIn("evidence.select@1", grouped["reasoning"])

    def test_prompt_ids_are_unique(self) -> None:
        # Kiểm ĐÚNG tính duy nhất của prompt_id, không kiểm số lượng: `PROMPTS`
        # là dict khoá theo prompt_id nên hai spec trùng id sẽ nuốt nhau trong
        # im lặng — đó mới là điều cần bắt. Chốt một con số cứng chỉ khiến mọi
        # lần thêm prompt đều làm đỏ test mà không phát hiện lỗi nào.
        ids = [spec.prompt_id for spec in PROMPTS.values()]
        self.assertEqual(len(ids), len(set(ids)))
        for prompt_id, spec in PROMPTS.items():
            self.assertEqual(prompt_id, spec.prompt_id)


if __name__ == "__main__":
    unittest.main()
