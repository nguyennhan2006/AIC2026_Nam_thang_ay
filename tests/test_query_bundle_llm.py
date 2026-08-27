"""Tier 2 (LLM) của query prep: LLM chỉ ĐỀ XUẤT, rule là nền.

Mọi test dùng client giả — không gọi mạng, không tốn tiền, tất định.
Điều được kiểm là HỢP ĐỒNG AN TOÀN: hỏng kiểu gì thì bundle rule vẫn đi tiếp.
"""

from __future__ import annotations

import json
import unittest

from online.adapters.fpt_query_bundle import FptQueryBundlePreparer
from online.domain.models import SearchRequest, TaskType
from online.services.query.models import AnswerType, SearchQueryBundle
from online.services.query.router import QueryRouter

FISH_QUERY = (
    "Hình ảnh một con cá được đặt lên cân, sau đó có cảnh một con cá khác "
    "cùng loài bị một người cầm đuôi. Con số hiển thị cuối cùng trên cân là bao nhiêu?"
)

GOOD_PAYLOAD = {
    "visual_vi": "bàn tay đeo đồng hồ đổ chất lỏng vào bát trắng đặt trên cân điện tử, có một con cá nhỏ trong bát",
    "visual_en": "a hand pouring liquid into a white bowl on a digital scale with a small fish",
    "caption_vi": "cá cân điện tử bát trắng bàn tay đổ chất lỏng cân cá",
    "ocr_terms": ["kg", "g"],
    "asr_vi": "con cá này nặng bao nhiêu ký",
    "events": [],
    "answer_type": "numeric",
}


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    """Client giả trả về nội dung dựng sẵn, đếm số lần gọi."""

    def __init__(self, text: str, *, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    def chat_completion(self, messages, **kwargs):  # noqa: ANN001 - khớp chữ ký thật
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _Reply(self.text)


def _rule_bundle(query: str = FISH_QUERY, task: TaskType = TaskType.QA) -> SearchQueryBundle:
    return QueryRouter().prepare_sync(SearchRequest(query=query, task=task))


class RefineTests(unittest.TestCase):
    def test_llm_rewrites_visual_query_into_a_frame_description(self):
        """Giá trị của Tier 2: câu kể chuyện -> mô tả khung hình.

        Đây là chỗ đo được rank 35 -> 1 trên gold L21_V023 frame 25995.
        """

        rule = _rule_bundle()
        client = FakeClient(json.dumps(GOOD_PAYLOAD, ensure_ascii=False))
        refined = FptQueryBundlePreparer(client, model_id="m").refine(rule, task="QA")

        self.assertIn("bát trắng", refined.visual_query)
        self.assertIn("cân điện tử", refined.visual_query)
        # Phần hỏi và phần kể chuyện không còn nằm trong query đưa cho CLIP.
        self.assertNotIn("bao nhiêu", refined.visual_query)
        self.assertNotIn("sau đó", refined.visual_query)

    def test_each_engine_gets_its_own_shape_of_data(self):
        rule = _rule_bundle()
        client = FakeClient(json.dumps(GOOD_PAYLOAD, ensure_ascii=False))
        refined = FptQueryBundlePreparer(client, model_id="m").refine(rule, task="QA")

        self.assertNotEqual(refined.visual_query, refined.caption_query)
        self.assertNotEqual(refined.visual_query, refined.ocr_query)
        self.assertIn("kg", refined.ocr_query)
        # OCR chỉ nhận chữ có thể hiện trên màn hình, không nhận mô tả cảnh.
        self.assertNotIn("bàn tay", refined.ocr_query)
        self.assertEqual(refined.answer_type, AnswerType.NUMERIC)

    def test_quoted_phrases_from_rule_survive_llm_ocr_terms(self):
        """Chữ trong ngoặc kép là bằng chứng chắc chắn, không được LLM ghi đè."""

        rule = _rule_bundle(
            'Cảnh có dòng chữ "TÒA ÁN PHÚC THẨM PARIS"', TaskType.TEXTUAL_KIS
        )
        self.assertIn("TÒA ÁN PHÚC THẨM PARIS", rule.exact_phrases)

        payload = dict(GOOD_PAYLOAD, ocr_terms=["paris"])
        client = FakeClient(json.dumps(payload, ensure_ascii=False))
        refined = FptQueryBundlePreparer(client, model_id="m").refine(
            rule, task="TEXTUAL_KIS"
        )

        self.assertIn("TÒA ÁN PHÚC THẨM PARIS", refined.ocr_query)

    def test_json_wrapped_in_prose_or_fence_is_still_parsed(self):
        """Model `fast` hay bọc JSON trong ```json — không được coi là hỏng."""

        rule = _rule_bundle()
        wrapped = "Đây là kết quả:\n```json\n" + json.dumps(
            GOOD_PAYLOAD, ensure_ascii=False
        ) + "\n```"
        client = FakeClient(wrapped)
        refined = FptQueryBundlePreparer(client, model_id="m").refine(rule, task="QA")

        self.assertIn("bát trắng", refined.visual_query)


class FallbackTests(unittest.TestCase):
    """Hợp đồng an toàn: LLM hỏng kiểu gì thì bundle rule cũng phải đi tiếp."""

    def _assert_unchanged(self, client: FakeClient) -> None:
        rule = _rule_bundle()
        refined = FptQueryBundlePreparer(client, model_id="m").refine(rule, task="QA")
        self.assertEqual(refined.visual_query, rule.visual_query)
        self.assertEqual(refined.ocr_query, rule.ocr_query)

    def test_provider_error_keeps_rule_bundle(self):
        from online.adapters.provider_errors import ProviderError

        self._assert_unchanged(FakeClient("", error=ProviderError("503 upstream")))

    def test_malformed_json_keeps_rule_bundle(self):
        self._assert_unchanged(FakeClient("xin lỗi, tôi không chắc"))

    def test_empty_fields_keep_rule_bundle(self):
        payload = {"visual_vi": "", "ocr_terms": [], "answer_type": ""}
        self._assert_unchanged(FakeClient(json.dumps(payload)))

    def test_too_short_visual_is_rejected(self):
        """Chốt chặn chống đúng lỗi cũ: visual query bị rút còn vài chữ."""

        rule = _rule_bundle()
        payload = dict(GOOD_PAYLOAD, visual_vi="ao sau")
        client = FakeClient(json.dumps(payload, ensure_ascii=False))
        refined = FptQueryBundlePreparer(client, model_id="m").refine(rule, task="QA")

        self.assertEqual(refined.visual_query, rule.visual_query)
        self.assertNotEqual(refined.visual_query, "ao sau")

    def test_unknown_answer_type_is_ignored(self):
        rule = _rule_bundle()
        payload = dict(GOOD_PAYLOAD, answer_type="banana")
        client = FakeClient(json.dumps(payload, ensure_ascii=False))
        refined = FptQueryBundlePreparer(client, model_id="m").refine(rule, task="QA")

        self.assertEqual(refined.answer_type, rule.answer_type)


class CacheTests(unittest.TestCase):
    def test_same_query_calls_the_model_once(self):
        client = FakeClient(json.dumps(GOOD_PAYLOAD, ensure_ascii=False))
        preparer = FptQueryBundlePreparer(client, model_id="m")

        preparer.refine(_rule_bundle(), task="QA")
        preparer.refine(_rule_bundle(), task="QA")

        self.assertEqual(client.calls, 1)


class RouterIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_router_without_refiner_is_pure_rule(self):
        router = QueryRouter()
        bundle = await router.prepare(SearchRequest(query=FISH_QUERY, task=TaskType.QA))
        self.assertEqual(bundle.visual_query, router.prepare_sync(
            SearchRequest(query=FISH_QUERY, task=TaskType.QA)
        ).visual_query)

    async def test_router_applies_refiner(self):
        client = FakeClient(json.dumps(GOOD_PAYLOAD, ensure_ascii=False))
        router = QueryRouter(refiner=FptQueryBundlePreparer(client, model_id="m"))
        bundle = await router.prepare(SearchRequest(query=FISH_QUERY, task=TaskType.QA))
        self.assertIn("bát trắng", bundle.visual_query)

    async def test_refiner_raising_does_not_kill_the_query(self):
        class Exploding:
            def refine(self, bundle, *, task):  # noqa: ANN001, ARG002
                raise RuntimeError("boom")

        router = QueryRouter(refiner=Exploding())
        bundle = await router.prepare(SearchRequest(query=FISH_QUERY, task=TaskType.QA))
        self.assertTrue(bundle.visual_query)


if __name__ == "__main__":
    unittest.main()
