"""FptQaAnswerer (PR-15+): parse JSON strict, không bịa khi evidence rỗng,
và lỗi provider phải nổi lên thành DependencyUnavailableError để caller
(`QaProcessor.answer_async`) fallback về rule-based thay vì sập cả request.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from online.adapters.fpt_client import FptChatResult, FptUsage
from online.adapters.provider_errors import ProviderTimeoutError
from online.adapters.qa_llm import FptQaAnswerer
from online.domain.candidate import FrameEvidence
from online.domain.evidence import EvidencePack
from online.errors import DependencyUnavailableError


def _pack(**overrides) -> EvidencePack:
    frame = FrameEvidence(
        keyframe_id="p1_F000100", video_id="L21_V001", scene_id="p1",
        frame_idx=100, timestamp_sec=4.0, image_path="f.jpg",
    )
    defaults = dict(
        candidate_id="p1", video_id="L21_V001", scene_id="p1",
        start_frame=0, end_frame_exclusive=200, start_sec=0.0, end_sec=8.0,
        keyframes=[frame], best_frame_idx=100,
        caption_text="hai xe máy va chạm liên hoàn trên quốc lộ",
    )
    defaults.update(overrides)
    return EvidencePack(**defaults)


def _usage() -> FptUsage:
    return FptUsage(model_id="test-llm", input_tokens=10, output_tokens=5, latency_ms=20, retry_count=0)


class FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self.error is not None:
            raise self.error
        return FptChatResult(text=self.text or "", usage=_usage(), raw={})


class FptQaAnswererTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_json_becomes_answer_candidate(self) -> None:
        client = FakeClient(text='{"answer": "2", "answer_type": "count", "confidence": 0.8}')
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("Có bao nhiêu xe máy?", _pack())
        self.assertIsNotNone(result)
        self.assertEqual(result.canonical, "2")
        self.assertEqual(result.answer_type, "count")
        self.assertEqual(result.confidence, 0.8)
        self.assertEqual(result.source, "fpt_llm")

    async def test_json_wrapped_in_prose_is_extracted(self) -> None:
        client = FakeClient(text='Đây là kết quả: {"answer": "đỏ", "answer_type": "color", "confidence": 0.6} nhé')
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("Áo màu gì?", _pack())
        self.assertIsNotNone(result)
        self.assertEqual(result.canonical, "đỏ")

    async def test_empty_answer_field_returns_none(self) -> None:
        client = FakeClient(text='{"answer": "", "answer_type": "entity", "confidence": 0.9}')
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("Đó là gì?", _pack())
        self.assertIsNone(result)

    async def test_malformed_json_returns_none_not_raise(self) -> None:
        client = FakeClient(text="không phải JSON chút nào")
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("Đó là gì?", _pack())
        self.assertIsNone(result)

    async def test_unknown_answer_type_falls_back_to_other(self) -> None:
        client = FakeClient(text='{"answer": "42", "answer_type": "banana", "confidence": 0.5}')
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("?", _pack())
        self.assertEqual(result.answer_type, "other")

    async def test_confidence_out_of_range_is_clamped(self) -> None:
        client = FakeClient(text='{"answer": "42", "answer_type": "count", "confidence": 5}')
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("?", _pack())
        self.assertEqual(result.confidence, 1.0)

    async def test_empty_evidence_short_circuits_without_calling_client(self) -> None:
        client = FakeClient(text='{"answer": "x", "answer_type": "other", "confidence": 0.5}')
        answerer = FptQaAnswerer(client, model_id="test-llm")
        result = await answerer.answer("?", _pack(caption_text=None))
        self.assertIsNone(result)
        self.assertEqual(client.calls, [])

    async def test_provider_error_is_wrapped_as_dependency_unavailable(self) -> None:
        client = FakeClient(error=ProviderTimeoutError("timeout"))
        answerer = FptQaAnswerer(client, model_id="test-llm")
        with self.assertRaises(DependencyUnavailableError):
            await answerer.answer("?", _pack())


if __name__ == "__main__":
    unittest.main()
