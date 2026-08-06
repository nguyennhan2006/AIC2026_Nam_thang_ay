"""Các model FPT vừa được cắm vào những chỗ trước đây rỗng.

Trước đợt này, file `.env.fpt.local` khai báo 9 model nhưng chỉ 3 biến được
`Settings` đọc — và đo thực tế cho thấy MỘT trong ba cái đó (QA LLM) hỏng ở
mọi lệnh gọi, nên thực chất chỉ 2 model sống. Test ở đây khoá lại phần wiring
mới bằng client giả, không chạm mạng.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from online.adapters.fpt_query import (
    FptQueryExpander,
    FptQueryTranslator,
    TranslatingTextEncoder,
)
from online.adapters.provider_errors import ProviderError, SchemaInvalidError
from online.adapters.rerank import FptVlmReranker
from online.domain.evidence import EvidencePack
from online.domain.candidate import FrameEvidence
from online.errors import DependencyUnavailableError


def run(coro):
    return asyncio.run(coro)


class _FakeChat:
    """Client giả trả sẵn text theo thứ tự, hoặc ném lỗi."""

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        return type("R", (), {"text": reply})()


def _pack(candidate_id: str, frame_paths: list[str]) -> EvidencePack:
    return EvidencePack(
        candidate_id=candidate_id,
        video_id="L21_V001",
        scene_id=candidate_id,
        start_frame=0,
        end_frame_exclusive=100,
        start_sec=0.0,
        end_sec=4.0,
        best_frame_idx=0,
        keyframes=[
            FrameEvidence(
                keyframe_id=f"{candidate_id}_{index}",
                video_id="L21_V001",
                scene_id=candidate_id,
                frame_idx=index,
                timestamp_sec=float(index),
                image_path=path,
            )
            for index, path in enumerate(frame_paths)
        ],
    )


class ReasoningModelResponseTests(unittest.TestCase):
    """`content=None` có ba nguyên nhân khác hẳn nhau, không được gộp làm một."""

    def _chat(self, *, content, reasoning, finish_reason, max_tokens=200):
        from online.adapters.fpt_client import FptClient

        client = FptClient.__new__(FptClient)
        body = {
            "model": "Qwen3.6-27B",
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": reasoning},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        client._call_with_retry = lambda path, payload: (body, 10, 0)
        return client.chat_completion([], model="Qwen3.6-27B", max_tokens=max_tokens)

    def test_budget_eaten_by_reasoning_gives_an_actionable_error(self) -> None:
        """Hết `max_tokens` giữa phần suy luận -> phải nói ĐÚNG cách sửa.

        Thông báo cũ ("content không phải string") trỏ sang lỗi schema và làm
        mất rất nhiều thời gian, trong khi nguyên nhân thật chỉ là ngân sách
        token: Qwen3.6-27B cần ~1650 token chỉ để dịch một câu.
        """

        with self.assertRaises(SchemaInvalidError) as ctx:
            self._chat(content=None, reasoning="đang nghĩ...", finish_reason="length")
        message = str(ctx.exception)
        self.assertIn("REASONING", message)
        self.assertIn("max_tokens", message)
        self.assertIn("gemma-4-31B-it", message)

    def test_answer_delivered_in_reasoning_field_is_still_the_answer(self) -> None:
        """Quirk đo được của FPT: model reasoning + `response_format=json_object`
        trả câu trả lời HOÀN CHỈNH trong `reasoning_content`, `content` là None,
        `finish_reason` vẫn là "stop". Đọc sai field ở đây làm hỏng TOÀN BỘ QA
        qua LLM — đã xảy ra thật và không ai phát hiện vì QA lặng lẽ rơi về
        rule-based.
        """

        result = self._chat(
            content=None,
            reasoning='{"answer":"chai nhựa","confidence":0.9}',
            finish_reason="stop",
        )
        self.assertIn("chai nhựa", result.text)

    def test_no_content_and_no_reasoning_is_still_a_schema_error(self) -> None:
        with self.assertRaises(SchemaInvalidError):
            self._chat(content=None, reasoning=None, finish_reason="stop")


class VlmRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        for name in ("a0.jpg", "a1.jpg", "b0.jpg"):
            (self.tmp / name).write_bytes(b"\xff\xd8\xff\xdb not-a-real-jpeg")

    def _reranker(self, replies, **kwargs):
        return FptVlmReranker(
            _FakeChat(replies), model_id="vlm", data_root=self.tmp, **kwargs
        )

    def test_pack_score_is_the_best_frame_not_the_average(self) -> None:
        """Một scene đúng chỉ cần MỘT khung hình chứng minh.

        Các khung khác cùng scene có thể rơi vào cảnh chuyển; lấy trung bình sẽ
        phạt oan chính những scene đúng có nhiều frame.
        """

        reranker = self._reranker(
            [
                '{"relevance":0.1,"must_match_coverage":0.1,"contradictions":[],"evidence_summary":"mờ"}',
                '{"relevance":0.9,"must_match_coverage":0.8,"contradictions":[],"evidence_summary":"rõ"}',
            ],
            frames_per_candidate=2,
        )
        [result] = run(reranker.score("q", [_pack("A", ["a0.jpg", "a1.jpg"])]))
        self.assertAlmostEqual(result["relevance"], 0.9)
        self.assertEqual(result["evidence_summary"], "rõ", "phải lấy mô tả của frame tốt nhất")
        self.assertEqual(result["frames_scored"], 2)

    def test_results_stay_aligned_with_the_input_packs(self) -> None:
        """`rerank_pipeline` ghép `results[i]` với `head[i]` theo VỊ TRÍ.

        Trả lệch thứ tự thì mỗi candidate nhận verdict của candidate khác — sai
        âm thầm, vì mọi thứ vẫn chạy và vẫn ra bảng kết quả.
        """

        # `max_concurrency=1` để fake trả reply theo đúng thứ tự: các lệnh gọi
        # thật chạy song song qua `asyncio.to_thread` nên thứ tự tiêu thụ reply
        # không xác định. Đó là giới hạn của FAKE, không phải của adapter —
        # adapter ghép kết quả theo chỉ số job chứ không theo thứ tự trả về, và
        # đó chính là tính chất test này khoá lại.
        reranker = self._reranker(
            [
                '{"relevance":0.2,"must_match_coverage":0,"contradictions":[],"evidence_summary":"A"}',
                '{"relevance":0.7,"must_match_coverage":0,"contradictions":[],"evidence_summary":"B"}',
            ],
            frames_per_candidate=1,
            max_concurrency=1,
        )
        packs = [_pack("A", ["a0.jpg"]), _pack("B", ["b0.jpg"])]
        results = run(reranker.score("q", packs))
        self.assertEqual([item["candidate_id"] for item in results], ["A", "B"])
        self.assertAlmostEqual(results[1]["relevance"], 0.7)

    def test_one_broken_frame_does_not_kill_the_stage_but_is_counted(self) -> None:
        reranker = self._reranker(
            [
                ProviderError("502"),
                '{"relevance":0.6,"must_match_coverage":0,"contradictions":[],"evidence_summary":"ok"}',
            ],
            frames_per_candidate=2,
        )
        [result] = run(reranker.score("q", [_pack("A", ["a0.jpg", "a1.jpg"])]))
        self.assertAlmostEqual(result["relevance"], 0.6)
        self.assertEqual((result["frames_scored"], result["frames_failed"]), (1, 1))

    def test_every_call_failing_raises_so_the_pipeline_reports_it(self) -> None:
        reranker = self._reranker(
            [ProviderError("502"), ProviderError("502")], frames_per_candidate=2
        )
        with self.assertRaises(DependencyUnavailableError):
            run(reranker.score("q", [_pack("A", ["a0.jpg", "a1.jpg"])]))

    def test_missing_image_files_raise_instead_of_scoring_nothing(self) -> None:
        """Ảnh không có thì trả toàn 0.0 sẽ xếp lại hạng theo dữ liệu rỗng.

        Đó là hạ cấp âm thầm: điểm số vẫn ra, chỉ là vô nghĩa.
        """

        reranker = self._reranker([], frames_per_candidate=2)
        with self.assertRaises(DependencyUnavailableError) as ctx:
            run(reranker.score("q", [_pack("A", ["khong-ton-tai.jpg"])]))
        self.assertIn("AIC_DATA_ROOT", str(ctx.exception))


class QueryTranslationTests(unittest.TestCase):
    def test_dense_branch_encodes_the_english_translation(self) -> None:
        """Vector ảnh sinh bằng CLIP, mà text tower CLIP chỉ biết tiếng Anh."""

        seen: list[str] = []

        class Inner:
            async def encode(self, text: str) -> list[float]:
                seen.append(text)
                return [0.0]

        client = _FakeChat(["a column of water erupting from the ground"])
        encoder = TranslatingTextEncoder(
            Inner(), FptQueryTranslator(client, model_id="fast")
        )
        run(encoder.encode("cột nước phun lên từ lòng đất"))
        self.assertEqual(seen, ["a column of water erupting from the ground"])

    def test_translation_is_cached_per_query(self) -> None:
        """Một buổi eval chạy lại cùng bộ truy vấn nhiều lần; không cache thì
        vừa trả tiền lặp vừa làm kết quả dao động giữa các lần chạy."""

        client = _FakeChat(["water column", "KHAC HAN"])
        translator = FptQueryTranslator(client, model_id="fast")
        self.assertEqual(translator.translate("cột nước"), "water column")
        self.assertEqual(translator.translate("cột nước"), "water column")
        self.assertEqual(len(client.calls), 1)

    def test_disk_cache_survives_a_new_process(self) -> None:
        """Mỗi lần eval trước đây dịch lại cả 40 truy vấn dù nhiệt độ 0.

        Cache đĩa biến một kết quả đáng lẽ tất định thành thật sự tất định, và
        che luôn lỗi dịch ~1/40 lượt — lần chạy sau dùng bản dịch đã có thay vì
        tung xúc xắc lại.
        """

        cache = Path(tempfile.mkdtemp())
        first = _FakeChat(["water column erupting"])
        FptQueryTranslator(first, model_id="fast", cache_dir=cache).translate("cột nước")
        self.assertEqual(len(first.calls), 1)

        # Tiến trình mới: client rỗng, nếu còn gọi mạng thì sẽ trả "" và hỏng.
        second = _FakeChat([])
        text = FptQueryTranslator(second, model_id="fast", cache_dir=cache).translate("cột nước")
        self.assertEqual(text, "water column erupting")
        self.assertEqual(second.calls, [], "lẽ ra phải đọc từ cache, không gọi lại")

    def test_empty_response_is_retried_before_giving_up(self) -> None:
        """`FptClient` coi "trả về chuỗi rỗng" là thành công nên không retry.

        Mất bản dịch = mất cả nhánh dense của truy vấn đó, nên đáng thử lại
        ngay trong adapter.
        """

        client = _FakeChat(["", "", "a column of water"])
        translator = FptQueryTranslator(client, model_id="fast", max_attempts=3)
        self.assertEqual(translator.translate("cột nước"), "a column of water")
        self.assertEqual(len(client.calls), 3)

    def test_gives_up_loudly_after_max_attempts(self) -> None:
        client = _FakeChat(["", "", ""])
        translator = FptQueryTranslator(client, model_id="fast", max_attempts=3)
        with self.assertRaises(DependencyUnavailableError):
            translator.translate("cột nước")

    def test_failed_translation_raises_rather_than_encoding_vietnamese(self) -> None:
        """Rơi về bản tiếng Việt chính là tái lập trạng thái hỏng cần loại bỏ.

        Nhánh dense sẽ báo `failed` kèm warning; các nhánh khác vẫn chạy.
        """

        class Inner:
            async def encode(self, text: str) -> list[float]:
                raise AssertionError("không được encode khi dịch hỏng")

        client = _FakeChat([ProviderError("503")])
        encoder = TranslatingTextEncoder(
            Inner(), FptQueryTranslator(client, model_id="fast")
        )
        with self.assertRaises(DependencyUnavailableError):
            run(encoder.encode("cột nước"))

    def test_warmup_contract_survives_the_wrapper(self) -> None:
        """`build_container` dò warmup bằng `hasattr`. Wrapper nuốt mất method
        này sẽ âm thầm khôi phục lỗi cold-start của FIX-DETERMINISM-01."""

        warmed: list[bool] = []

        class Inner:
            def warmup(self) -> None:
                warmed.append(True)

            async def encode(self, text: str) -> list[float]:
                return [0.0]

        encoder = TranslatingTextEncoder(
            Inner(), FptQueryTranslator(_FakeChat([]), model_id="fast")
        )
        self.assertTrue(hasattr(encoder, "warmup"))
        encoder.warmup()
        self.assertEqual(warmed, [True])


class QueryExpanderTests(unittest.TestCase):
    def test_extracts_vietnamese_terms_from_a_wrapped_json_array(self) -> None:
        client = _FakeChat(['Đây là kết quả:\n```json\n["vòi nước", "tia nước"]\n```'])
        expander = FptQueryExpander(client, model_id="fast")
        self.assertEqual(expander.expand("cột nước"), "vòi nước tia nước")

    def test_expander_failure_degrades_to_no_extra_terms(self) -> None:
        """Khác với dịch cho CLIP: BM25 không có term mở rộng vẫn chạy ĐÚNG
        hoàn toàn, nên giết cả nhánh vì một lần gọi LLM lỗi là đánh đổi tệ."""

        expander = FptQueryExpander(_FakeChat([ProviderError("503")]), model_id="fast")
        self.assertEqual(expander.expand("cột nước"), "")

    def test_garbage_output_is_not_pasted_into_the_query(self) -> None:
        expander = FptQueryExpander(_FakeChat(["xin lỗi, tôi không hiểu"]), model_id="fast")
        self.assertEqual(expander.expand("cột nước"), "")


if __name__ == "__main__":
    unittest.main()
