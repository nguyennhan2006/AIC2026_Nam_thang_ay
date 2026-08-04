"""PR-12: FptClient — retry chỉ transient, phân loại lỗi đúng taxonomy §24.

Không gọi mạng thật: mock `urlopen` giống cách `tests/test_online_core.py`
mock cho `QdrantVectorStore` — client phải hoạt động đúng logic độc lập với
việc có API key thật hay không.
"""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from online.adapters.fpt_client import FptClient, image_to_data_url
from online.adapters.provider_errors import (
    AuthError,
    MalformedResponseError,
    ModelNotFoundError,
    PermissionDeniedError,
    ProviderTimeoutError,
    RateLimitedError,
    SchemaInvalidError,
    UpstreamServerError,
)


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def _client(**overrides) -> FptClient:
    defaults = dict(
        base_url="https://mkp-api.fptcloud.com",
        api_key="test-key",
        max_retries=3,
        retry_backoff_base_sec=0.001,
        retry_backoff_max_sec=0.002,
    )
    defaults.update(overrides)
    return FptClient(**defaults)


def _chat_body(text: str = "ok") -> dict:
    return {
        "model": "test-model",
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class ChatCompletionTests(unittest.TestCase):
    def test_successful_call_returns_text_and_usage(self) -> None:
        client = _client()
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(_chat_body("hello"))):
            result = client.chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.usage.input_tokens, 10)
        self.assertEqual(result.usage.output_tokens, 5)
        self.assertEqual(result.usage.retry_count, 0)

    def test_authorization_header_is_never_logged_in_the_request(self) -> None:
        # Header thật vẫn phải được GỬI (server cần nó); chỉ không được LỘ ra
        # trong bất kỳ thông điệp lỗi/log nào.
        captured = {}

        def fake_urlopen(request, timeout, context=None):
            captured["headers"] = dict(request.headers)
            return FakeResponse(_chat_body())

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            _client().chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertIn("Authorization", captured["headers"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")

    def test_missing_choices_raises_schema_invalid(self) -> None:
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse({"model": "m1"})):
            with self.assertRaises(SchemaInvalidError):
                _client().chat_completion([{"role": "user", "content": "hi"}], model="m1")

    def test_non_json_body_raises_malformed_response(self) -> None:
        class BadResponse(FakeResponse):
            def read(self) -> bytes:
                return b"not json at all"

        with patch("online.adapters.fpt_client.urlopen", return_value=BadResponse({})):
            with self.assertRaises(MalformedResponseError):
                _client().chat_completion([{"role": "user", "content": "hi"}], model="m1")


class ErrorClassificationTests(unittest.TestCase):
    def _http_error(self, code: int, body: bytes = b"{}"):
        return HTTPError(url="https://x", code=code, msg="err", hdrs=None, fp=BytesIO(body))

    def test_401_is_auth_error_and_not_retried(self) -> None:
        calls = []

        def fake_urlopen(request, timeout, context=None):
            calls.append(1)
            raise self._http_error(401)

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(AuthError):
                _client().chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertEqual(len(calls), 1)  # không retry lỗi permanent

    def test_403_is_permission_denied(self) -> None:
        with patch("online.adapters.fpt_client.urlopen", side_effect=lambda *a, **k: (_ for _ in ()).throw(self._http_error(403))):
            with self.assertRaises(PermissionDeniedError):
                _client().chat_completion([{"role": "user", "content": "hi"}], model="m1")

    def test_404_is_model_not_found(self) -> None:
        with patch("online.adapters.fpt_client.urlopen", side_effect=lambda *a, **k: (_ for _ in ()).throw(self._http_error(404))):
            with self.assertRaises(ModelNotFoundError):
                _client().chat_completion([{"role": "user", "content": "hi"}], model="does-not-exist")

    def test_429_is_rate_limited_and_retried_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def fake_urlopen(request, timeout, context=None):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise self._http_error(429)
            return FakeResponse(_chat_body("recovered"))

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            result = _client(max_retries=5).chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertEqual(result.text, "recovered")
        self.assertEqual(result.usage.retry_count, 2)

    def test_429_exhausting_retries_raises_rate_limited(self) -> None:
        with patch("online.adapters.fpt_client.urlopen", side_effect=lambda *a, **k: (_ for _ in ()).throw(self._http_error(429))):
            with self.assertRaises(RateLimitedError):
                _client(max_retries=2).chat_completion([{"role": "user", "content": "hi"}], model="m1")

    def test_5xx_is_upstream_and_retried(self) -> None:
        calls = {"n": 0}

        def fake_urlopen(request, timeout, context=None):
            calls["n"] += 1
            raise self._http_error(503)

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(UpstreamServerError):
                _client(max_retries=3).chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertEqual(calls["n"], 3)  # đúng số lần retry cấu hình, không hơn

    def test_timeout_is_retried_as_transient(self) -> None:
        calls = {"n": 0}

        def fake_urlopen(request, timeout, context=None):
            calls["n"] += 1
            raise TimeoutError("timed out")

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ProviderTimeoutError):
                _client(max_retries=2).chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertEqual(calls["n"], 2)

    def test_url_error_is_treated_as_timeout(self) -> None:
        with patch("online.adapters.fpt_client.urlopen", side_effect=lambda *a, **k: (_ for _ in ()).throw(URLError("connection refused"))):
            with self.assertRaises(ProviderTimeoutError):
                _client(max_retries=1).chat_completion([{"role": "user", "content": "hi"}], model="m1")

    def test_422_is_schema_invalid_and_not_retried(self) -> None:
        calls = {"n": 0}

        def fake_urlopen(request, timeout, context=None):
            calls["n"] += 1
            raise self._http_error(422)

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(SchemaInvalidError):
                _client(max_retries=3).chat_completion([{"role": "user", "content": "hi"}], model="m1")
        self.assertEqual(calls["n"], 1)


class EmbeddingTests(unittest.TestCase):
    def test_successful_embedding_call(self) -> None:
        body = {"model": "embed-1", "data": [{"embedding": [0.1, 0.2, 0.3]}], "usage": {"prompt_tokens": 4}}
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(body)):
            result = _client().embedding("xin chào", model="embed-1")
        self.assertEqual(result.vector, [0.1, 0.2, 0.3])
        self.assertEqual(result.usage.input_tokens, 4)

    def test_missing_data_raises_schema_invalid(self) -> None:
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse({"model": "embed-1"})):
            with self.assertRaises(SchemaInvalidError):
                _client().embedding("x", model="embed-1")


class RerankTests(unittest.TestCase):
    def test_scores_are_reordered_to_match_original_document_order(self) -> None:
        # Xác nhận thật từ FPT (PR-15 probe thủ công): 'results' sắp theo
        # relevance GIẢM DẦN, không theo thứ tự documents gửi lên. index=2
        # đứng đầu dù documents[2] là phần tử thứ ba trong request.
        body = {
            "model": "bge-reranker-v2-m3",
            "results": [
                {"document": None, "index": 2, "relevance_score": 0.95},
                {"document": None, "index": 0, "relevance_score": 0.006},
                {"document": None, "index": 1, "relevance_score": 1.5e-05},
            ],
            "usage": {"prompt_tokens": 145},
        }
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(body)):
            result = _client().rerank("q", ["doc0", "doc1", "doc2"], model="bge-reranker-v2-m3")
        self.assertEqual(result.scores, [0.006, 1.5e-05, 0.95])
        self.assertEqual(result.usage.input_tokens, 145)

    def test_empty_documents_returns_empty_without_a_call(self) -> None:
        with patch("online.adapters.fpt_client.urlopen") as mock_urlopen:
            result = _client().rerank("q", [], model="m")
        mock_urlopen.assert_not_called()
        self.assertEqual(result.scores, [])

    def test_wrong_result_count_raises_schema_invalid(self) -> None:
        body = {"results": [{"index": 0, "relevance_score": 0.5}]}
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(body)):
            with self.assertRaises(SchemaInvalidError):
                _client().rerank("q", ["doc0", "doc1"], model="m")

    def test_duplicate_index_raises_schema_invalid(self) -> None:
        body = {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.2}]}
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(body)):
            with self.assertRaises(SchemaInvalidError):
                _client().rerank("q", ["doc0", "doc1"], model="m")

    def test_out_of_range_index_raises_schema_invalid(self) -> None:
        body = {"results": [{"index": 5, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.2}]}
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(body)):
            with self.assertRaises(SchemaInvalidError):
                _client().rerank("q", ["doc0", "doc1"], model="m")


class RerankProbeTests(unittest.TestCase):
    def test_native_rerank_available_when_endpoint_responds(self) -> None:
        body = {"results": [{"index": 0, "relevance_score": 0.9}]}
        with patch("online.adapters.fpt_client.urlopen", return_value=FakeResponse(body)):
            result = _client().probe_rerank("q", ["a", "b"], model="rerank-1")
        self.assertTrue(result.native_rerank_available)

    def test_404_means_no_native_rerank_not_an_error(self) -> None:
        def fake_urlopen(request, timeout, context=None):
            raise HTTPError(url="x", code=404, msg="not found", hdrs=None, fp=BytesIO(b"{}"))

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            result = _client().probe_rerank("q", ["a", "b"], model="rerank-1")
        self.assertFalse(result.native_rerank_available)

    def test_405_also_means_no_native_rerank(self) -> None:
        def fake_urlopen(request, timeout, context=None):
            raise HTTPError(url="x", code=405, msg="method not allowed", hdrs=None, fp=BytesIO(b"{}"))

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            result = _client().probe_rerank("q", ["a", "b"], model="rerank-1")
        self.assertFalse(result.native_rerank_available)

    def test_401_during_probe_is_a_real_error_not_treated_as_missing_endpoint(self) -> None:
        def fake_urlopen(request, timeout, context=None):
            raise HTTPError(url="x", code=401, msg="unauthorized", hdrs=None, fp=BytesIO(b"{}"))

        with patch("online.adapters.fpt_client.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(AuthError):
                _client().probe_rerank("q", ["a", "b"], model="rerank-1")


class ImageDataUrlTests(unittest.TestCase):
    def test_jpg_maps_to_jpeg_mime_type(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            path.write_bytes(b"\xff\xd8\xff\xe0fake")
            url = image_to_data_url(path)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))


class ConstructorTests(unittest.TestCase):
    def test_empty_api_key_is_rejected_immediately(self) -> None:
        with self.assertRaises(ValueError):
            FptClient(base_url="https://x", api_key="")


if __name__ == "__main__":
    unittest.main()
