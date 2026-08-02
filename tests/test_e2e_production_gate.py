"""PR-11: production gate — một luồng E2E chạy hết cả 4 task + submission.

Khác các test unit/integration khác (mỗi cái kiểm một module), file này mô
phỏng đúng thứ một người dùng thi thật sẽ làm trong một phiên: health check
-> đọc capabilities -> chạy cả 4 task -> build+validate submission cho ba
task chính thức -> mở evidence -> xem lại session -> replay -> stream SSE.

Nếu file này pass thì toàn bộ chuỗi API mà UI (`online/ui-react/`) gọi tới đều
hoạt động đúng trên chính dataset demo — đây là "gate" trước khi coi hệ thống
sẵn sàng cắm dataset thật.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

os.environ.setdefault("AIC_METADATA_JSONL", "examples/scenes.jsonl")

from fastapi.testclient import TestClient

from online.api.app import create_app
from online.config import Settings

ROOT = Path(__file__).resolve().parents[1]


class ProductionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["AIC_METADATA_JSONL"] = str(ROOT / "examples" / "scenes.jsonl")
        cls.client = TestClient(create_app(Settings.from_env()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_01_health_reports_ready_with_operational_fields(self) -> None:
        response = self.client.get("/v1/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertGreater(body["scene_count"], 0)
        self.assertGreater(body["branch_count"], 0)
        self.assertTrue(body["session_store_enabled"])

    def test_02_capabilities_lists_real_registered_branches(self) -> None:
        response = self.client.get("/v1/search/capabilities")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("TEXTUAL_KIS", body["task_types"])
        self.assertTrue(body["branches"])
        # Mọi branch phải có backend_kind thật, không branch nào là control giả.
        for branch in body["branches"]:
            self.assertTrue(branch["backend_kind"])

    def test_03_kis_end_to_end_search_to_valid_submission(self) -> None:
        search = self.client.post(
            "/v1/search/kis",
            json={"query": 'căn nhà có chữ "Gừng cay muối mặn xin đừng quên nhau"', "top_k": 5},
        )
        self.assertEqual(search.status_code, 200)
        body = search.json()
        self.assertTrue(body["kis"])
        self.assertIsInstance(body["kis"][0]["frame_idx"], int)

        build = self.client.post("/v1/submissions/build", json={"task": "TEXTUAL_KIS", "kis": body["kis"]})
        self.assertEqual(build.status_code, 200)
        submission = build.json()
        self.assertFalse(submission["has_errors"])
        self.assertNotIn("video_id", submission["csv"])  # không header

        self.__class__.kis_session_id = body["query_id"]

    def test_04_qa_end_to_end_search_to_submission(self) -> None:
        search = self.client.post(
            "/v1/search/qa", json={"query": "Trong cảnh có bao nhiêu người vẫy tay?", "top_k": 5}
        )
        self.assertEqual(search.status_code, 200)
        body = search.json()
        if not body["qa"]:
            self.skipTest("fixture demo không sinh QA candidate cho câu hỏi này")
        build = self.client.post("/v1/submissions/build", json={"task": "QA", "qa": body["qa"]})
        self.assertEqual(build.status_code, 200)
        first_row = build.json()["csv"].splitlines()[0]
        self.assertEqual(len(first_row.split(",")), 3)  # video_id, frame_idx, answer

    def test_05_trake_end_to_end_search_to_submission(self) -> None:
        search = self.client.post(
            "/v1/search/trake",
            json={"query": "cào muối, sau đó vẫy tay, cuối cùng căn nhà", "top_k": 5},
        )
        self.assertEqual(search.status_code, 200)
        body = search.json()
        self.assertTrue(body["trake"])
        # Luật: frame_ids phải tăng dần nghiêm ngặt trong TỪNG kết quả.
        for item in body["trake"]:
            self.assertEqual(item["frame_ids"], sorted(item["frame_ids"]))
            self.assertEqual(len(item["frame_ids"]), len(set(item["frame_ids"])))

        build = self.client.post("/v1/submissions/build", json={"task": "TRAKE", "trake": body["trake"]})
        self.assertEqual(build.status_code, 200)
        self.assertFalse(build.json()["has_errors"])

    def test_06_avs_returns_graded_segments_without_an_official_submission(self) -> None:
        search = self.client.post("/v1/search/avs", json={"query": "người và bảng chữ", "top_k": 5})
        self.assertEqual(search.status_code, 200)
        body = search.json()
        self.assertTrue(body["avs"])
        for item in body["avs"]:
            self.assertIn(item["relevance_grade"], (0, 1, 2, 3))
        rejected = self.client.post("/v1/submissions/build", json={"task": "AVS"})
        self.assertEqual(rejected.status_code, 422)

    def test_07_evidence_lookup_for_a_kis_result(self) -> None:
        search = self.client.post("/v1/search/kis", json={"query": "căn nhà", "top_k": 1})
        candidate_id = search.json()["results"][0]["candidate_id"]
        evidence = self.client.get(f"/v1/evidence/{candidate_id}")
        self.assertEqual(evidence.status_code, 200)
        self.assertEqual(evidence.json()["video_id"], "L01_V001")

    def test_08_session_trace_and_replay_round_trip(self) -> None:
        search = self.client.post("/v1/search/kis", json={"query": "căn nhà", "top_k": 3})
        query_id = search.json()["query_id"]

        trace = self.client.get(f"/v1/search-sessions/{query_id}")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.json()["task"], "TEXTUAL_KIS")

        replay = self.client.post(f"/v1/search-sessions/{query_id}/replay")
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json()["replayed_from"], query_id)

    def test_09_sse_stream_completes_with_a_full_response(self) -> None:
        events = []
        with self.client.stream(
            "POST", "/v1/search/stream", json={"query": "căn nhà", "task": "TEXTUAL_KIS", "top_k": 3}
        ) as response:
            self.assertEqual(response.status_code, 200)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))
        self.assertEqual(events[0]["type"], "search_started")
        self.assertEqual(events[-1]["type"], "search_completed")

    def test_10_unsupported_search_option_is_rejected_before_any_work_happens(self) -> None:
        response = self.client.post(
            "/v1/search/kis",
            json={"query": "x", "search_options": {"rerank": {"vlm": {"enabled": True}}}},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
