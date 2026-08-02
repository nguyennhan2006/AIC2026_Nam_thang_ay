"""PR-08: endpoint /v1/submissions/* qua HTTP thật (không mock container)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("AIC_METADATA_JSONL", "examples/scenes.jsonl")

from fastapi.testclient import TestClient

from online.api.app import create_app
from online.config import Settings

ROOT = Path(__file__).resolve().parents[1]


class SubmissionRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["AIC_METADATA_JSONL"] = str(ROOT / "examples" / "scenes.jsonl")
        cls.client = TestClient(create_app(Settings.from_env()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def _kis_results(self) -> list[dict]:
        response = self.client.post(
            "/v1/search/kis",
            json={
                "query": 'căn nhà có chữ "Gừng cay muối mặn xin đừng quên nhau"',
                "top_k": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["kis"]

    def test_build_kis_returns_csv_with_no_header(self) -> None:
        kis = self._kis_results()
        response = self.client.post("/v1/submissions/build", json={"task": "TEXTUAL_KIS", "kis": kis})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["has_errors"])
        first_line = body["csv"].splitlines()[0]
        self.assertRegex(first_line, r"^L01_V001,\d+$")
        self.assertNotIn("video_id", body["csv"])  # không có header

    def test_build_kis_flags_frame_beyond_video_length(self) -> None:
        kis = self._kis_results()
        bad = [dict(kis[0], frame_idx=999999)]
        response = self.client.post("/v1/submissions/build", json={"task": "TEXTUAL_KIS", "kis": bad})
        body = response.json()
        self.assertTrue(body["has_errors"])
        self.assertTrue(any(item["code"] == "frame_out_of_bounds" for item in body["issues"]))

    def test_validate_endpoint_matches_build_issues(self) -> None:
        kis = self._kis_results()
        bad = [dict(kis[0], frame_idx=999999)]
        build = self.client.post("/v1/submissions/build", json={"task": "TEXTUAL_KIS", "kis": bad})
        validate = self.client.post("/v1/submissions/validate", json={"task": "TEXTUAL_KIS", "kis": bad})
        self.assertEqual(
            {item["code"] for item in build.json()["issues"]},
            {item["code"] for item in validate.json()},
        )

    def test_evaluate_local_scores_a_correct_kis_submission(self) -> None:
        kis = self._kis_results()
        response = self.client.post("/v1/submissions/evaluate-local", json={
            "task": "TEXTUAL_KIS", "kis": kis, "video_id": "L01_V001",
            "intervals": [{"start_frame": 600, "end_frame": 779}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 1.0)

    def test_evaluate_local_scores_zero_for_wrong_video(self) -> None:
        kis = self._kis_results()
        response = self.client.post("/v1/submissions/evaluate-local", json={
            "task": "TEXTUAL_KIS", "kis": kis, "video_id": "L99_V999",
            "intervals": [{"start_frame": 0, "end_frame": 100}],
        })
        self.assertEqual(response.json()["score"], 0.0)

    def test_avs_task_is_rejected_for_submission_build(self) -> None:
        response = self.client.post("/v1/submissions/build", json={"task": "AVS"})
        self.assertEqual(response.status_code, 422)

    def test_qa_csv_has_three_columns(self) -> None:
        response = self.client.post(
            "/v1/search/qa", json={"query": "Trong cảnh có bao nhiêu người vẫy tay?", "top_k": 3}
        )
        qa = response.json()["qa"]
        if not qa:
            self.skipTest("fixture không sinh được QA candidate nào cho câu hỏi này")
        build = self.client.post("/v1/submissions/build", json={"task": "QA", "qa": qa})
        first_line = build.json()["csv"].splitlines()[0]
        self.assertEqual(len(first_line.split(",")), 3)

    def test_trake_csv_has_video_plus_all_frame_ids(self) -> None:
        response = self.client.post("/v1/search/trake", json={
            "query": "cào muối, sau đó vẫy tay, cuối cùng căn nhà", "top_k": 3,
        })
        trake = response.json()["trake"]
        self.assertTrue(trake)
        build = self.client.post("/v1/submissions/build", json={"task": "TRAKE", "trake": trake})
        first_line = build.json()["csv"].splitlines()[0]
        self.assertEqual(first_line.split(",")[0], "L01_V001")
        self.assertGreaterEqual(len(first_line.split(",")), 3)


if __name__ == "__main__":
    unittest.main()
