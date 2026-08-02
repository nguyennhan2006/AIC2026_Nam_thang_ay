"""PR-09: /v1/search thống nhất, /v1/search/stream (SSE), /v1/search-sessions/*
qua HTTP thật (TestClient, không mock container)."""

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


class UnifiedSearchRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["AIC_METADATA_JSONL"] = str(ROOT / "examples" / "scenes.jsonl")
        cls.client = TestClient(create_app(Settings.from_env()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def test_missing_task_is_rejected(self) -> None:
        response = self.client.post("/v1/search", json={"query": "căn nhà"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("task is required", response.json()["detail"])

    def test_unified_endpoint_matches_convenience_endpoint_behavior(self) -> None:
        query = 'căn nhà có chữ "Gừng cay muối mặn xin đừng quên nhau"'
        unified = self.client.post(
            "/v1/search", json={"query": query, "task": "TEXTUAL_KIS", "top_k": 3}
        )
        convenience = self.client.post("/v1/search/kis", json={"query": query, "top_k": 3})
        self.assertEqual(unified.status_code, 200)
        self.assertEqual(convenience.status_code, 200)
        self.assertEqual(unified.json()["kis"][0]["frame_idx"], convenience.json()["kis"][0]["frame_idx"])

    def test_body_task_conflicting_with_path_still_rejected_via_convenience(self) -> None:
        response = self.client.post("/v1/search/kis", json={"query": "x", "task": "QA"})
        self.assertEqual(response.status_code, 422)


class SearchSessionRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["AIC_METADATA_JSONL"] = str(ROOT / "examples" / "scenes.jsonl")
        cls.client = TestClient(create_app(Settings.from_env()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def _run_a_search(self) -> dict:
        response = self.client.post(
            "/v1/search/kis",
            json={"query": 'căn nhà có chữ "Gừng cay muối mặn xin đừng quên nhau"', "top_k": 3},
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_convenience_endpoint_search_is_also_recorded_as_a_session(self) -> None:
        body = self._run_a_search()
        trace = self.client.get(f"/v1/search-sessions/{body['query_id']}")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.json()["task"], "TEXTUAL_KIS")

    def test_unknown_session_is_404(self) -> None:
        response = self.client.get("/v1/search-sessions/no-such-session")
        self.assertEqual(response.status_code, 404)

    def test_replay_returns_a_new_session_id_and_links_back(self) -> None:
        body = self._run_a_search()
        replay = self.client.post(f"/v1/search-sessions/{body['query_id']}/replay")
        self.assertEqual(replay.status_code, 200)
        replay_body = replay.json()
        self.assertEqual(replay_body["replayed_from"], body["query_id"])
        self.assertNotEqual(replay_body["query_id"], body["query_id"])
        self.assertEqual(replay_body["kis"][0]["frame_idx"], body["kis"][0]["frame_idx"])

    def test_replay_of_unknown_session_is_404(self) -> None:
        response = self.client.post("/v1/search-sessions/no-such-session/replay")
        self.assertEqual(response.status_code, 404)


class SearchStreamRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["AIC_METADATA_JSONL"] = str(ROOT / "examples" / "scenes.jsonl")
        cls.client = TestClient(create_app(Settings.from_env()))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)

    def _events(self, body: dict) -> list[dict]:
        events = []
        with self.client.stream("POST", "/v1/search/stream", json=body) as response:
            self.assertEqual(response.status_code, 200)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))
        return events

    def test_stream_requires_task(self) -> None:
        response = self.client.post("/v1/search/stream", json={"query": "x"})
        self.assertEqual(response.status_code, 422)

    def test_stream_ends_with_search_completed_containing_the_response(self) -> None:
        events = self._events({"query": "căn nhà", "task": "TEXTUAL_KIS", "top_k": 3})
        self.assertEqual(events[0]["type"], "search_started")
        self.assertEqual(events[-1]["type"], "search_completed")
        self.assertIn("kis", events[-1]["response"])

    def test_stream_reports_unsupported_options_as_an_error_event(self) -> None:
        events = self._events({
            "query": "x", "task": "TEXTUAL_KIS",
            "search_options": {"branches": {"khong_ton_tai": {"weight": 2.0}}},
        })
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")


if __name__ == "__main__":
    unittest.main()
