from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.add_groundtruth import append_entry, build_entry, _next_query_id


class AddGroundtruthTests(unittest.TestCase):
    def test_build_entry_with_interval_and_scene_ids(self) -> None:
        entry = build_entry(
            query="Người áo vàng đang bơi",
            video_id="L01_V001",
            start_sec=12.5,
            end_sec=18.0,
            scene_ids=["L01_V001_S0002"],
            query_id="q99",
        )
        self.assertEqual(entry, {
            "query_id": "q99",
            "query": "Người áo vàng đang bơi",
            "video_id": "L01_V001",
            "start_sec": 12.5,
            "end_sec": 18.0,
            "scene_ids": ["L01_V001_S0002"],
        })

    def test_build_entry_rejects_bad_video_id(self) -> None:
        with self.assertRaises(ValueError):
            build_entry(query="x", video_id="not-an-id", start_sec=None, end_sec=None, scene_ids=[], query_id="q1")

    def test_build_entry_rejects_end_before_start(self) -> None:
        with self.assertRaises(ValueError):
            build_entry(query="x", video_id="L01_V001", start_sec=10.0, end_sec=5.0, scene_ids=[], query_id="q1")

    def test_next_query_id_increments_past_existing(self) -> None:
        existing = [{"query_id": "q1"}, {"query_id": "q3"}, {"query_id": "not-a-number"}]
        self.assertEqual(_next_query_id(existing), "q4")
        self.assertEqual(_next_query_id([]), "q1")

    def test_append_entry_writes_valid_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.jsonl"
            entry = build_entry(query="x", video_id="L01_V001", start_sec=None, end_sec=None, scene_ids=[], query_id="q1")
            append_entry(path, entry)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), entry)


if __name__ == "__main__":
    unittest.main()
