"""PR-01 contract tests: taxonomy chuẩn + `frame_idx` xuyên suốt.

Hai lỗi mà các test này chặn vĩnh viễn:

1. Projection online làm mất `frame_idx` (lỗi cũ của `project_scene`), khiến
   kết quả search không nộp bài được.
2. Route ghi đè `task` của body bằng task của path một cách im lặng, khiến
   client chạy sai task mà không biết.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import unittest

from online.adapters.bm25 import LexicalRetriever
from online.adapters.json_metadata import JsonlSceneRepository, project_scene
from online.domain.models import SearchRequest, TaskType
from online.domain.tasks import normalize_task
from online.errors import TaskConflictError
from online.services.search import SearchService

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JSONL = ROOT / "examples" / "scenes.jsonl"


def run(coro):
    return asyncio.run(coro)


class TaskTaxonomyTests(unittest.TestCase):
    def test_canonical_names_are_the_competition_names(self) -> None:
        self.assertEqual(
            [item.value for item in TaskType],
            ["TEXTUAL_KIS", "QA", "TRAKE", "AVS"],
        )

    def test_legacy_aliases_normalize_to_canonical(self) -> None:
        for alias, expected in (
            ("kis", TaskType.TEXTUAL_KIS),
            ("KIS", TaskType.TEXTUAL_KIS),
            ("vqa", TaskType.QA),
            ("Q&A", TaskType.QA),
            ("sequence", TaskType.TRAKE),
            ("temporal", TaskType.TRAKE),
            ("avs", TaskType.AVS),
        ):
            with self.subTest(alias=alias):
                self.assertEqual(normalize_task(alias), expected)

    def test_unknown_task_is_rejected_not_guessed(self) -> None:
        with self.assertRaises(ValueError):
            normalize_task("known_item_search")

    def test_request_accepts_alias_at_the_api_boundary(self) -> None:
        self.assertEqual(
            SearchRequest(query="x", task="sequence").task, TaskType.TRAKE
        )
        # Không khai báo task là hợp lệ: endpoint sẽ quyết định.
        self.assertIsNone(SearchRequest(query="x").task)

    def test_body_task_conflicting_with_path_task_is_an_error(self) -> None:
        from online.api.routes import _search_with_task

        request = SearchRequest(query="x", task=TaskType.QA)
        with self.assertRaises(TaskConflictError) as ctx:
            run(_search_with_task(request, TaskType.TEXTUAL_KIS, container=None))
        self.assertIn("QA", str(ctx.exception))
        self.assertIn("TEXTUAL_KIS", str(ctx.exception))


class FrameMappingTests(unittest.TestCase):
    def test_projection_keeps_frame_idx_and_derives_boundary_distance(self) -> None:
        raw = json.loads(EXAMPLE_JSONL.read_text(encoding="utf-8").splitlines()[2])
        scene = project_scene(raw)
        self.assertEqual(scene.scene_id, "L01_V001_S0003")
        self.assertEqual(scene.start_frame, 600)
        self.assertEqual(scene.end_frame_exclusive, 780)
        frame = scene.keyframes[0]
        self.assertEqual(frame.frame_idx, 600)
        # frame nằm đúng biên trái của scene -> khoảng cách biên = 0.
        self.assertEqual(frame.boundary_distance_frames, 0)

    def test_frame_idx_round_trips_to_keyframe_id_and_image_path(self) -> None:
        repository = run(JsonlSceneRepository.load(EXAMPLE_JSONL))
        for scene in run(repository.all()):
            for frame in scene.keyframes:
                with self.subTest(keyframe=frame.keyframe_id):
                    self.assertEqual(
                        frame.keyframe_id, f"{scene.scene_id}_F{frame.frame_idx:06d}"
                    )
                    self.assertIn(f"{frame.frame_idx:06d}", frame.image_path)
                    self.assertTrue(
                        scene.start_frame
                        <= frame.frame_idx
                        < scene.end_frame_exclusive
                    )

    def test_every_search_hit_carries_a_submittable_frame_idx(self) -> None:
        async def scenario():
            repository = await JsonlSceneRepository.load(EXAMPLE_JSONL)
            retrievers = [
                await LexicalRetriever.build(field, repository)
                for field in ("caption", "ocr", "asr", "keyword")
            ]
            service = SearchService(repository, retrievers, candidate_limit=20)
            return await service.search(
                SearchRequest(query="đoàn người vẫy tay", top_k=5)
            )

        response = run(scenario())
        self.assertTrue(response.results)
        for hit in response.results:
            with self.subTest(scene=hit.scene_id):
                self.assertIsInstance(hit.best_frame_idx, int)
                self.assertTrue(
                    hit.start_frame <= hit.best_frame_idx < hit.end_frame_exclusive
                )
                self.assertEqual(hit.warnings, [])

    def test_scene_document_rejects_keyframe_outside_its_interval(self) -> None:
        raw = json.loads(EXAMPLE_JSONL.read_text(encoding="utf-8").splitlines()[0])
        raw["keyframes"][0]["frame_idx"] = raw["end_frame_exclusive"] + 5
        with self.assertRaises(ValueError):
            project_scene(raw)


if __name__ == "__main__":
    unittest.main()
