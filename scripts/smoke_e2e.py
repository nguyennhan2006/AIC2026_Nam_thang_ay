from __future__ import annotations

import asyncio
from pathlib import Path

from datasection.exporter import verify_export
from online.adapters.bm25 import LexicalRetriever
from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest, TaskType
from online.services.search import SearchService
from scripts.seed_demo import main as seed


async def run() -> None:
    root = Path(__file__).resolve().parents[1]
    seed()
    verify_export(root / "storage/exports")
    repository = await JsonlSceneRepository.load(root / "storage/exports/scenes.jsonl")
    retrievers = [await LexicalRetriever.build(field, repository) for field in ("caption", "ocr", "asr", "keyword")]
    service = SearchService(repository, retrievers, candidate_limit=20)
    response = await service.search(SearchRequest(query='"Gừng cay muối mặn"', task=TaskType.KIS, top_k=1))
    assert response.results[0].scene_id == "L01_V001_S0002"
    print("E2E OK:", response.results[0].scene_id, response.results[0].best_keyframe_id)


if __name__ == "__main__":
    asyncio.run(run())
