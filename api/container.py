"""Composition root: the only module that selects concrete infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from online.adapters.bm25 import LexicalRetriever
from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import HashingTextEncoder, RemoteTextEncoder
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.vector_stores import InMemoryVectorStore, QdrantVectorStore
from online.config import Settings
from online.services.search import SearchService
from online.services.vqa import VQAService


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    repository: JsonlSceneRepository
    search_service: SearchService
    vqa_service: VQAService


async def build_container(settings: Settings) -> AppContainer:
    repository = await JsonlSceneRepository.load(settings.metadata_jsonl)
    lexical = [
        await LexicalRetriever.build(field, repository)
        for field in ("caption", "ocr", "asr", "keyword")
    ]

    if settings.backend == "qdrant":
        encoder = RemoteTextEncoder(
            settings.embedding_url or "", settings.request_timeout_sec
        )
        vector_store = QdrantVectorStore(
            settings.qdrant_url or "",
            settings.qdrant_scene_collection,
            settings.qdrant_vector_name,
            api_key=settings.qdrant_api_key,
            timeout_sec=settings.request_timeout_sec,
        )
    else:
        encoder = HashingTextEncoder()
        rows = []
        for scene in await repository.all():
            search_text = " ".join(scene.captions + scene.keywords)
            rows.append(
                (
                    scene.scene_id,
                    scene.video_id,
                    await encoder.encode(search_text),
                    {
                        "scene_id": scene.scene_id,
                        "video_id": scene.video_id,
                        "scene_idx": scene.scene_idx,
                        "start_sec": scene.start_sec,
                        "end_sec": scene.end_sec,
                        "has_ocr": bool(scene.ocr_texts),
                        "has_asr": bool(scene.asr_texts),
                    },
                )
            )
        vector_store = InMemoryVectorStore(rows)

    retrievers = [DenseRetriever(encoder, vector_store), *lexical]
    search_service = SearchService(
        repository,
        retrievers,
        candidate_limit=settings.candidate_limit,
        rrf_k=settings.rrf_k,
    )
    return AppContainer(
        settings=settings,
        repository=repository,
        search_service=search_service,
        vqa_service=VQAService(search_service, repository),
    )

