"""Composition root: the only module that selects concrete infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

from online.adapters.bm25 import LexicalRetriever
from online.adapters.color_search import ColorSearchRetriever
from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import HashingTextEncoder, RemoteTextEncoder
from online.adapters.event_search import EventSearchRetriever, JsonlEventRepository
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.ocr_fuzzy import OcrFuzzyRetriever
from online.adapters.vector_stores import InMemoryVectorStore, QdrantVectorStore
from online.config import Settings
from online.services.query_expansion import QueryExpansionRetriever
from online.services.query_prep import PreparedQueryPlanner
from online.services.rules import RuleConfig
from online.services.search import SearchService
from online.services.vqa import VQAService


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    repository: JsonlSceneRepository
    search_service: SearchService
    vqa_service: VQAService
    vector_store: object
    event_repository: JsonlEventRepository | None = None


async def build_container(settings: Settings) -> AppContainer:
    repository = await JsonlSceneRepository.load(settings.metadata_jsonl)
    lexical = []
    for field in ("caption", "ocr", "asr", "keyword"):
        retriever = await LexicalRetriever.build(field, repository)
        # Phương án K: chỉ wrap caption/keyword — OCR/ASR phải giữ nguyên văn.
        if settings.enable_expansion and field in ("caption", "keyword"):
            retriever = QueryExpansionRetriever(retriever)
        lexical.append(retriever)

    if settings.backend == "qdrant":
        encoder = RemoteTextEncoder(
            settings.embedding_url or "", settings.request_timeout_sec, settings.embedding_api_key
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
    if settings.enable_ocr_fuzzy:
        retrievers.append(await OcrFuzzyRetriever.build(repository))
    if settings.enable_object_search:
        retrievers.append(await LexicalRetriever.build("object", repository))
    if settings.enable_action_search:
        retrievers.append(await LexicalRetriever.build("action", repository))
    if settings.enable_color_search:
        retrievers.append(await ColorSearchRetriever.build(repository))

    events_path = settings.metadata_jsonl.with_name("events.jsonl")
    event_repository = await JsonlEventRepository.load(events_path) if events_path.exists() else None
    if settings.enable_event_search:
        if event_repository is None:
            raise ValueError(
                f"AIC_ENABLE_EVENT_SEARCH is set but {events_path} does not exist — "
                "run the offline pipeline/exporter (which now always writes events.jsonl) first"
            )
        retrievers.append(await EventSearchRetriever.build(event_repository))

    search_service = SearchService(
        repository,
        retrievers,
        planner=PreparedQueryPlanner() if settings.enable_query_prep else None,
        candidate_limit=settings.candidate_limit,
        rrf_k=settings.rrf_k,
        rule_config=RuleConfig() if settings.enable_rules else None,
    )
    return AppContainer(
        settings=settings,
        repository=repository,
        search_service=search_service,
        vqa_service=VQAService(search_service, repository),
        vector_store=vector_store,
        event_repository=event_repository,
    )
