"""Composition root: the only module that selects concrete infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from online.adapters.bm25 import LexicalRetriever
from online.adapters.color_search import ColorSearchRetriever
from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import HashingTextEncoder, LocalClipTextEncoder, RemoteTextEncoder
from online.adapters.event_search import EventSearchRetriever, JsonlEventRepository
from online.adapters.frame_vector_store import build_frame_vector_rows
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.ocr_fuzzy import OcrFuzzyRetriever
from online.adapters.rerank import BgeTextReranker, QwenVlReranker
from online.adapters.session_store import InMemorySessionStore
from online.adapters.vector_stores import InMemoryVectorStore, QdrantVectorStore
from online.config import Settings
from online.services.query_expansion import QueryExpansionRetriever
from online.services.query_prep import PreparedQueryPlanner
from online.services.evidence_builder import EvidenceBuilder
from online.services.rerank_pipeline import RerankPipeline
from online.services.rules import RuleConfig
from online.services.search import SearchService
from online.services.vqa import VQAService


def _read_dataset_version(metadata_jsonl: Path) -> str | None:
    """`build_id` của dataset_manifest.json — đủ để phân biệt hai lần export.

    Không lỗi nếu thiếu file (vd metadata trỏ thẳng tới một .jsonl không đi
    kèm manifest) — session trace khi đó chỉ ghi `dataset_version=None`.
    """

    manifest_path = metadata_jsonl.with_name("dataset_manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return manifest.get("build_id")


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

    local_frame_rows: list[tuple[str, str, list[float], dict]] = []
    has_real_embeddings = False
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
        local_frame_rows, has_real_embeddings = await build_frame_vector_rows(
            repository, settings.data_root
        )
        if has_real_embeddings:
            # Export có embedding thật (PR-13, scripts/embed_keyframes_local.py):
            # text encoder PHẢI cùng model CLIP để chung không gian embedding
            # với vector ảnh, nếu không cosine similarity vô nghĩa.
            encoder = LocalClipTextEncoder(
                settings.visual_embedding_model, revision=settings.visual_embedding_model_revision
            )
            vector_store = InMemoryVectorStore(local_frame_rows)
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
                            # start/end_frame đi kèm payload để candidate từ vector
                            # store cũng truy ngược được về khoảng frame, không phải
                            # chờ hydrate mới biết (PR-01 frame contract).
                            "start_frame": scene.start_frame,
                            "end_frame": scene.end_frame_exclusive - 1,
                            "start_sec": scene.start_sec,
                            "end_sec": scene.end_sec,
                            "has_ocr": bool(scene.ocr_texts),
                            "has_asr": bool(scene.asr_texts),
                        },
                    )
                )
            vector_store = InMemoryVectorStore(rows)

    # Backend local KHÔNG PHẢI lúc nào cũng dense visual thật: nếu export
    # chưa có embedding pack (PR-13 chưa chạy), HashingTextEncoder +
    # InMemoryVectorStore hash trên text caption — đăng ký đúng tên
    # `lexical_hash_fallback` để /capabilities không quảng cáo nhầm và
    # ablation không bị đọc sai (PR-03). Có embedding thật thì dùng cùng tên
    # `dense_visual` như nhánh qdrant vì bản chất giờ giống nhau (chỉ khác
    # backend lưu vector).
    if settings.backend == "qdrant" or has_real_embeddings:
        dense = DenseRetriever(
            encoder, vector_store, branch_id="dense_visual", backend_kind="vector"
        )
    else:
        dense = DenseRetriever(
            encoder,
            vector_store,
            branch_id="lexical_hash_fallback",
            backend_kind="lexical_fallback",
        )
    retrievers = [dense, *lexical]
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

    evidence_builder = EvidenceBuilder(
        repository,
        model_versions={
            key: value
            for key, value in (
                ("text_reranker", settings.rerank_text_model if settings.rerank_text_url else ""),
                ("vlm_reranker", settings.rerank_vlm_model if settings.rerank_vlm_url else ""),
                ("dense_backend", settings.backend),
            )
            if value
        },
    )
    rerank_pipeline = RerankPipeline(
        evidence_builder,
        text_reranker=(
            BgeTextReranker(
                settings.rerank_text_url,
                model_id=settings.rerank_text_model,
                timeout_sec=settings.request_timeout_sec,
                api_key=settings.rerank_api_key,
            )
            if settings.rerank_text_url
            else None
        ),
        vlm_reranker=(
            QwenVlReranker(
                settings.rerank_vlm_url,
                model_id=settings.rerank_vlm_model,
                timeout_sec=max(settings.request_timeout_sec, 30.0),
                api_key=settings.rerank_api_key,
            )
            if settings.rerank_vlm_url
            else None
        ),
    )
    search_service = SearchService(
        repository,
        retrievers,
        planner=PreparedQueryPlanner() if settings.enable_query_prep else None,
        candidate_limit=settings.candidate_limit,
        rrf_k=settings.rrf_k,
        rule_config=RuleConfig() if settings.enable_rules else None,
        rerank_pipeline=rerank_pipeline,
        evidence_builder=evidence_builder,
        # PR-09: mọi search đi qua SearchService.search() đều được ghi trace —
        # kể cả gọi qua endpoint convenience /search/kis, không chỉ /v1/search.
        session_store=InMemorySessionStore(),
        dataset_version=_read_dataset_version(settings.metadata_jsonl),
    )
    return AppContainer(
        settings=settings,
        repository=repository,
        search_service=search_service,
        vqa_service=VQAService(search_service, repository),
        vector_store=vector_store,
        event_repository=event_repository,
    )
