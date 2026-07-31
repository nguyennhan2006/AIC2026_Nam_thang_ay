"""Versioned HTTP routes; business logic remains in application services."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pathlib import Path

from online.api.container import AppContainer
from online.domain.models import (
    SearchRequest,
    SearchResponse,
    TaskType,
    VQARequest,
    VQAResponse,
)


router = APIRouter(prefix="/v1")


def get_container(request: Request) -> AppContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="application is not ready")
    return container


Container = Annotated[AppContainer, Depends(get_container)]


@router.get("/health")
async def health(container: Container) -> dict:
    scenes = await container.repository.all()
    vector_ready = await container.vector_store.health()
    if not vector_ready:
        raise HTTPException(status_code=503, detail="vector backend is not ready")
    return {
        "status": "ok",
        "backend": container.settings.backend,
        "scene_count": len(scenes),
        "dataset": str(container.settings.metadata_jsonl),
    }


async def _search_with_task(
    request: SearchRequest, task: TaskType, container: AppContainer
) -> SearchResponse:
    response = await container.search_service.search(
        request.model_copy(update={"task": task})
    )
    if task == TaskType.SEQUENCE:
        if not response.sequences:
            raise HTTPException(
                status_code=404,
                detail=f"no ordered sequence matched the query: {request.query!r}",
            )
    elif not response.results:
        raise HTTPException(
            status_code=404,
            detail=f"no scene matched the query: {request.query!r}",
        )
    return response


@router.post("/search/kis", response_model=SearchResponse)
async def search_kis(request: SearchRequest, container: Container) -> SearchResponse:
    return await _search_with_task(request, TaskType.KIS, container)


@router.post("/search/avs", response_model=SearchResponse)
async def search_avs(request: SearchRequest, container: Container) -> SearchResponse:
    return await _search_with_task(request, TaskType.AVS, container)


@router.post("/search/sequence", response_model=SearchResponse)
async def search_sequence(request: SearchRequest, container: Container) -> SearchResponse:
    return await _search_with_task(request, TaskType.SEQUENCE, container)


@router.post("/vqa", response_model=VQAResponse)
async def answer_vqa(request: VQARequest, container: Container) -> VQAResponse:
    response = await container.vqa_service.answer(request)
    if not response.evidence:
        raise HTTPException(
            status_code=404,
            detail=f"no evidence found for the question: {request.question!r}",
        )
    return response


@router.get("/scenes/{scene_id}")
async def get_scene(scene_id: str, container: Container) -> dict:
    scene = await container.repository.get(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="scene not found")
    return scene.model_dump(mode="json")


@router.get("/events/{event_id}")
async def get_event(event_id: str, container: Container) -> dict:
    if container.event_repository is None:
        raise HTTPException(status_code=404, detail="no event data available for this dataset")
    event = await container.event_repository.get(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="event not found")
    return event.model_dump(mode="json")


@router.get("/events/{event_id}/neighbors")
async def get_event_neighbors(event_id: str, container: Container) -> dict:
    if container.event_repository is None:
        raise HTTPException(status_code=404, detail="no event data available for this dataset")
    event = await container.event_repository.get(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="event not found")
    previous = await container.event_repository.get(event.previous_event_id) if event.previous_event_id else None
    next_ = await container.event_repository.get(event.next_event_id) if event.next_event_id else None
    return {
        "previous": previous.model_dump(mode="json") if previous else None,
        "next": next_.model_dump(mode="json") if next_ else None,
    }


@router.get("/search/capabilities")
async def search_capabilities(container: Container) -> dict:
    """Real, introspected capabilities — every branch listed here is actually
    registered in `container.search_service.retrievers` right now, not an
    aspirational/hardcoded list (Search Mixing Console clean-code rule #9)."""

    branches = []
    for retriever in container.search_service.retrievers:
        modality = getattr(retriever, "modality", None)
        branches.append({
            "branch_id": retriever.name,
            "modality": modality.value if modality is not None else None,
            "status": "ready",
        })
    return {
        "task_types": [item.value for item in TaskType],
        "branches": branches,
        # weighted_sum/max_score reuse rank-derived contribution, not a properly
        # score-normalized weighted sum — see online/services/fusion.py docstring.
        "fusion_methods": ["rrf", "weighted_sum", "max_score", "intersection", "union"],
        "rerank": {
            "rules": container.search_service.rule_config is not None,
            # BGE/Qwen3-VL rerank need a remote model server that does not exist
            # yet — see docs/14_TECHNICAL_PREPARATION.md Phase 3.
            "text": False,
            "vlm": False,
        },
        "events_available": container.event_repository is not None,
    }


@router.get("/media/{artifact_path:path}")
async def media(artifact_path: str, container: Container) -> FileResponse:
    root = container.settings.data_root.resolve()
    target = (root / artifact_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(400, "invalid media path")
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".svg", ".mp4", ".webm", ".mkv"}
    if target.suffix.casefold() not in allowed:
        raise HTTPException(403, "artifact type is not public media")
    relative = target.relative_to(root)
    if not (relative.parts[:1] == ("processed",) or relative.parts[:2] == ("raw", "videos")):
        raise HTTPException(403, "artifact is outside public media roots")
    if not target.is_file():
        raise HTTPException(404, "media not found")
    return FileResponse(target)
