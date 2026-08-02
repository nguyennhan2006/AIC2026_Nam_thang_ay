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
from online.errors import TaskConflictError


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
    """Chạy search với task của endpoint.

    Body được phép bỏ trống `task`. Nếu body khai báo task KHÁC path thì đó là
    lỗi của client — trước PR-01 request kiểu đó bị ghi đè im lặng nên người
    gọi không bao giờ biết mình đang chạy sai task.
    """

    if request.task is not None and request.task != task:
        raise TaskConflictError(body_task=request.task.value, path_task=task.value)
    response = await container.search_service.search(
        request.model_copy(update={"task": task})
    )
    if task == TaskType.TRAKE:
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
    return await _search_with_task(request, TaskType.TEXTUAL_KIS, container)


@router.post("/search/qa", response_model=SearchResponse)
async def search_qa(request: SearchRequest, container: Container) -> SearchResponse:
    return await _search_with_task(request, TaskType.QA, container)


@router.post("/search/avs", response_model=SearchResponse)
async def search_avs(request: SearchRequest, container: Container) -> SearchResponse:
    return await _search_with_task(request, TaskType.AVS, container)


@router.post("/search/trake", response_model=SearchResponse)
async def search_trake(request: SearchRequest, container: Container) -> SearchResponse:
    return await _search_with_task(request, TaskType.TRAKE, container)


@router.post("/search/sequence", response_model=SearchResponse, deprecated=True)
async def search_sequence(request: SearchRequest, container: Container) -> SearchResponse:
    """Alias cũ của `/search/trake`; giữ để client hiện tại không gãy."""

    return await _search_with_task(request, TaskType.TRAKE, container)


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
    """Capability đã introspect thật — mọi branch ở đây đang thực sự đăng ký
    trong `container.search_service.registry`, không phải danh sách mong muốn.

    `branch_id`/`execution_ids` trả về ở đây chính là chuỗi mà cấu hình
    `search_options.branches` dùng, và `supported_controls` chỉ liệt kê
    control branch thật sự đọc — UI không được vẽ control ngoài danh sách này.
    """

    branches = [
        item.model_dump(mode="json")
        for item in container.search_service.registry.capabilities()
    ]
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
