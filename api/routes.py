"""Versioned HTTP routes; business logic remains in application services."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

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
    return {
        "status": "ok",
        "backend": container.settings.backend,
        "scene_count": len(scenes),
    }


async def _search_with_task(
    request: SearchRequest, task: TaskType, container: AppContainer
) -> SearchResponse:
    return await container.search_service.search(request.model_copy(update={"task": task}))


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
    return await container.vqa_service.answer(request)


@router.get("/scenes/{scene_id}")
async def get_scene(scene_id: str, container: Container) -> dict:
    scene = await container.repository.get(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail="scene not found")
    return scene.model_dump(mode="json")

