"""Versioned HTTP routes; business logic remains in application services."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pathlib import Path

from online.api.container import AppContainer
from online.domain.models import (
    Candidate,
    Modality,
    SearchRequest,
    SearchResponse,
    TaskType,
    VQARequest,
    VQAResponse,
)
from online.domain.drafts import (
    DraftListResponse,
    DraftSaveRequest,
    SubmissionDraft,
)
from online.domain.submission import (
    EvaluateLocalRequest,
    EvaluateLocalResponse,
    SubmissionBuildRequest,
    SubmissionBuildResponse,
    SubmissionIssue,
)
from online.competition.scorer import (
    KisGold,
    QaGold,
    TrakeGold,
    GoldInterval,
    score_kis,
    score_qa,
    score_trake,
)
from online.competition.submission_builder import (
    build_kis_submission,
    build_qa_submission,
    build_trake_submission,
    kis_to_csv,
    qa_to_csv,
    trake_to_csv,
)
from online.competition.submission_validator import (
    has_errors,
    validate_kis,
    validate_qa,
    validate_trake,
)
from online.errors import TaskConflictError
from online.services.capabilities import (
    UNSUPPORTED,
    UNSUPPORTED_BRANCH_CONTROLS,
    UnsupportedSearchOptionError,
    validate_search_options,
)


router = APIRouter(prefix="/v1")


async def get_container(request: Request) -> AppContainer:
    """Container, CHỜ nếu nó còn đang nạp ở luồng nền.

    Chờ chứ không 503: người bấm Tìm kiếm lúc server mới lên muốn truy vấn
    chạy khi hệ sẵn sàng, không muốn một lỗi phải tự bấm lại. Muốn biết tiến
    độ mà không phải chờ thì hỏi `GET /v1/startup`.
    """

    container = getattr(request.app.state, "container", None)
    if container is not None:
        return container
    boot = getattr(request.app.state, "boot", None)
    if boot is None or boot.task is None:
        raise HTTPException(status_code=503, detail="application is not ready")
    try:
        # `shield` là bắt buộc: `await` thẳng một Task thì client ngắt kết nối
        # giữa chừng sẽ HUỶ LUÔN việc nạp container, giết server cho tất cả
        # những người còn lại.
        return await asyncio.shield(boot.task)
    except asyncio.CancelledError:
        raise HTTPException(status_code=503, detail="server đang tắt") from None
    except Exception as exc:  # noqa: BLE001 - lỗi khởi động, trả nguyên văn
        raise HTTPException(
            status_code=503, detail=f"nạp container thất bại lúc khởi động: {exc}"
        ) from exc


Container = Annotated[AppContainer, Depends(get_container)]


@router.get("/startup")
async def startup(request: Request) -> dict:
    """Tiến độ khởi động — KHÔNG BAO GIỜ chờ, kể cả khi còn đang nạp.

    Tách khỏi `/v1/health` vì hai câu hỏi khác nhau: health hỏi "hệ có khoẻ
    không" (và phải chờ tới lúc trả lời được), startup hỏi "đã xong chưa" và
    phải trả lời tức thì để UI vẽ được thanh chờ thay vì một màn hình trắng.
    """

    boot = getattr(request.app.state, "boot", None)
    if boot is None:
        # Container gắn tay (test, script) — không có pha khởi động nào cả.
        ready = getattr(request.app.state, "container", None) is not None
        return {"status": "ready" if ready else "warming", "phase": "unknown",
                "elapsed_sec": 0.0, "error": None}
    return boot.snapshot()


@router.get("/health")
async def health(container: Container) -> dict:
    scenes = await container.repository.all()
    vector_ready = await container.vector_store.health()
    if not vector_ready:
        raise HTTPException(status_code=503, detail="vector backend is not ready")
    manifest = container.dataset_manifest or {}
    return {
        "status": "ok",
        "backend": container.settings.backend,
        "scene_count": len(scenes),
        "dataset": str(container.settings.metadata_jsonl),
        # PR-11: quan sát vận hành tối thiểu không cần Prometheus — build_id
        # của chính dataset đang phục vụ, để phân biệt "server cũ trỏ data
        # mới" khỏi "server mới trỏ data cũ" khi debug production.
        "dataset_version": container.search_service.dataset_version,
        "branch_count": len(container.search_service.retrievers),
        "session_store_enabled": container.search_service.session_store is not None,
        # UI competition studio — dataset stats cards (Scenes/Keyframes/Videos/
        # ASR segments). video_count/keyframe_count đọc từ dataset_manifest.json
        # (đã tính sẵn lúc offline assemble); None nếu export không kèm manifest
        # thay vì đoán bằng 0 — UI phải hiển thị "—" chứ không phải "0" sai lệch.
        "video_count": manifest.get("video_count"),
        "keyframe_count": manifest.get("keyframe_count"),
        "asr_segment_count": sum(len(scene.asr_texts) for scene in scenes),
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
    # Option chưa chạy thật -> 422 ngay, không nhận rồi lờ đi (PR-04).
    validate_search_options(
        request.search_options,
        container.search_service.registry.capabilities(),
        rerank_stages=container.search_service.rerank_pipeline.available_stages,
    )
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


@router.post("/search", response_model=SearchResponse)
async def unified_search(request: SearchRequest, container: Container) -> SearchResponse:
    """Endpoint thống nhất (PR-09) — convenience endpoint bên dưới chỉ là
    wrapper mỏng gọi `_search_with_task` với task đã biết trước từ path.

    `task` là bắt buộc ở đây (khác convenience endpoint, nơi path đã ngầm
    định task): không có path nào để suy ra, nên bỏ trống task là lỗi
    tường minh thay vì âm thầm rơi về TEXTUAL_KIS.
    """

    if request.task is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "task is required for POST /v1/search; set it explicitly or call "
                "a convenience endpoint (/v1/search/kis, /qa, /trake, /avs)"
            ),
        )
    return await _search_with_task(request, request.task, container)


@router.post("/search/stream")
async def search_stream(request: SearchRequest, container: Container):
    """SSE thật: mỗi sự kiện phát ra đúng lúc giai đoạn đó xong, không phải
    một loạt sự kiện giả lập sau khi search đã chạy xong từ lâu.

    Task bắt buộc như `/v1/search` (không có path để suy ra). Không validate
    trước cả stream — lỗi search_options được phát như một sự kiện `error`
    duy nhất, EventSource phía client vẫn đóng kết nối bình thường.
    """

    if request.task is None:
        raise HTTPException(status_code=422, detail="task is required for /v1/search/stream")
    try:
        validate_search_options(
            request.search_options,
            container.search_service.registry.capabilities(),
            rerank_stages=container.search_service.rerank_pipeline.available_stages,
        )
    except UnsupportedSearchOptionError as exc:
        # `except ... as exc` bị Python tự `del exc` khi ra khỏi block, nên
        # phải chụp message vào biến thường TRƯỚC khi định nghĩa generator —
        # generator chỉ thực sự chạy khi StreamingResponse duyệt nó, lúc đó
        # block except đã kết thúc từ lâu.
        message = str(exc)

        async def error_only():
            yield f"data: {json.dumps({'type': 'error', 'message': message}, ensure_ascii=False)}\n\n"
        return StreamingResponse(error_only(), media_type="text/event-stream")

    async def events():
        async for event in container.search_service.search_stream(request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/search-sessions/{session_id}")
async def get_search_session(session_id: str, container: Container) -> dict:
    """Trace của một lần search đã chạy — session được tạo NGẦM bởi chính
    lần search đó (không có endpoint "tạo session rỗng": search luôn đi
    trước, session luôn là dấu vết của nó)."""

    if container.search_service.session_store is None:
        raise HTTPException(status_code=404, detail="session store is not enabled")
    trace = await container.search_service.session_store.get(session_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return trace.model_dump(mode="json")


@router.post("/search-sessions/{session_id}/replay", response_model=SearchResponse)
async def replay_search_session(session_id: str, container: Container) -> SearchResponse:
    """Chạy lại đúng request đã lưu của `session_id` — session mới, có
    `replayed_from` trỏ về session gốc để so sánh hai lần chạy."""

    response = await container.search_service.replay(session_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
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


@router.get("/videos")
async def list_videos(container: Container) -> dict:
    """Metadata mọi video: fps, frame_count, duration, đường dẫn media.

    UI cần `fps` THẬT để quy đổi frame <-> giây. Đo trên corpus hiện tại:
    V001/V002 chạy 30 fps nhưng **V003 chạy 25 fps** — đoán 30 cho tất cả thì
    tua lệch 20% trên V003, và thanh kéo chỉnh frame bằng tay thành vô dụng.

    `media_available` phân biệt "video này không có trong dataset" với "có
    trong dataset nhưng thiếu file mp4" — V002/V003 hiện rơi vào vế sau, và UI
    phải nói được điều đó thay vì hiện một player 404.
    """

    root = container.settings.data_root.resolve()
    return {
        "videos": [
            {
                "video_id": item.video_id,
                "media_path": item.source_path,
                "fps": item.fps,
                "frame_count": item.frame_count,
                "duration_sec": item.duration_sec,
                "width": item.width,
                "height": item.height,
                "media_available": (root / item.source_path).is_file(),
            }
            for item in await container.repository.all_videos()
        ]
    }


@router.get("/videos/{video_id}/frames")
async def list_video_frames(video_id: str, container: Container) -> dict:
    """Mọi keyframe của một video: `frame_idx` + đường dẫn ảnh.

    Để tab chỉnh frame soát được CẢ những video thiếu file mp4. `storage/raw/videos/`
    hiện chỉ có `L21_V001.mp4`, nhưng ảnh keyframe thì đủ cho cả ba video — nên
    người chấm vẫn nhìn được nội dung, chỉ là ở mật độ keyframe thay vì mượt
    như video.

    Payload nhỏ (855 keyframe cho toàn corpus 3 video) nên trả một lần rồi tra
    tại chỗ, không cần endpoint tra từng frame.
    """

    scenes = await container.repository.all()
    frames = sorted(
        (
            {
                "frame_idx": frame.frame_idx,
                "image_path": frame.image_path,
                "timestamp_sec": frame.timestamp_sec,
                "scene_id": frame.scene_id,
            }
            for scene in scenes
            if scene.video_id == video_id
            for frame in scene.keyframes
        ),
        key=lambda item: item["frame_idx"],
    )
    if not frames:
        raise HTTPException(status_code=404, detail=f"no keyframes for video {video_id}")
    return {"video_id": video_id, "frames": frames}


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


@router.get("/evidence/{candidate_id}")
async def get_evidence(candidate_id: str, container: Container) -> dict:
    """Evidence pack đầy đủ cho một candidate.

    Candidate id ở tầng scene chính là `scene_id`; ở tầng frame là
    `keyframe_id`. Pack được dựng lazy nên endpoint này an toàn để UI gọi khi
    người dùng mở một kết quả ra xem.
    """

    scene_id = candidate_id.split("_F")[0] if "_F" in candidate_id else candidate_id
    document = await container.repository.get(scene_id)
    if document is None:
        raise HTTPException(status_code=404, detail=f"unknown candidate: {candidate_id}")
    frame_idx = None
    if "_F" in candidate_id:
        try:
            frame_idx = int(candidate_id.rsplit("_F", 1)[1])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid keyframe id") from exc
    candidate = Candidate(
        candidate_id=candidate_id,
        entity_type="frame" if frame_idx is not None else "scene",
        scene_id=document.scene_id,
        video_id=document.video_id,
        event_id=document.event_id,
        frame_idx=frame_idx,
        source="evidence_lookup",
        modality=Modality.VISUAL,
        raw_score=0.0,
        rank=1,
    )
    pack = await container.search_service.evidence_builder.build(candidate)
    if pack is None:
        raise HTTPException(status_code=404, detail=f"no evidence for {candidate_id}")
    return pack.model_dump(mode="json")


@router.get("/search/capabilities")
async def search_capabilities(container: Container) -> dict:
    """Capability đã introspect thật — mọi branch ở đây đang thực sự đăng ký
    trong `container.search_service.registry`, không phải danh sách mong muốn.

    `branch_id`/`execution_ids` trả về ở đây chính là chuỗi mà cấu hình
    `search_options.branches` dùng, và `supported_controls` chỉ liệt kê
    control branch thật sự đọc — UI không được vẽ control ngoài danh sách này.
    """

    branches = [
        {
            **item.model_dump(mode="json"),
            "default_weight": container.search_service.branch_weights.get(item.branch_id),
        }
        for item in container.search_service.registry.capabilities()
    ]
    return {
        "task_types": [item.value for item in TaskType],
        "branches": branches,
        # weighted_sum/max_score reuse rank-derived contribution, not a properly
        # score-normalized weighted sum — see online/services/fusion.py docstring.
        "fusion_methods": [
            "rrf", "weighted_sum", "max_score", "intersection", "union",
            "norm_sum", "norm_max", "margin_sum", "entropy_sum",
        ],
        # Option bị từ chối kèm lý do: UI hiện được "vì sao control này mờ đi"
        # thay vì để người dùng thử rồi ăn 422 mà không hiểu.
        "unsupported_options": {
            path: reason for path, (_value, reason) in sorted(UNSUPPORTED.items())
        }
        | {
            f"branches.*.{control}": reason
            for control, reason in sorted(UNSUPPORTED_BRANCH_CONTROLS.items())
        },
        # Introspect thật: tầng nào có model server thì báo True, không hardcode.
        "rerank": {
            "rules": container.search_service.rule_config is not None,
            **container.search_service.rerank_pipeline.available_stages,
        },
        "events_available": container.event_repository is not None,
    }


# ---------------------------------------------------------------------------
# Bản nháp sắp xếp — dùng chung cả đội (FB-003)
# ---------------------------------------------------------------------------
#
# Ở SERVER chứ không localStorage: cả đội trỏ vào cùng một backend, nên lưu ở
# đây là tự động thấy được của nhau. Nháp KHÔNG đi qua submission_validator —
# nó được phép còn dở, đó là mục đích của nó.


def _draft_store(container: AppContainer):
    store = container.draft_store
    if store is None:
        raise HTTPException(status_code=503, detail="kho bản nháp chưa được cấu hình")
    return store


@router.get("/submission-drafts", response_model=DraftListResponse)
async def list_drafts(container: Container) -> DraftListResponse:
    return DraftListResponse(drafts=await _draft_store(container).list())


@router.post("/submission-drafts", response_model=SubmissionDraft)
async def save_draft(request: DraftSaveRequest, container: Container) -> SubmissionDraft:
    if not request.name.strip():
        raise HTTPException(status_code=422, detail="bản nháp phải có tên để người khác tìm lại")
    return await _draft_store(container).save(request)


@router.delete("/submission-drafts/{draft_id}")
async def delete_draft(draft_id: str, container: Container) -> dict:
    if not await _draft_store(container).delete(draft_id):
        raise HTTPException(status_code=404, detail=f"không có bản nháp {draft_id!r}")
    return {"deleted": draft_id}


def _submission_lookup(container: AppContainer):
    async def lookup(video_id: str) -> int | None:
        return await container.repository.video_frame_count(video_id)

    return lookup


@router.post("/submissions/build", response_model=SubmissionBuildResponse)
async def build_submission(
    request: SubmissionBuildRequest, container: Container
) -> SubmissionBuildResponse:
    """Dựng CSV đúng format BTC + validate, từ kết quả một lần `/v1/search/*`.

    Không tự cắt hay tự sửa dòng sai — chỉ báo `issues`; người dùng quyết
    định sửa gì trước khi tải CSV xuống.
    """

    lookup = _submission_lookup(container)
    if request.task == TaskType.TEXTUAL_KIS:
        items = build_kis_submission(request.kis)
        issues = await validate_kis(items, frame_count=lookup)
        csv_text = kis_to_csv(items)
    elif request.task == TaskType.QA:
        items = build_qa_submission(request.qa)
        issues = await validate_qa(items, frame_count=lookup)
        csv_text = qa_to_csv(items)
    elif request.task == TaskType.TRAKE:
        items = build_trake_submission(request.trake)
        issues = await validate_trake(items, frame_count=lookup)
        csv_text = trake_to_csv(items)
    else:
        # AVS là task đánh giá NỘI BỘ — người dùng xác nhận 2026-08-06: vòng sơ
        # loại chỉ nộp KIS, QA, TRAKE. Đầu ra AVS là danh sách segment có thứ
        # hạng, lấy thẳng từ `/v1/search/avs`, không đi qua exporter CSV.
        #
        # Từng có một lần thử thêm CSV cho AVS ở đây, dựa trên docstring của
        # `online/domain/submission.py` và `docs/19` §14.1. Cả hai nguồn đó
        # SAI: docstring viện dẫn `docs/12 §6 "Xuất CSV nộp bài"` — mục không
        # tồn tại. Luật BTC mới là nguồn đúng.
        raise HTTPException(
            status_code=422,
            detail="AVS là task nội bộ, chưa có format nộp chính thức — "
                   "dùng /v1/search/avs để lấy danh sách segment có thứ hạng",
        )
    return SubmissionBuildResponse(
        task=request.task,
        item_count=len(items),
        csv=csv_text,
        has_errors=has_errors(issues),
        issues=[SubmissionIssue(**dataclasses.asdict(item)) for item in issues],
    )


@router.post("/submissions/validate", response_model=list[SubmissionIssue])
async def validate_submission(
    request: SubmissionBuildRequest, container: Container
) -> list[SubmissionIssue]:
    """Validate không kèm CSV — dùng sau khi người dùng sửa tay trong Submission Board."""

    lookup = _submission_lookup(container)
    if request.task == TaskType.TEXTUAL_KIS:
        issues = await validate_kis(build_kis_submission(request.kis), frame_count=lookup)
    elif request.task == TaskType.QA:
        issues = await validate_qa(build_qa_submission(request.qa), frame_count=lookup)
    elif request.task == TaskType.TRAKE:
        issues = await validate_trake(build_trake_submission(request.trake), frame_count=lookup)
    else:
        raise HTTPException(
            status_code=422, detail="AVS là task nội bộ, chưa có format nộp chính thức"
        )
    return [SubmissionIssue(**dataclasses.asdict(item)) for item in issues]


@router.post("/submissions/evaluate-local", response_model=EvaluateLocalResponse)
async def evaluate_local(request: EvaluateLocalRequest) -> EvaluateLocalResponse:
    """Chấm thử một submission trên gold người dùng tự dán vào (xem trước điểm)."""

    intervals = tuple(
        GoldInterval(start_frame=item.start_frame, end_frame=item.end_frame)
        for item in request.intervals
    )
    if request.task == TaskType.TEXTUAL_KIS:
        result = score_kis(
            build_kis_submission(request.kis), KisGold(request.video_id, intervals)
        )
    elif request.task == TaskType.QA:
        result = score_qa(
            build_qa_submission(request.qa),
            QaGold(request.video_id, intervals, tuple(request.accepted_answers)),
        )
    elif request.task == TaskType.TRAKE:
        items = build_trake_submission(request.trake)
        if not items:
            raise HTTPException(status_code=422, detail="no TRAKE row to evaluate")
        step_windows = tuple(
            GoldInterval(start_frame=item.start_frame, end_frame=item.end_frame)
            for item in request.step_windows
        )
        result = score_trake(items[0], TrakeGold(request.video_id, step_windows))
    else:
        # AVS là task nội bộ, không nộp bài, nên cũng không có scorer cục bộ
        # theo luật BTC. Muốn đo chất lượng AVS thì dùng
        # `scripts/eval_tasks.py --tasks AVS` (nDCG khử trùng theo sự kiện).
        raise HTTPException(
            status_code=422,
            detail="AVS là task nội bộ — đo bằng scripts/eval_tasks.py --tasks AVS",
        )
    return EvaluateLocalResponse(
        score=result.score, best_rank=result.best_rank, detail=result.detail
    )


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_RANGE_CHUNK = 1 << 20


def _range_response(target: Path, range_header: str, media_type: str) -> Response:
    """Trả 206 Partial Content cho một dải byte.

    VÌ SAO PHẢI TỰ CÀI: `FileResponse` của Starlette 0.38 KHÔNG xử lý header
    `Range` — nó trả 200 kèm toàn bộ file. Với video 130MB, trình duyệt không
    tua được: đặt `currentTime` bị bỏ qua và player đứng ở 0:00. Đó chính là
    lỗi làm chức năng "bấm vào dòng submission để xem đúng đoạn" vô dụng.
    """

    size = target.stat().st_size
    match = _RANGE_RE.fullmatch(range_header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise HTTPException(416, "invalid range", headers={"content-range": f"bytes */{size}"})

    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else size - 1
    else:
        # `bytes=-N` = N byte CUỐI file.
        start = max(size - int(match.group(2)), 0)
        end = size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        raise HTTPException(416, "range not satisfiable", headers={"content-range": f"bytes */{size}"})

    def stream():
        remaining = end - start + 1
        with target.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(_RANGE_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type=media_type,
        headers={
            "content-range": f"bytes {start}-{end}/{size}",
            "content-length": str(end - start + 1),
            "accept-ranges": "bytes",
        },
    )


@router.get("/media/{artifact_path:path}")
async def media(artifact_path: str, request: Request, container: Container) -> Response:
    root = container.settings.data_root.resolve()
    target = (root / artifact_path).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(400, "invalid media path")
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".svg", ".mp4", ".webm", ".mkv"}
    suffix = target.suffix.casefold()
    if suffix not in allowed:
        raise HTTPException(403, "artifact type is not public media")
    relative = target.relative_to(root)
    if not (relative.parts[:1] == ("processed",) or relative.parts[:2] == ("raw", "videos")):
        raise HTTPException(403, "artifact is outside public media roots")
    if not target.is_file():
        raise HTTPException(404, "media not found")

    media_type = {
        ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
    }.get(suffix, "application/octet-stream")
    range_header = request.headers.get("range")
    if range_header and suffix in (".mp4", ".webm", ".mkv"):
        return _range_response(target, range_header, media_type)
    # Không có Range: vẫn phải quảng cáo `accept-ranges` để trình duyệt biết
    # nó ĐƯỢC PHÉP tua, nếu không nó sẽ không gửi Range ở lần sau.
    return FileResponse(target, headers={"accept-ranges": "bytes"})
