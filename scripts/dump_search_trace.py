"""FIX-DETERMINISM-01: đổ toàn bộ dấu vết một lần search ra JSON.

Dùng để chạy CÙNG truy vấn dưới nhiều `PYTHONHASHSEED` khác nhau rồi so từng
tầng, tìm ĐIỂM PHÂN KỲ ĐẦU TIÊN — thay vì chỉ so metric cuối (metric cuối chỉ
cho biết "có khác", không cho biết "khác từ đâu").

Bối cảnh: cùng lệnh, cùng dữ liệu, `mean_r_score` từng dao động 0.075 <-> 0.212
giữa hai lần chạy liên tiếp. Cố định seed thì hết dao động, nên nguồn nhiễu là
thứ tự lặp của `set`/`dict` chuỗi rò vào xếp hạng.

Đổ theo thứ tự pipeline để định vị được tầng hỏng::

    branch      thứ tự candidate của TỪNG nhánh, trước fusion
    fused       sau fuse_candidates
    dedup       sau deduplicate_for_task
    hits        sau hydrate + format
    task        đầu ra của task processor

Chạy::

    PYTHONHASHSEED=0 python -m scripts.dump_search_trace --out outputs/determinism/seed0.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest
from online.domain.search_config import FusionOptions, SearchOptions
from online.domain.tasks import TaskType
from online.services.deduplication import deduplicate_for_task
from online.services.query_planner import compute_modality_weights
from scripts.eval_kis import build_service

UNLIMITED = SearchOptions(fusion=FusionOptions(max_results_per_video=1_000_000))


def brief(candidate) -> dict:
    return {
        "id": candidate.candidate_id,
        "scene": candidate.scene_id,
        "src": candidate.source,
        "rank": candidate.rank,
        "score": round(candidate.raw_score, 9),
        "branches": sorted(candidate.branch_scores),
    }


async def trace_query(service, repository, query: str, task: TaskType) -> dict:
    plan = await service.planner.plan(SearchRequest(query=query, task=task, search_options=UNLIMITED))
    out: dict = {"query": query, "task": task.value}

    # --- từng nhánh, TRƯỚC fusion ---
    lists, statuses = await service.orchestrator.execute(plan, service.candidate_limit)
    out["branch"] = {
        status.execution_id: [brief(c) for c in candidates]
        for candidates, status in zip(lists, statuses, strict=False)
    }
    out["branch_order"] = [status.execution_id for status in statuses]

    # --- fused / dedup / hits ---
    fused, _ = await service._retrieve(plan, service.candidate_limit)
    out["fused"] = [brief(c) for c in fused]
    deduped = deduplicate_for_task(fused, task, max_per_video_override=1_000_000)
    out["dedup"] = [brief(c) for c in deduped]
    hits = await service._hydrate(deduped[:100], plan.normalized_query)
    out["hits"] = [
        {"scene": h.scene_id, "frame": h.best_frame_idx, "score": round(h.score, 9),
         "branches": sorted(h.matched_branches)}
        for h in hits
    ]

    # --- đường TRAKE per-event (nơi dao động được quan sát) ---
    if task == TaskType.TRAKE and len(plan.events) >= 2:
        per_event = []
        for event in plan.events:
            weights = compute_modality_weights(
                event.text, event.exact_phrases,
                allow_zero=getattr(service.planner, "allow_zero_modality", True),
            )
            event_plan = plan.model_copy(update={
                "normalized_query": event.text, "events": [event], "modality_weights": weights,
            })
            candidates, _ = await service._retrieve(event_plan, service.candidate_limit)
            event_hits = await service._hydrate(candidates, event.text)
            per_event.append([
                {"scene": h.scene_id, "frame": h.best_frame_idx, "score": round(h.score, 9)}
                for h in event_hits[:30]
            ])
        out["trake_events"] = per_event

    response = await service.search(
        SearchRequest(query=query, task=task, top_k=100, search_options=UNLIMITED)
    )
    out["task_output"] = [
        {"video": item.video_id,
         "frame": getattr(item, "frame_idx", None) or getattr(item, "frame_ids", None),
         "rank": item.rank}
        for item in (response.kis or response.qa or response.trake or response.avs)
    ]
    return out


async def main_async(args: argparse.Namespace) -> None:
    repository = await JsonlSceneRepository.load(args.metadata)
    service = await build_service(
        "fusion", repository, backend="local", use_rules=False,
        use_expansion=False, use_query_prep=False, candidate_limit=100,
    )
    payload = {
        "hash_seed": os.environ.get("PYTHONHASHSEED", "(unset)"),
        "traces": [
            await trace_query(service, repository, query, TaskType(task))
            for query, task in [
                (args.kis_query, "TEXTUAL_KIS"),
                (args.trake_query, "TRAKE"),
            ]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"seed={payload['hash_seed']} -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đổ dấu vết search để so xuyên hash seed")
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_l21_enriched/scenes.jsonl"))
    parser.add_argument("--kis-query", default="cảnh báo sạt lở nguy hiểm ven sông")
    parser.add_argument(
        "--trake-query",
        default="Tìm video về giếng nước bất ngờ phun cao và căn chỉnh bốn khoảnh khắc: "
                "(1) cột nước được quay từ xa; (2) một người đàn ông tiến sát cột nước; "
                "(3) người này cầm chai hoặc vật chứa cạnh dòng nước; "
                "(4) nhiều người cùng chỉ về phía cột nước",
    )
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
