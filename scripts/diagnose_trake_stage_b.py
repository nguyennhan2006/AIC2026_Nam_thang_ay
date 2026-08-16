"""Đo `gold_region_rank` NGAY TRONG đường Stage B thật của TRAKE.

Vì sao phải có script riêng thay vì gọi `search(task=TEXTUAL_KIS)` cho từng
event: hai đường KHÁC NHAU về vật chất.

    Stage B thật (online/services/search.py, nhánh TRAKE)
        plan -> per-event modality weights
             -> _retrieve(event_plan, candidate_limit)
             -> _hydrate
        KHÔNG dedup, KHÔNG KisProcessor.rank, KHÔNG cắt top_k

    Gọi KIS cho từng event (proxy)
        _retrieve -> deduplicate_for_task -> rerank -> _hydrate
                  -> _format_results(top_k) -> KisProcessor.rank

Proxy cho số BI QUAN hơn thật (13/35 so với 17/35, rank trung vị 10 so với 4)
vì dedup và KisProcessor đẩy vùng gold xuống. Tối ưu một branch mới trên proxy
sai lệch là tối ưu nhầm mục tiêu — đó là lý do script này tồn tại.

Metric xuất ra là baseline chính thức cho DENSE-TEXT-01:

    gold_region_recall@{20,50,100}
    rank của vùng gold khi tìm được
    số query có ĐỦ mọi event retrieve được  <- điều kiện cần của complete chain

Chạy::

    python -m scripts.diagnose_trake_stage_b \\
        --metadata storage/exports_l21/scenes.jsonl \\
        --out outputs/evaluation/trake_stage_b_baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest
from online.domain.tasks import TaskType
from online.services.query_planner import compute_modality_weights
from scripts.eval_kis import build_service
from scripts.eval_tasks import WindowPolicy, load_fps, load_gold, resolve_step_window


async def collect(
    metadata: Path,
    gold_path: Path,
    policy: WindowPolicy,
    *,
    pipeline: str = "container",
    candidate_limit: int | None = None,
) -> list[dict]:
    repository = await JsonlSceneRepository.load(metadata)
    scenes = await repository.all()
    fps = load_fps(metadata)
    gold = [item for item in load_gold(gold_path) if item.task == TaskType.TRAKE]
    if pipeline == "container":
        # ĐÚNG bộ nhánh server chạy. Bản cũ mặc định dùng `build_service`, mà
        # nó dựng một hệ KHÁC: có `bm25_ocr` + `ocr_fuzzy` (production đã TẮT cả
        # hai vì đo được là gây hại) và thiếu hẳn `bm25_object`/`bm25_action`/
        # `color_search`. Số đo ra vì thế không nói về hệ đang chạy — đã suýt
        # dùng nó để xếp lại thứ tự cả một kế hoạch thí nghiệm.
        from online.api.container import build_container
        from online.config import Settings

        service = (await build_container(Settings.from_env())).search_service
    else:
        service = await build_service(
            "fusion", repository, backend="local", use_rules=False,
            use_expansion=False, use_query_prep=False, candidate_limit=100,
        )
    if candidate_limit is not None:
        service.candidate_limit = candidate_limit

    rows: list[dict] = []
    for item in gold:
        plan = await service.planner.plan(SearchRequest(query=item.query, task=TaskType.TRAKE))
        if len(plan.events) < 2:
            continue
        for index, (event, step) in enumerate(zip(plan.events, item.steps, strict=False)):
            # Sao chép ĐÚNG những gì nhánh TRAKE của SearchService làm.
            weights = compute_modality_weights(
                event.text, event.exact_phrases,
                allow_zero=getattr(service.planner, "allow_zero_modality", True),
            )
            event_plan = plan.model_copy(
                update={
                    "normalized_query": event.text,
                    "events": [event],
                    "modality_weights": weights,
                }
            )
            candidates, _statuses = await service._retrieve(event_plan, service.candidate_limit)
            hits = await service._hydrate(candidates, event.text)

            tolerance, source = resolve_step_window(step, scenes, fps, policy)
            rank = next(
                (position for position, hit in enumerate(hits, start=1)
                 if step.contains(hit.best_frame_idx, tolerance)),
                None,
            )
            rows.append({
                "query_id": item.query_id,
                "event_order": index + 1,
                "event_text": event.text,
                "stage_b_candidate_count": len(hits),
                "gold_region_rank": rank,
                "gold_region_present": rank is not None,
                "gold_window": {
                    "start_frame": step.start_frame,
                    "end_frame": step.end_frame,
                    "window_width_sec": round(tolerance * 2 / fps, 3),
                    "interval_source": source,
                },
            })
    return rows


def report(rows: list[dict]) -> None:
    found = [row["gold_region_rank"] for row in rows if row["gold_region_rank"]]
    print(f"=== Stage B thật — {len(rows)} bước ===")
    print(f"pool trung bình mỗi bước : "
          f"{statistics.mean(row['stage_b_candidate_count'] for row in rows):.0f} candidate")
    for k in (20, 50, 100):
        print(f"gold_region_recall@{k:<4d}: {sum(1 for r in found if r <= k)}/{len(rows)}")
    if found:
        print(f"  khi tìm được: rank trung vị={statistics.median(found):.0f} "
              f"top-1={sum(1 for r in found if r == 1)} "
              f"top-5={sum(1 for r in found if r <= 5)} "
              f"top-20={sum(1 for r in found if r <= 20)}")
    by_query: dict[str, list[bool]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row["gold_region_present"])
    complete = sum(1 for values in by_query.values() if all(values))
    print(f"query có ĐỦ mọi event retrieve được: {complete}/{len(by_query)}")
    print("  (đây là điều kiện CẦN của complete_chain_rate — bằng 0 thì chuỗi "
          "đầy đủ là bất khả thi bất kể beam tốt tới đâu)")


async def _main() -> None:
    parser = argparse.ArgumentParser(description="TRAKE Stage B gold-region recall")
    parser.add_argument("--metadata", type=Path, default=Path("storage/exports_l21/scenes.jsonl"))
    parser.add_argument(
        "--gold", type=Path, default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl")
    )
    parser.add_argument("--window-min-sec", type=float, default=2.0)
    parser.add_argument("--window-max-sec", type=float, default=7.0)
    parser.add_argument("--window-ratio", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--pipeline", choices=("container", "legacy"), default="container",
        help="container = dung online/api/container.py, tuc dung bo nhanh server chay. "
             "legacy = build_service cua eval_kis (bo nhanh KHAC, chi de doi chung).",
    )
    parser.add_argument(
        "--candidate-limit", type=int, default=None,
        help="Ghi de so candidate moi buoc (TRK-C07).",
    )
    args = parser.parse_args()

    policy = WindowPolicy(
        min_sec=args.window_min_sec, max_sec=args.window_max_sec, ratio=args.window_ratio
    )
    rows = await collect(
        args.metadata, args.gold, policy,
        pipeline=args.pipeline, candidate_limit=args.candidate_limit,
    )
    report(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nchi tiết từng bước -> {args.out}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
