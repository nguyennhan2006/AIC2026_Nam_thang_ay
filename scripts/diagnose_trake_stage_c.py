"""FRAME-REFINE-01: tách metric Stage C khỏi Stage B.

Vì sao phải tách (docs/20_EXPERIMENT_LOG.md § CAPTION-ENRICH-01): metric cũ
`gold_region_recall` kiểm `step.contains(hit.best_frame_idx, tol)` — nó TRỘN
hai tầng làm một. Ba thí nghiệm liên tiếp đã tối ưu tầng retrieval trong khi
tầng đó đã đạt 35/35, chỉ vì không ai tách nó ra.

Ba chỉ số ở đây tách bạch hẳn:

    scene_recall
        Stage B có trả về đúng scene chứa mốc gold không.

    frame_oracle_coverage
        Trong scene đó, có TỒN TẠI keyframe nào rơi vào dung sai không.
        Đây là trần cứng của Stage C — không keyframe hợp lệ thì không
        thuật toán chọn frame nào cứu được.

    frame_selection_accuracy_given_oracle
        Khi oracle tồn tại, Stage C có chọn ĐÚNG cái đó không.
        ĐÂY mới là thứ FRAME-REFINE-01 được phép cải thiện.

Gold CHỈ dùng để chấm. Không bao giờ đưa vào đường chọn frame lúc chạy.

Chạy::

    python -m scripts.diagnose_trake_stage_c \\
        --metadata storage/exports_l21_repaired/scenes.jsonl \\
        --out outputs/evaluation/trake_stage_c_baseline.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest
from online.domain.tasks import TaskType
from online.services.query_planner import compute_modality_weights
from online.services.trake.frame_refinement import RefinementConfig, refine_step
from scripts.eval_kis import build_service
from scripts.eval_tasks import WindowPolicy, load_fps, load_gold, resolve_step_window


async def collect(metadata: Path, gold_path: Path, policy: WindowPolicy) -> list[dict]:
    repository = await JsonlSceneRepository.load(metadata)
    scenes = await repository.all()
    fps = load_fps(metadata)
    gold = [item for item in load_gold(gold_path) if item.task == TaskType.TRAKE]
    service = await build_service(
        "fusion", repository, backend="local", use_rules=False,
        use_expansion=False, use_query_prep=False, candidate_limit=100,
    )

    rows: list[dict] = []
    for item in gold:
        plan = await service.planner.plan(SearchRequest(query=item.query, task=TaskType.TRAKE))
        if len(plan.events) < 2:
            continue
        for index, (event, step) in enumerate(zip(plan.events, item.steps, strict=False)):
            weights = compute_modality_weights(
                event.text, event.exact_phrases,
                allow_zero=getattr(service.planner, "allow_zero_modality", True),
            )
            event_plan = plan.model_copy(update={
                "normalized_query": event.text, "events": [event], "modality_weights": weights,
            })
            candidates, _ = await service._retrieve(event_plan, service.candidate_limit)
            hits = await service._hydrate(candidates, event.text)

            tolerance, source = resolve_step_window(step, scenes, fps, policy)
            centre = (step.start_frame + step.end_frame) // 2
            owner = next(
                (s for s in scenes if s.start_frame <= centre < s.end_frame_exclusive), None
            )

            record: dict = {
                "query_id": item.query_id, "event_order": index + 1,
                "event_text": event.text,
                "scene_id": owner.scene_id if owner else None,
                "gold_center_frame": centre,
                "tolerance_frames": round(tolerance, 1),
                "window_source": source,
                "scene_retrieved": False, "scene_rank": None,
                "available_keyframes": [], "oracle_valid_keyframes": [],
                "oracle_keyframe_exists": False,
                "selected_frame": None, "selected_frame_hit": False,
            }
            if owner is None:
                rows.append(record)
                continue

            rank = next(
                (position for position, hit in enumerate(hits, 1) if hit.scene_id == owner.scene_id),
                None,
            )
            record["scene_retrieved"] = rank is not None
            record["scene_rank"] = rank

            frames = [frame.frame_idx for frame in owner.keyframes]
            oracle = [f for f in frames if step.contains(f, tolerance)]
            record["available_keyframes"] = frames
            record["oracle_valid_keyframes"] = oracle
            record["oracle_keyframe_exists"] = bool(oracle)

            if rank is not None:
                # Chạy ĐÚNG Stage C thật, dùng anchor mà Stage B đưa ra.
                anchor = next(hit.best_frame_idx for hit in hits if hit.scene_id == owner.scene_id)
                refined = refine_step(
                    index + 1, event.text, owner, anchor, config=RefinementConfig()
                )
                record["anchor_frame"] = anchor
                record["selected_frame"] = refined.frame_idx
                record["selected_frame_hit"] = step.contains(refined.frame_idx, tolerance)
                record["refinement"] = refined.refinement
            rows.append(record)
    return rows


def report(rows: list[dict]) -> dict:
    total = len(rows)
    scene_ok = sum(1 for r in rows if r["scene_retrieved"])
    oracle = [r for r in rows if r["scene_retrieved"] and r["oracle_keyframe_exists"]]
    selected_ok = sum(1 for r in oracle if r["selected_frame_hit"])
    hit_overall = sum(1 for r in rows if r["selected_frame_hit"])

    print(f"=== Stage C — {total} bước ===")
    print(f"scene_recall                          : {scene_ok}/{total}")
    print(f"frame_oracle_coverage                 : {len(oracle)}/{total}"
          f"   (scene đúng VÀ có keyframe hợp lệ)")
    print(f"frame_selection_accuracy_given_oracle : {selected_ok}/{len(oracle)}"
          f"   <- FRAME-REFINE-01 chỉ được cải thiện chỉ số NÀY")
    print(f"frame_hit tổng                        : {hit_overall}/{total}")
    print(f"\nkhông có keyframe hợp lệ (trần cứng)  : "
          f"{sum(1 for r in rows if r['scene_retrieved'] and not r['oracle_keyframe_exists'])}/{total}"
          f"   -> cần DENSE-FRAME-01, không sửa được bằng code")

    wrong = [r for r in oracle if not r["selected_frame_hit"]]
    if wrong:
        print(f"\n{len(wrong)} bước CÓ keyframe hợp lệ nhưng Stage C chọn sai:")
        for r in wrong:
            print(f"  {r['scene_id']}  {r['event_text'][:38]}")
            print(f"    có={r['available_keyframes']} hợp lệ={r['oracle_valid_keyframes']} "
                  f"anchor={r.get('anchor_frame')} -> CHỌN {r['selected_frame']}")
    return {
        "steps": total, "scene_recall": scene_ok,
        "frame_oracle_coverage": len(oracle),
        "frame_selection_accuracy_given_oracle": selected_ok,
        "frame_hit": hit_overall,
    }


async def main_async(args: argparse.Namespace) -> None:
    policy = WindowPolicy(
        min_sec=args.window_min_sec, max_sec=args.window_max_sec, ratio=args.window_ratio
    )
    rows = await collect(args.metadata, args.gold, policy)
    summary = report(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\n-> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRAKE Stage C frame-selection metrics")
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_l21_repaired/scenes.jsonl"))
    parser.add_argument("--gold", type=Path,
                        default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl"))
    parser.add_argument("--window-min-sec", type=float, default=2.0)
    parser.add_argument("--window-max-sec", type=float, default=7.0)
    parser.add_argument("--window-ratio", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
