"""DENSE-TEXT-01: A/B/C trên đường Stage B THẬT của TRAKE.

    A  baseline hiện tại (4 BM25 + CLIP ảnh)
    B  chỉ caption dense (E5)
    C  baseline + caption dense, fuse bằng Weighted RRF sẵn có

Đo bằng đúng harness đã chốt ở scripts/diagnose_trake_stage_b.py — KHÔNG dùng
proxy `search(task=KIS)` vì đường đó có dedup + KisProcessor, cho số bi quan
hơn Stage B thật.
"""
from __future__ import annotations
import argparse, asyncio, json, statistics
from pathlib import Path

from online.adapters.dense_text import CaptionDenseRetriever
from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest
from online.domain.tasks import TaskType
from online.services.query_planner import compute_modality_weights
from online.adapters.dense_text import ENCODER_KINDS, build_text_encoder
from scripts.eval_kis import build_service
from scripts.eval_tasks import WindowPolicy, load_fps, load_gold, resolve_step_window

POLICY = WindowPolicy()

async def measure(service, scenes, fps, gold) -> list[dict]:
    rows = []
    for item in gold:
        plan = await service.planner.plan(SearchRequest(query=item.query, task=TaskType.TRAKE))
        if len(plan.events) < 2:
            continue
        for index, (event, step) in enumerate(zip(plan.events, item.steps, strict=False)):
            weights = compute_modality_weights(
                event.text, event.exact_phrases,
                allow_zero=getattr(service.planner, "allow_zero_modality", True))
            event_plan = plan.model_copy(update={"normalized_query": event.text,
                                                 "events": [event], "modality_weights": weights})
            candidates, _ = await service._retrieve(event_plan, service.candidate_limit)
            hits = await service._hydrate(candidates, event.text)
            tolerance, _src = resolve_step_window(step, scenes, fps, POLICY)
            rank = next((position for position, hit in enumerate(hits, 1)
                         if step.contains(hit.best_frame_idx, tolerance)), None)
            rows.append({"query_id": item.query_id, "event_order": index + 1,
                         "event_text": event.text, "gold_region_rank": rank,
                         "pool": len(hits)})
    return rows

def report(name: str, rows: list[dict]) -> dict:
    found = [r["gold_region_rank"] for r in rows if r["gold_region_rank"]]
    by_query: dict[str, list[bool]] = {}
    for r in rows:
        by_query.setdefault(r["query_id"], []).append(r["gold_region_rank"] is not None)
    complete = sum(1 for v in by_query.values() if all(v))
    summary = {
        "variant": name, "steps": len(rows),
        "R@20": sum(1 for r in found if r <= 20), "R@50": sum(1 for r in found if r <= 50),
        "R@100": sum(1 for r in found if r <= 100),
        "median_rank": statistics.median(found) if found else None,
        "all_events": complete, "queries": len(by_query),
        "pool": round(statistics.mean(r["pool"] for r in rows), 1),
    }
    print(f"  {name:22s} R@20={summary['R@20']:2d}/{len(rows)}  R@50={summary['R@50']:2d}/{len(rows)}"
          f"  R@100={summary['R@100']:2d}/{len(rows)}  median={summary['median_rank']}"
          f"  du_moi_event={complete}/{len(by_query)}  pool={summary['pool']}")
    return summary

async def main_async(args):
    repo = await JsonlSceneRepository.load(args.metadata)
    scenes = await repo.all(); fps = load_fps(args.metadata)
    gold = [g for g in load_gold(args.gold) if g.task == TaskType.TRAKE]
    # `for_passages=False`: đây là phía query, y như container.
    encoder = build_text_encoder(args.encoder, str(args.model), for_passages=False)
    # Warmup BẮT BUỘC, đúng lý do container đã ghi: lần `encode()` đầu tiên nuốt
    # trọn thời gian nạp/khởi tạo, vượt `AIC_BRANCH_TIMEOUT_MS`, và
    # `_raise_if_all_degraded` giết cả lượt đo. Đã gặp thật với E5 ở đây:
    # `caption_dense.raw=timeout` giữa variant B. Thiếu dòng này thì kết quả
    # đo được là kết quả của một cuộc đua nạp model, không phải của model.
    encoder.warmup()

    # Encoder thứ hai (tuỳ chọn) cho variant D: hai nhánh dense text CÙNG chạy.
    # Đây là câu hỏi khác hẳn A/B — A/B hỏi "model nào tốt hơn", D hỏi "hai
    # model có tìm ra những bước KHÁC nhau không". Hai model hoà nhau ở C vẫn
    # có thể thắng khi cộng lại, nếu chúng hỏng ở những chỗ khác nhau.
    encoder2 = None
    if args.encoder2:
        encoder2 = build_text_encoder(args.encoder2, str(args.model2), for_passages=False)
        encoder2.warmup()

    variants = ["A baseline", "B dense only", "C baseline+dense"]
    if encoder2 is not None:
        variants.append("D baseline+ca hai")

    out = {}
    print(f"=== DENSE-TEXT-01 tren Stage B that (encoder={args.encoder}"
          + (f" + {args.encoder2}" if encoder2 is not None else "") + ") ===")
    for name in variants:
        svc = await build_service("fusion", repo, backend="local", use_rules=False,
                                  use_expansion=False, use_query_prep=False, candidate_limit=100)
        dense = CaptionDenseRetriever(args.index, encoder)
        # Cùng chốt container dùng: e5 và jina_v3 đều 1024 chiều nên chạy lệch
        # họ vẫn ra số, chỉ là số vô nghĩa — và ở đây số đó sẽ đi thẳng vào
        # bảng so sánh.
        dense.assert_encoder_kind(encoder)
        if name.startswith("B"):
            svc.retrievers = [dense]
        elif name.startswith("C"):
            svc.retrievers = [*svc.retrievers, dense]
        elif name.startswith("D"):
            # branch_id RIÊNG cho nhánh thứ hai, nếu không fusion sẽ coi hai
            # nhánh là một và phiếu của nhánh sau ghi đè nhánh trước.
            dense2 = CaptionDenseRetriever(
                args.index2, encoder2, branch_id=f"caption_dense_{args.encoder2}"
            )
            dense2.assert_encoder_kind(encoder2)
            svc.retrievers = [*svc.retrievers, dense, dense2]
        if not name.startswith("A"):
            from online.services.registry import RetrieverRegistry
            from online.services.retrieval_orchestrator import RetrievalOrchestrator
            svc.registry = RetrieverRegistry(svc.retrievers)
            svc.orchestrator = RetrievalOrchestrator(svc.retrievers)
        out[name] = report(name, await measure(svc, scenes, fps, gold))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("storage/exports_l21/scenes.jsonl"))
    p.add_argument("--gold", type=Path, default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl"))
    p.add_argument("--model", type=Path, default=Path("storage/models/multilingual-e5-large"))
    p.add_argument("--encoder", choices=ENCODER_KINDS, default="e5")
    p.add_argument("--index", type=Path, default=Path("storage/indexes_l21/caption_dense"))
    # Bộ thứ hai, bật variant D. Bỏ trống -> chỉ chạy A/B/C như trước.
    p.add_argument("--encoder2", choices=ENCODER_KINDS, default=None)
    p.add_argument("--model2", type=Path, default=None)
    p.add_argument("--index2", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("outputs/evaluation/dense_text_01.json"))
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
