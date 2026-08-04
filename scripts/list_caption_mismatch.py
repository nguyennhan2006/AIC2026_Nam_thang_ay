"""CAPTION-ENRICH-01 bước 1: liệt kê chính xác scene cần sinh lại caption.

Chỉ lấy scene ĐÃ có keyframe và ĐÃ có caption nhưng retrieval vẫn không tìm
ra vùng gold — tức lỗi nằm ở NỘI DUNG caption, không phải thiếu dữ liệu.
Scene không có candidate (đã sửa ở SCENE-COVERAGE-01) không nằm trong đây.
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from online.adapters.dense_text import CaptionDenseRetriever
from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SearchRequest
from online.domain.tasks import TaskType
from online.services.query_planner import compute_modality_weights
from scripts.build_caption_dense_index import E5Encoder, build_document_text
from scripts.eval_kis import build_service
from scripts.eval_tasks import WindowPolicy, load_fps, load_gold, resolve_step_window

POLICY = WindowPolicy()

async def main_async(args):
    repo = await JsonlSceneRepository.load(args.metadata)
    scenes = await repo.all(); fps = load_fps(args.metadata)
    by_id = {s.scene_id: s for s in scenes}
    gold = [g for g in load_gold(args.gold) if g.task == TaskType.TRAKE]
    svc = await build_service("fusion", repo, backend="local", use_rules=False,
                              use_expansion=False, use_query_prep=False, candidate_limit=100)
    dense = CaptionDenseRetriever(args.index, E5Encoder(str(args.model)))
    svc.retrievers = [dense]
    from online.services.registry import RetrieverRegistry
    from online.services.retrieval_orchestrator import RetrievalOrchestrator
    svc.registry = RetrieverRegistry(svc.retrievers); svc.orchestrator = RetrievalOrchestrator(svc.retrievers)

    targets, stats = [], {"found":0,"no_scene":0,"empty_doc":0,"mismatch":0}
    for g in gold:
        plan = await svc.planner.plan(SearchRequest(query=g.query, task=TaskType.TRAKE))
        for event, step in zip(plan.events, g.steps):
            w = compute_modality_weights(event.text, event.exact_phrases)
            ep = plan.model_copy(update={"normalized_query":event.text,"events":[event],"modality_weights":w})
            tol,_ = resolve_step_window(step, scenes, fps, POLICY)
            c,_ = await svc._retrieve(ep,100); h = await svc._hydrate(c, event.text)
            if any(step.contains(x.best_frame_idx,tol) for x in h):
                stats["found"] += 1; continue
            centre=(step.start_frame+step.end_frame)//2
            owner=next((s for s in scenes if s.start_frame<=centre<s.end_frame_exclusive), None)
            if owner is None: stats["no_scene"] += 1; continue
            doc = build_document_text(owner)
            if not doc.strip(): stats["empty_doc"] += 1; continue
            stats["mismatch"] += 1
            targets.append({
                "scene_id": owner.scene_id, "video_id": owner.video_id,
                "query_id": g.query_id, "event_text": event.text,
                "gold_center_frame": centre,
                "window_sec": round(tol*2/fps, 2),
                "scene_frames": [owner.start_frame, owner.end_frame_exclusive],
                "scene_sec": [round(owner.start_sec,1), round(owner.end_sec,1)],
                "keyframes": [f.frame_idx for f in owner.keyframes],
                "keyframe_paths": [f.image_path for f in owner.keyframes],
                "caption_old": " ".join(owner.captions)[:400],
                "ocr_old": " ".join(owner.ocr_texts)[:200],
            })
    print(json.dumps(stats, ensure_ascii=False))
    print(f"\n=== {len(targets)} scene can sinh lai caption ===")
    for t in targets:
        print(f"\n{t['scene_id']}  ({t['query_id']}, {t['scene_sec'][0]}-{t['scene_sec'][1]}s, "
              f"{len(t['keyframes'])} keyframe)")
        print(f"  EVENT   : {t['event_text']}")
        print(f"  caption : {t['caption_old'][:120]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(targets, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, default=Path("storage/exports_l21_repaired/scenes.jsonl"))
    p.add_argument("--gold", type=Path, default=Path("examples/AIC2026_L21_V001_queries_4tasks.jsonl"))
    p.add_argument("--model", type=Path, default=Path("storage/models/multilingual-e5-large"))
    p.add_argument("--index", type=Path, default=Path("storage/indexes_l21/caption_dense_repaired"))
    p.add_argument("--out", type=Path, default=Path("outputs/evaluation/caption_enrich_targets.json"))
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
