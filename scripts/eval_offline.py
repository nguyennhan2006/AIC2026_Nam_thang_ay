#!/usr/bin/env python3
"""Eval offline — chạy trực tiếp không qua HTTP server.

Usage:
    python scripts/eval_offline.py [--top-k 100] [--output docs/42_eval_results.jsonl]
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from online.adapters.json_metadata import JsonlSceneRepository
from online.services.search import SearchService


def load_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def run_query(service: SearchService, query: dict, top_k: int = 100) -> dict:
    from online.domain.models import SearchRequest

    task = query.get("task", "AVS").upper()
    query_text = query.get("query_en") or query.get("query_vi", "")

    request = SearchRequest(
        query=query_text,
        task=task,
        top_k=top_k,
    )

    start = time.time()
    try:
        response = await service.search(request)
        elapsed = time.time() - start

        return {
            "query_id": query.get("query_id", "unknown"),
            "task": query.get("task"),
            "query_vi": query.get("query_vi", ""),
            "query_en": query.get("query_en", ""),
            "dense_query_en": query.get("dense_query_en", ""),
            "target_video": query.get("target_video", ""),
            "target_intervals": query.get("target_intervals", []),
            "elapsed_sec": round(elapsed, 2),
            "status": "ok",
            "results_count": len(response.results),
            "top_results": [
                {
                    "video_id": r.video_id,
                    "frame_idx": r.frame_idx,
                    "score": r.score,
                    "candidate_id": r.candidate_id,
                }
                for r in response.results[:10]
            ],
            "all_results": [
                {
                    "video_id": r.video_id,
                    "frame_idx": r.frame_idx,
                    "score": r.score,
                    "candidate_id": r.candidate_id,
                }
                for r in response.results
            ],
        }
    except Exception as e:
        return {
            "query_id": query.get("query_id", "unknown"),
            "task": query.get("task"),
            "elapsed_sec": round(time.time() - start, 2),
            "status": "error",
            "error": str(e),
        }


async def main():
    parser = argparse.ArgumentParser(description="Eval offline (no HTTP server)")
    parser.add_argument("--queries", default="examples/gold_all3.jsonl")
    parser.add_argument("--output", default="docs/42_eval_results.jsonl")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries")
    print(f"Will retrieve top_k={args.top_k}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Build service
    from online.api.container import build_container

    print("Building container (this loads all models)...")
    container = await build_container()
    service = SearchService(container=container)
    print("Container ready!")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = []
    tasks_summary = {"KIS": [], "VQA": [], "AVS": [], "TRAKE": []}
    total_queries = len(queries)

    for i, q in enumerate(queries):
        qid = q.get("query_id", f"q{i}")
        task = q.get("task", "UNKNOWN")
        print(f"[{i+1:3d}/{total_queries}] {qid:12s} ({task:6s})...", end=" ", flush=True)

        result = await run_query(service, q, top_k=args.top_k)
        results.append(result)

        if result["status"] == "ok":
            print(f"OK  {result['elapsed_sec']:.1f}s  ({result['results_count']} results)")
        else:
            print(f"ERR {result.get('error', 'unknown')[:60]}")

        tasks_summary[task].append(result["status"] == "ok")

        # Save incrementally
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total_ok = 0
    for task, statuses in tasks_summary.items():
        ok = sum(statuses)
        total = len(statuses)
        total_ok += ok
        print(f"  {task:6s}: {ok}/{total} OK ({ok/total*100:.1f}%)")

    avg_time = sum(r["elapsed_sec"] for r in results if r["status"] == "ok") / total_ok if total_ok else 0
    print(f"\n  Total: {total_ok}/{total_queries} OK")
    print(f"  Avg query time: {avg_time:.1f}s")
    print(f"\nResults: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
