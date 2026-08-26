#!/usr/bin/env python3
"""Chạy 120 query trên server để đánh giá caption_dense branch.

Usage:
    python scripts/eval_vastai_server.py --server http://IP:8001 [--output docs/42_eval_results.jsonl]
"""

import argparse
import json
import time
import httpx
from pathlib import Path
from datetime import datetime


def load_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_query(client: httpx.Client, server_url: str, query: dict, top_k: int = 100) -> dict:
    # Map task names to correct endpoints
    task_name = query.get("task", "AVS").upper()
    task_to_endpoint = {
        "KIS": "/v1/search/kis",
        "VQA": "/v1/search/qa",
        "AVS": "/v1/search/avs",
        "TRAKE": "/v1/search/trake",
    }
    endpoint = f"{server_url}{task_to_endpoint.get(task_name, f'/v1/search/{task_name.lower()}')}"

    # Dùng query_en hoặc query_vi tùy model
    query_text = query.get("query_en") or query.get("query_vi", "")

    payload = {
        "query": query_text,
        "task": query.get("task", "AVS"),
        "top_k": top_k,
    }

    start = time.time()
    try:
        resp = client.post(endpoint, json=payload, timeout=120.0)
        elapsed = time.time() - start
        resp.raise_for_status()
        result = resp.json()
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
            "results_count": len(result.get("results", [])),
            "top_results": [
                {"video_id": r.get("video_id"), "frame_idx": r.get("frame_idx"),
                 "score": r.get("score", 0), "candidate_id": r.get("candidate_id", "")}
                for r in result.get("results", [])[:10]
            ],
            "all_results": result.get("results", []),
            "full_response": result,
        }
    except httpx.HTTPStatusError as e:
        return {
            "query_id": query.get("query_id", "unknown"),
            "task": query.get("task"),
            "elapsed_sec": round(time.time() - start, 2),
            "status": f"error_{e.response.status_code}",
            "error": str(e),
        }
    except Exception as e:
        return {
            "query_id": query.get("query_id", "unknown"),
            "task": query.get("task"),
            "elapsed_sec": round(time.time() - start, 2),
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Eval 120 queries")
    parser.add_argument("--server", default="http://localhost:8001",
                        help="Server URL")
    parser.add_argument("--queries", default="examples/gold_all3.jsonl",
                        help="Path to query file")
    parser.add_argument("--output", default="docs/42_eval_results.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of results to retrieve")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries from {args.queries}")
    print(f"Will retrieve top_k={args.top_k} candidates per query")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Check server health
    health_url = f"{args.server}/v1/startup"
    try:
        with httpx.Client() as client:
            resp = client.get(health_url, timeout=10.0)
            print(f"Server status: {resp.status_code}")
            print(f"Startup response: {resp.json()}")
    except Exception as e:
        print(f"WARNING: Cannot reach server at {args.server}: {e}")
        print("Continuing anyway...")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results = []
    tasks_summary = {"KIS": [], "VQA": [], "AVS": [], "TRAKE": []}
    total_queries = len(queries)

    with httpx.Client(timeout=120.0) as client:
        for i, q in enumerate(queries):
            qid = q.get("query_id", f"q{i}")
            task = q.get("task", "UNKNOWN")
            print(f"[{i+1:3d}/{total_queries}] {qid:12s} ({task:6s})...", end=" ", flush=True)

            result = run_query(client, args.server, q, top_k=args.top_k)
            results.append(result)

            if result["status"] == "ok":
                print(f"OK  {result['elapsed_sec']:.1f}s  ({result['results_count']} results)")
            else:
                print(f"ERR {result.get('error', result.get('status', 'unknown'))[:60]}")

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
    print(f"\n  Total: {total_ok}/{total_queries} OK ({total_ok/total_queries*100:.1f}%)")
    print(f"  Avg query time: {avg_time:.1f}s")
    print(f"\nResults saved to: {args.output}")

    # Save summary report
    summary_path = args.output.replace(".jsonl", "_summary.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "server": args.server,
        "top_k": args.top_k,
        "total_queries": total_queries,
        "successful": total_ok,
        "success_rate": f"{total_ok/total_queries*100:.1f}%",
        "avg_query_time_sec": round(avg_time, 2),
        "by_task": {task: {"ok": sum(s), "total": len(s),
                          "rate": f"{sum(s)/len(s)*100:.1f}%"}
                    for task, s in tasks_summary.items()},
        "results_file": args.output,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
