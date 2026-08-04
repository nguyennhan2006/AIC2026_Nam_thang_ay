"""Small dependency-free Online latency/error smoke load."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
from time import perf_counter
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--query", default='"Gừng cay muối mặn"')
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    body = json.dumps({"query": args.query, "top_k": 10}).encode()
    headers = {"Content-Type":"application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    def one() -> float:
        started = perf_counter()
        request = Request(args.url.rstrip("/") + "/v1/search/kis", data=body, headers=headers, method="POST")
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            json.loads(response.read())
        return (perf_counter() - started) * 1000

    latencies, errors = [], []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one) for _ in range(args.requests)]
        for future in as_completed(futures):
            try:
                latencies.append(future.result())
            except Exception as exc:
                errors.append(str(exc))
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered)-1, max(0, round(.95*len(ordered))-1))] if ordered else None
    print(json.dumps({"requests":args.requests, "ok":len(latencies), "errors":len(errors), "p50_ms":statistics.median(ordered) if ordered else None, "p95_ms":p95, "error_examples":errors[:3]}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
