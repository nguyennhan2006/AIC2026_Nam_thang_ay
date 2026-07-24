from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .builder import build_index
from .config import EngineConfig
from .engine import LocalHybridSearchEngine


def _json_print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aic-local-search")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build the local hybrid index")
    build.add_argument("--input-root", required=True, type=Path)
    build.add_argument("--index-dir", required=True, type=Path)
    build.add_argument("--vector-backend", choices=["auto", "faiss", "numpy"], default="auto")

    query = subparsers.add_parser("query", help="Search scenes")
    query.add_argument("--index-dir", required=True, type=Path)
    query.add_argument("--text", required=True)
    query.add_argument("--visual-text")
    query.add_argument("--query-vector-npy", type=Path)
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--task", choices=["auto", "frame", "scene", "temporal", "qa"], default="auto")
    query.add_argument("--video-id")
    query.add_argument("--start-sec", type=float)
    query.add_argument("--end-sec", type=float)
    query.add_argument("--no-vector", action="store_true")
    query.add_argument("--match-all", action="store_true")

    sequence = subparsers.add_parser("sequence", help="Search ordered scene steps")
    sequence.add_argument("--index-dir", required=True, type=Path)
    sequence.add_argument("--step", action="append", required=True)
    sequence.add_argument("--top-k", type=int, default=5)
    sequence.add_argument("--no-vector", action="store_true")
    sequence.add_argument("--max-gap-sec", type=float, default=120.0)

    inspect = subparsers.add_parser("inspect", help="Print index metadata")
    inspect.add_argument("--index-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "build":
        report = build_index(
            args.input_root,
            args.index_dir,
            EngineConfig(vector_backend=args.vector_backend),
        )
        _json_print(report.to_dict())
        return 0
    if args.command == "inspect":
        manifest = json.loads((args.index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        _json_print(manifest)
        return 0

    with LocalHybridSearchEngine(args.index_dir) as engine:
        if args.command == "query":
            query_vector = None
            if args.query_vector_npy:
                query_vector = np.load(args.query_vector_npy)
                if query_vector.ndim == 2:
                    query_vector = query_vector[0]
            results = engine.search(
                args.text,
                visual_query=args.visual_text,
                query_vector=query_vector,
                use_vector=not args.no_vector,
                task=args.task,
                top_k=args.top_k,
                video_id=args.video_id,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                match_all_terms=args.match_all,
            )
        else:
            results = engine.search_sequence(
                args.step,
                top_k=args.top_k,
                max_gap_sec=args.max_gap_sec,
                use_vector=not args.no_vector,
            )
    _json_print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
