"""Local command-line smoke runner using the same composition root as the API."""

from __future__ import annotations

import argparse
import asyncio
import sys

from online.api.container import build_container

from online.config import Settings
from online.domain.models import SearchRequest, TaskType


async def _run(args: argparse.Namespace) -> None:
    container = await build_container(Settings.from_env())
    response = await container.search_service.search(
        SearchRequest(
            query=args.query,
            task=TaskType(args.task),
            top_k=args.top_k,
            debug=args.debug,
        )
    )
    print(response.model_dump_json(indent=2))


def main() -> None:
    # Legacy Windows consoles default to a code page that cannot print
    # Vietnamese; force UTF-8 so query results never crash the CLI.
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="AIC 2026 Online V1 search")
    parser.add_argument("query")
    parser.add_argument(
        "--task", choices=[item.value for item in TaskType], default="kis"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()

