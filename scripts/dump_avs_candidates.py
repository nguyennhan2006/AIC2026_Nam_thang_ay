"""Chạy retrieval MỘT lần, ghi lại toàn bộ evidence pack tới được `AvsProcessor`.

Cùng lý do với `dump_kis_features.py`: một lượt eval mất ~43 phút, gần hết là
chờ API, nên không thể thử 5 biến thể chấm điểm bằng 5 lượt chạy.

Khác một điểm quan trọng: ở đây ghi lại **nguyên vẹn `EvidencePack`** chứ không
phải một bộ đặc trưng rút gọn. Nhờ vậy `replay_avs_grading.py` gọi được CHÍNH
`AvsProcessor.rank` thật — giữ nguyên cổng grade, gom cụm, MMR và cap — và chỉ
thay đúng một thứ: cách chấm tiêu chí. Viết lại phần chọn kết quả ở harness sẽ
trôi khỏi bản thật lúc nào không biết, và khi đó ablation đo nhầm thứ khác.

    python -m scripts.dump_avs_candidates --disable-branch event_search \
        --disable-branch ocr_fuzzy --out outputs/evaluation/avs_candidates.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from online.api.container import build_container
from online.competition.gold_text import resolve_gold_text
from online.config import Settings
from online.domain.models import SearchRequest
from online.domain.search_config import BranchRuntimeOptions, FusionOptions, SearchOptions
from online.domain.tasks import TaskType
from online.services import avs as avs_module

_MAX_TOP_K = 200


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    container = await build_container(settings)
    service = container.search_service

    captured: dict = {}
    original = avs_module.AvsProcessor.rank

    def spy(self, query, packs, *, retrieval_scores=None, limit=100,
            normalizers=None, diagnostics=None):
        captured["args"] = (query, packs, retrieval_scores, normalizers, limit)
        return original(self, query, packs, retrieval_scores=retrieval_scores,
                        limit=limit, normalizers=normalizers, diagnostics=diagnostics)

    avs_module.AvsProcessor.rank = spy
    try:
        options = SearchOptions(
            fusion=FusionOptions(max_results_per_video=1_000_000),
            branches={n: BranchRuntimeOptions(enabled=False) for n in args.disable_branch},
        )
        gold = [
            json.loads(line)
            for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        queries = [item for item in gold if item.get("task") == "AVS"]

        dumped: list[dict] = []
        for index, item in enumerate(queries, start=1):
            print(f"  [{index}/{len(queries)}] {item['query_id']}", file=sys.stderr, flush=True)
            captured.clear()
            await service.search(SearchRequest(
                query=resolve_gold_text(item), task=TaskType.AVS,
                top_k=_MAX_TOP_K, search_options=options,
            ))
            if "args" not in captured:
                print(f"    bỏ qua: không tới được AvsProcessor", file=sys.stderr, flush=True)
                continue
            query, packs, scores, normalizers, limit = captured["args"]
            dumped.append({
                "query_id": item["query_id"],
                "query": query,
                "target_video": item["target_video"],
                "relevant_intervals": item.get("relevant_intervals", []),
                "limit": limit,
                "best_retrieval_score": (
                    normalizers.best_retrieval_score if normalizers is not None
                    else max((scores or {}).values(), default=1.0)
                ),
                "retrieval_scores": dict(scores or {}),
                "packs": [json.loads(p.model_dump_json()) for p in packs],
            })
    finally:
        avs_module.AvsProcessor.rank = original

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dumped, ensure_ascii=False), encoding="utf-8")
    print(f"\nđã ghi {out}: {len(dumped)} truy vấn, "
          f"{sum(len(d['packs']) for d in dumped)} evidence pack")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="examples/gold_all3.jsonl")
    parser.add_argument("--out", default="outputs/evaluation/avs_candidates.json")
    parser.add_argument("--disable-branch", action="append", default=[])
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
