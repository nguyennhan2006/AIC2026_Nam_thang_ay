"""Chạy retrieval MỘT lần, ghi ra sáu thành phần điểm của từng candidate KIS.

Vì sao tách khỏi `eval_tasks.py`: một lượt eval trên 120 truy vấn mất ~43 phút,
gần hết là chờ API dịch và rerank. Thử một bộ trọng số mà tốn 43 phút thì
không dò được gì — và trọng số là thứ cần thử hàng chục cấu hình.

Script này trả giá 43 phút ĐÚNG MỘT LẦN. Sau đó `sweep_kis_weights.py` xếp
hạng lại offline trong mili-giây, vì đổi trọng số không đổi tập candidate,
không đổi văn bản scene, và không đổi frame được chọn — chỉ đổi thứ tự.

Sáu thành phần dump ra đúng bằng những gì `KisProcessor.rank` dùng. Script
sweep tự kiểm chứng điều đó bằng cách so thứ hạng tính lại với `online_rank`
ghi kèm; lệch là biết ngay bộ đặc trưng thiếu thứ gì, chứ không âm thầm đo
nhầm.

    python -m scripts.dump_kis_features --gold examples/gold_all3.jsonl \
        --out outputs/evaluation/kis_features.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from online.api.container import build_container
from online.config import Settings
from online.domain.models import SearchRequest
from online.domain.search_config import BranchRuntimeOptions, FusionOptions, SearchOptions
from online.domain.tasks import TaskType
from online.services import kis as kis_module
from online.services.safe_frame import select_safe_frame

# Giới hạn của `SearchRequest.top_k`; xin nhiều hơn sẽ bị pydantic từ chối.
_MAX_TOP_K = 200


def _features(query, hits, documents, normalizers, config) -> list[dict]:
    signature = kis_module.build_signature(query)
    best_score = (
        normalizers.best_retrieval_score
        if normalizers is not None
        else (max((hit.score for hit in hits), default=0.0) or 1.0)
    )
    ceiling = (
        normalizers.branch_ceiling
        if normalizers is not None
        else (max((len(hit.matched_branches) for hit in hits), default=1) or 1)
    )

    rows: list[dict] = []
    for hit in hits:
        document = documents.get(hit.scene_id)
        if document is None:
            continue
        text = " ".join([
            *document.captions, *document.ocr_texts, *document.asr_texts,
            *document.object_labels, *document.action_tags,
        ])
        must, nice, contradicted = signature.coverage(text)
        rare = signature.rare_hits(text)
        safe = select_safe_frame(
            document, query, config.safe_frame, prefer_frame_idx=hit.best_frame_idx
        )
        rows.append({
            "scene_id": hit.scene_id,
            "video_id": hit.video_id,
            "frame_idx": safe.frame.frame_idx if safe else hit.best_frame_idx,
            "retrieval": hit.score / best_score if best_score else 0.0,
            "must": must,
            "nice": nice,
            "rare": min(rare / len(signature.rare_cues), 1.0) if signature.rare_cues else 0.0,
            "agreement": len(hit.matched_branches) / ceiling,
            "safe": safe.total if safe else 0.0,
            "contradicted": contradicted,
        })
    return rows


async def _run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    container = await build_container(settings)
    service = container.search_service

    # Chộp đúng đầu vào của `KisProcessor.rank`. Cách khác là dựng lại toàn bộ
    # đường fusion ở đây, và bản dựng lại đó sẽ trôi khỏi bản thật lúc nào
    # không biết.
    captured: dict = {}
    original = kis_module.KisProcessor.rank

    def spy(self, query, hits, documents, *, packs=None, limit=100, normalizers=None):
        captured["args"] = (query, hits, documents, normalizers)
        return original(self, query, hits, documents,
                        packs=packs, limit=limit, normalizers=normalizers)

    kis_module.KisProcessor.rank = spy
    try:
        options = SearchOptions(
            fusion=FusionOptions(max_results_per_video=1_000_000),
            branches={
                name: BranchRuntimeOptions(enabled=False) for name in args.disable_branch
            },
        )
        gold = [
            json.loads(line)
            for line in Path(args.gold).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        queries = [item for item in gold if item.get("task") == "KIS"]

        dumped: list[dict] = []
        for index, item in enumerate(queries, start=1):
            print(f"  [{index}/{len(queries)}] {item['query_id']}", file=sys.stderr, flush=True)
            captured.clear()
            response = await service.search(SearchRequest(
                query=item["query_vi"], task=TaskType.TEXTUAL_KIS,
                top_k=_MAX_TOP_K, search_options=options,
            ))
            if "args" not in captured:
                print(f"    bỏ qua: {item['query_id']} không tới được KisProcessor",
                      file=sys.stderr, flush=True)
                continue
            query, hits, documents, normalizers = captured["args"]
            rows = _features(query, hits, documents, normalizers,
                             service.kis_processor.config)

            intervals = item.get("target_intervals") or []

            def _is_gold(video_id: str, frame_idx: int) -> bool:
                return video_id == item["target_video"] and any(
                    interval["start_frame"] <= frame_idx <= interval["end_frame"]
                    for interval in intervals
                )

            for row in rows:
                row["gold"] = _is_gold(row["video_id"], row["frame_idx"])
            dumped.append({
                "query_id": item["query_id"],
                "query": item["query_vi"],
                "online_rank": next(
                    (rank for rank, result in enumerate(response.kis, start=1)
                     if _is_gold(result.video_id, result.frame_idx)),
                    None,
                ),
                "candidates": rows,
            })
    finally:
        kis_module.KisProcessor.rank = original

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dumped, ensure_ascii=False), encoding="utf-8")
    print(f"\nđã ghi {out}: {len(dumped)} truy vấn, "
          f"{sum(len(item['candidates']) for item in dumped)} candidate")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="examples/gold_all3.jsonl")
    parser.add_argument("--out", default="outputs/evaluation/kis_features.json")
    parser.add_argument("--disable-branch", action="append", default=[])
    args = parser.parse_args()
    metadata = os.environ.get("AIC_METADATA_JSONL")
    if metadata:
        print(f"metadata: {metadata}", file=sys.stderr)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
