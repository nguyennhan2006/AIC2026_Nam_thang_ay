"""KIS evaluation harness: Recall@K, MRR, hit-in-interval + ablation modes.

Đây là công cụ đo lường bắt buộc trước khi kết luận phương án search nào
tốt hơn. Nó chạy cùng một bộ query ground-truth qua nhiều cấu hình retriever
(ablation) và in bảng so sánh:

    Recall@1/5/20/50/100 : tỉ lệ query có >= 1 hit đúng trong top-K
    MRR                  : mean(1/rank của hit đúng đầu tiên), 0 nếu miss
    video-R@K            : recall chỉ tính đúng video (chẩn đoán: đúng video
                           nhưng lệch interval => vấn đề ở chọn scene/frame,
                           không phải ở retrieval video)

Định nghĩa "hit đúng" (hit-in-interval) cho một GT item:
    - nếu GT có ``scene_ids``: hit.scene_id nằm trong đó, HOẶC
    - hit.video_id == GT.video_id VÀ [hit.start_sec, hit.end_sec] giao với
      [GT.start_sec, GT.end_sec] (nếu GT không có interval thì chỉ cần đúng
      video).

Format ground-truth JSONL (mỗi dòng một query):
    {"query_id": "q1",
     "query": "đoàn người ... có chữ \"xin đừng quên nhau\"",
     "video_id": "L01_V001",
     "start_sec": 20.0, "end_sec": 26.0,
     "scene_ids": ["L01_V001_S0003"]}        # optional

Cách sử dụng
------------
Chạy toàn bộ ablation trên dữ liệu demo (không cần GPU/service ngoài):

    python -m scripts.seed_demo          # sinh storage/exports/scenes.jsonl
    python -m scripts.eval_kis --metadata storage/exports/scenes.jsonl \
        --groundtruth examples/kis_groundtruth.jsonl --mode all

LƯU Ý: đừng trỏ --metadata vào examples/scenes.jsonl — file đó không đạt
schema canonical của datasection nên JsonlSceneRepository sẽ từ chối khi
package datasection import được (đây cũng là lý do một test end-to-end
đang fail). Dùng export từ seed_demo hoặc pipeline offline thật.

Các mode (--mode):
    metadata_only : BM25 trên caption/ocr/asr/keyword (Phương án A)
    vector_only   : dense retriever (Phương án B)
    ocr_only      : OCR fuzzy retriever (Phương án C)
    fusion        : dense + BM25 + OCR fuzzy qua weighted RRF (Phương án D)
    all           : chạy cả 4 mode trên và in bảng so sánh

Cờ cải tiến (ablation từng thành phần, so với baseline cùng mode):
    --use-query-prep : PreparedQueryPlanner — tách target/ocr/context (F)
    --use-rules      : bonus/penalty sau RRF (E)
    --use-expansion  : VI→EN expansion cho BM25 caption/keyword (K)

Backend vector (--backend, mặc định local):
    local  : HashingTextEncoder + InMemoryVectorStore. LƯU Ý: hashing encoder
             chỉ là lexical-hash để smoke test — số vector_only ở local KHÔNG
             phản ánh chất lượng dense thật.
    qdrant : RemoteTextEncoder + QdrantVectorStore, đọc cấu hình từ env
             (AIC_QDRANT_URL, AIC_EMBEDDING_URL, ... — xem online/config.py).
             Dùng backend này để có số dense/fusion thật.

Tùy chọn khác: --top-k (mặc định 100, <= 200), --verbose (in rank hit đầu
tiên của từng query), --json <path> (ghi kết quả máy đọc được).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from online.adapters.bm25 import LexicalRetriever
from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import HashingTextEncoder, LocalClipTextEncoder, RemoteTextEncoder
from online.adapters.fpt_client import FptClient
from online.adapters.frame_vector_store import build_frame_vector_rows
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.ocr_fuzzy import OcrFuzzyRetriever
from online.adapters.rerank import BgeTextReranker, FptTextReranker
from online.adapters.vector_stores import InMemoryVectorStore, QdrantVectorStore
from online.domain.models import SearchRequest, TaskType
from online.services.evidence_builder import EvidenceBuilder
from online.services.query_expansion import QueryExpansionRetriever
from online.services.query_prep import PreparedQueryPlanner
from online.services.rerank_pipeline import RerankPipeline
from online.services.rules import RuleConfig
from online.services.search import SearchService


K_VALUES = (1, 5, 20, 50, 100)
MODES = ("metadata_only", "vector_only", "ocr_only", "fusion")


@dataclass(frozen=True, slots=True)
class GroundTruthItem:
    query_id: str
    query: str
    video_id: str
    start_sec: float | None
    end_sec: float | None
    scene_ids: tuple[str, ...]


def load_groundtruth(path: Path) -> list[GroundTruthItem]:
    items: list[GroundTruthItem] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            items.append(
                GroundTruthItem(
                    query_id=str(raw.get("query_id", f"q{line_number}")),
                    query=raw["query"],
                    video_id=raw["video_id"],
                    start_sec=raw.get("start_sec"),
                    end_sec=raw.get("end_sec"),
                    scene_ids=tuple(raw.get("scene_ids", [])),
                )
            )
    if not items:
        raise SystemExit(f"groundtruth file is empty: {path}")
    return items


def is_interval_hit(hit, gt: GroundTruthItem) -> bool:
    if gt.scene_ids and hit.scene_id in gt.scene_ids:
        return True
    if hit.video_id != gt.video_id:
        return False
    if gt.start_sec is None or gt.end_sec is None:
        return True  # GT chỉ ở mức video
    # So sánh strict (interval nửa mở): scene liền kề luôn chạm biên scene
    # đúng (end_sec == start_sec kế tiếp) — chạm biên KHÔNG được tính là hit.
    return hit.start_sec < gt.end_sec and hit.end_sec > gt.start_sec


async def build_dense(repository: JsonlSceneRepository, backend: str) -> DenseRetriever:
    if backend == "qdrant":
        # Cấu hình lấy từ env để khớp với container production (online/config.py).
        from online.config import Settings

        settings = Settings.from_env()
        if not settings.embedding_url or not settings.qdrant_url:
            raise SystemExit(
                "--backend qdrant cần AIC_EMBEDDING_URL và AIC_QDRANT_URL trong env"
            )
        encoder = RemoteTextEncoder(
            settings.embedding_url, settings.request_timeout_sec, settings.embedding_api_key
        )
        store = QdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_scene_collection,
            settings.qdrant_vector_name,
            api_key=settings.qdrant_api_key,
            timeout_sec=settings.request_timeout_sec,
        )
        return DenseRetriever(encoder, store)

    # local: dùng embedding thật (PR-13, scripts/embed_keyframes_local.py) nếu
    # export đã có — cùng logic chọn nhánh với online/api/container.py, để số
    # đo ở đây khớp với hành vi search thật thay vì luôn là hashing fallback.
    data_root = Path(os.getenv("AIC_DATA_ROOT", "storage")).resolve()
    frame_rows, has_real_embeddings = await build_frame_vector_rows(repository, data_root)
    if has_real_embeddings:
        from online.config import Settings

        settings = Settings.from_env()
        encoder = LocalClipTextEncoder(
            settings.visual_embedding_model, revision=settings.visual_embedding_model_revision
        )
        return DenseRetriever(encoder, InMemoryVectorStore(frame_rows), branch_id="dense_visual", backend_kind="vector")

    encoder = HashingTextEncoder()
    rows = []
    for scene in await repository.all():
        search_text = " ".join(scene.captions + scene.keywords)
        rows.append(
            (
                scene.scene_id,
                scene.video_id,
                await encoder.encode(search_text),
                {
                    "scene_id": scene.scene_id,
                    "video_id": scene.video_id,
                    "scene_idx": scene.scene_idx,
                    "start_sec": scene.start_sec,
                    "end_sec": scene.end_sec,
                    "has_ocr": bool(scene.ocr_texts),
                    "has_asr": bool(scene.asr_texts),
                },
            )
        )
    return DenseRetriever(encoder, InMemoryVectorStore(rows), branch_id="lexical_hash_fallback", backend_kind="lexical_fallback")


def _build_rerank_pipeline(repository: JsonlSceneRepository) -> RerankPipeline | None:
    """Cùng chiến lược ưu tiên FPT như `online/api/container.py` — nếu không
    có harness này sẽ luôn đo với rerank no-op, không phản ánh được cải
    thiện thật của FPT text reranker (PR-15)."""
    from online.config import Settings

    settings = Settings.from_env()
    text_reranker = None
    if settings.fpt_enabled and settings.fpt_rerank_model:
        text_reranker = FptTextReranker(FptClient.from_settings(settings), model_id=settings.fpt_rerank_model)
    elif settings.rerank_text_url:
        text_reranker = BgeTextReranker(
            settings.rerank_text_url,
            model_id=settings.rerank_text_model,
            timeout_sec=settings.request_timeout_sec,
            api_key=settings.rerank_api_key,
        )
    if text_reranker is None:
        return None
    return RerankPipeline(EvidenceBuilder(repository), text_reranker=text_reranker)


async def build_service(
    mode: str,
    repository: JsonlSceneRepository,
    *,
    backend: str,
    use_rules: bool,
    use_expansion: bool,
    use_query_prep: bool,
    use_rerank: bool = False,
    candidate_limit: int,
) -> SearchService:
    retrievers: list = []
    if mode in ("metadata_only", "fusion"):
        for field in ("caption", "ocr", "asr", "keyword"):
            lexical = await LexicalRetriever.build(field, repository)
            # Expansion chỉ wrap caption/keyword: OCR/ASR cần giữ nguyên văn.
            if use_expansion and field in ("caption", "keyword"):
                lexical = QueryExpansionRetriever(lexical)
            retrievers.append(lexical)
    if mode in ("vector_only", "fusion"):
        retrievers.append(await build_dense(repository, backend))
    if mode in ("ocr_only", "fusion"):
        retrievers.append(await OcrFuzzyRetriever.build(repository))

    planner = PreparedQueryPlanner() if use_query_prep else None
    rerank_pipeline = _build_rerank_pipeline(repository) if use_rerank else None
    if use_rerank and rerank_pipeline is None:
        raise SystemExit(
            "--use-rerank yêu cầu AIC_FPT_ENABLED=true + AIC_FPT_RERANK_MODEL "
            "(hoặc AIC_RERANK_TEXT_URL cho worker tự host) trong env"
        )
    return SearchService(
        repository,
        retrievers,
        planner=planner,
        candidate_limit=candidate_limit,
        rule_config=RuleConfig() if use_rules else None,
        rerank_pipeline=rerank_pipeline,
    )


async def evaluate_mode(
    mode: str,
    repository: JsonlSceneRepository,
    groundtruth: list[GroundTruthItem],
    args: argparse.Namespace,
) -> dict:
    service = await build_service(
        mode,
        repository,
        backend=args.backend,
        use_rules=args.use_rules,
        use_expansion=args.use_expansion,
        use_query_prep=args.use_query_prep,
        use_rerank=args.use_rerank,
        candidate_limit=max(args.top_k, 100),
    )
    ranks: list[int | None] = []          # rank hit-in-interval đầu tiên
    video_ranks: list[int | None] = []    # rank đúng-video đầu tiên
    per_query: list[dict] = []
    for gt in groundtruth:
        response = await service.search(
            SearchRequest(query=gt.query, task=TaskType.TEXTUAL_KIS, top_k=args.top_k)
        )
        rank = next(
            (
                index
                for index, hit in enumerate(response.results, start=1)
                if is_interval_hit(hit, gt)
            ),
            None,
        )
        video_rank = next(
            (
                index
                for index, hit in enumerate(response.results, start=1)
                if hit.video_id == gt.video_id
            ),
            None,
        )
        ranks.append(rank)
        video_ranks.append(video_rank)
        per_query.append(
            {
                "query_id": gt.query_id,
                "first_hit_rank": rank,
                "first_video_rank": video_rank,
                "top1_scene": response.results[0].scene_id if response.results else None,
            }
        )
        if args.verbose:
            print(
                f"  [{mode}] {gt.query_id}: hit_rank={rank} video_rank={video_rank}"
                f" top1={per_query[-1]['top1_scene']}"
            )

    total = len(groundtruth)
    metrics = {
        f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / total
        for k in K_VALUES
        if k <= args.top_k
    }
    metrics["mrr"] = sum(1 / r for r in ranks if r is not None) / total
    metrics[f"video_recall@{args.top_k}"] = (
        sum(1 for r in video_ranks if r is not None) / total
    )
    return {"mode": mode, "metrics": metrics, "per_query": per_query}


def print_table(results: list[dict], top_k: int) -> None:
    k_cols = [k for k in K_VALUES if k <= top_k]
    header = ["mode".ljust(14)] + [f"R@{k}".rjust(7) for k in k_cols]
    header += ["MRR".rjust(7), f"vidR@{top_k}".rjust(9)]
    print("".join(header))
    print("-" * (14 + 7 * len(k_cols) + 7 + 9))
    for row in results:
        metrics = row["metrics"]
        cells = [row["mode"].ljust(14)]
        cells += [f"{metrics[f'recall@{k}']:.3f}".rjust(7) for k in k_cols]
        cells.append(f"{metrics['mrr']:.3f}".rjust(7))
        cells.append(f"{metrics[f'video_recall@{top_k}']:.3f}".rjust(9))
        print("".join(cells))


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="KIS Recall@K / MRR evaluation with retriever ablations"
    )
    parser.add_argument("--metadata", type=Path, required=True,
                        help="đường dẫn scenes.jsonl (export của datasection)")
    parser.add_argument("--groundtruth", type=Path, required=True,
                        help="đường dẫn ground-truth JSONL (xem docstring)")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--backend", choices=("local", "qdrant"), default="local")
    parser.add_argument("--top-k", type=int, default=100,
                        help="độ sâu đánh giá, <= 200 (mặc định 100)")
    parser.add_argument("--use-rules", action="store_true",
                        help="bật bonus/penalty sau RRF (Phương án E)")
    parser.add_argument("--use-expansion", action="store_true",
                        help="bật VI→EN expansion cho BM25 (Phương án K)")
    parser.add_argument("--use-query-prep", action="store_true",
                        help="bật tách target/ocr/context (Phương án F)")
    parser.add_argument("--use-rerank", action="store_true",
                        help="bật text rerank thật (FPT ưu tiên, fallback worker tự host qua "
                             "AIC_RERANK_TEXT_URL) — cần env tương ứng, xem online/config.py")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", type=Path, default=None,
                        help="ghi kết quả JSON để lưu vết ablation")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 200:
        raise SystemExit("--top-k phải trong [1, 200]")

    repository = await JsonlSceneRepository.load(args.metadata)
    groundtruth = load_groundtruth(args.groundtruth)
    modes = list(MODES) if args.mode == "all" else [args.mode]

    flags = [
        name
        for name, enabled in (
            ("query_prep", args.use_query_prep),
            ("rules", args.use_rules),
            ("expansion", args.use_expansion),
            ("rerank", args.use_rerank),
        )
        if enabled
    ]
    print(
        f"queries={len(groundtruth)} scenes={len(await repository.all())} "
        f"backend={args.backend} flags={'+'.join(flags) or 'none'}"
    )
    if args.backend == "local" and any(m in modes for m in ("vector_only", "fusion")):
        print(
            "note: backend=local dùng HashingTextEncoder (không semantic) — "
            "số vector_only/fusion chỉ để smoke test, không phải chất lượng dense thật"
        )

    results = [
        await evaluate_mode(mode, repository, groundtruth, args) for mode in modes
    ]
    print()
    print_table(results, args.top_k)

    if args.json:
        payload = {
            "config": {
                "metadata": str(args.metadata),
                "groundtruth": str(args.groundtruth),
                "backend": args.backend,
                "top_k": args.top_k,
                "flags": flags,
            },
            "results": results,
        }
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    try:  # Console Windows có thể không ở UTF-8; tránh crash khi in tiếng Việt.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    asyncio.run(main())
