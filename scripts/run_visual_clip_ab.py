"""VISUAL-01: CLIP ViT-L/14 so với jina-clip-v2 ở nhánh `dense_visual`.

Khác hẳn DENSE-TEXT-01 (`scripts/run_dense_text_01.py`). Thí nghiệm đó so hai
encoder TEXT trên caption đã trích; ở đây so hai encoder ĐA PHƯƠNG THỨC khớp
truy vấn thẳng với pixel của keyframe.

Giả thuyết cần kiểm: text tower của `openai/clip-vit-large-patch14` là TIẾNG
ANH thuần, mà mọi truy vấn trong bộ gold đều là tiếng Việt. `jina-clip-v2`
dùng text tower jina-XLM-RoBERTa đa ngữ. Nếu giả thuyết đúng thì nhánh
`dense_visual` hiện tại đang chạy dưới sức nó nhiều — và đây không phải chuyện
tinh chỉnh trọng số, mà là truy vấn đi vào một tower không hiểu ngôn ngữ đó.

    A  baseline hiện tại (BM25 + OCR fuzzy + dense_visual CLIP)
    B  chỉ dense_visual CLIP
    C  chỉ dense_visual jina-clip-v2
    D  baseline nhưng THAY CLIP bằng jina-clip-v2
    E  baseline + CẢ HAI nhánh visual

B và C là phép so model sạch nhất (không nhánh nào khác che). D là câu hỏi
triển khai. E kiểm bổ sung lẫn nhau — nhớ đọc kèm control: ở DENSE-TEXT-01,
phần lớn cái lợi của "thêm một nhánh" hoá ra chỉ là hiệu ứng đếm nhánh của
fusion, không phải thông tin mới.

Đo bằng ĐÚNG harness Stage B của `scripts/run_dense_text_01.py` để mọi số
trong docs/20_EXPERIMENT_LOG.md so sánh được với nhau.

    python -m scripts.run_visual_clip_ab \\
        --metadata storage/exports_multivideo/scenes.jsonl \\
        --gold examples/gold_all3.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from online.adapters.dense_retriever import DenseRetriever
from online.adapters.encoders import LocalTextEncoder
from online.adapters.frame_vector_store import build_frame_vector_rows_by_index
from online.adapters.json_metadata import JsonlSceneRepository
from online.adapters.vector_stores import InMemoryVectorStore
from online.domain.tasks import TaskType
from scripts.eval_kis import build_service
from scripts.eval_tasks import load_fps, load_gold
from scripts.run_dense_text_01 import measure, report


def wrap_translation(encoder):
    """Bọc `TranslatingTextEncoder` đúng như container làm.

    Bỏ bước này là đo một CLIP KHÁC với CLIP đang phục vụ. Text tower của
    `openai/clip-vit-large-patch14` chỉ biết tiếng Anh, và không dịch thì nó dồn
    mọi truy vấn tiếng Việt về gần một điểm (đo được: cosine giữa 10 câu khác
    nghĩa hẳn nhau = 0.912). `build_service` của eval_kis KHÔNG bọc — cảnh báo
    có sẵn ở eval_kis.py:313 — nên mọi ablation nhánh visual trước đây đều đo
    một CLIP chạy dưới sức.
    """

    from online.adapters.fpt_client import FptClient
    from online.adapters.fpt_query import FptQueryTranslator, TranslatingTextEncoder
    from online.config import Settings

    settings = Settings.from_env()
    model_id = settings.fpt_fast_llm_model or settings.fpt_llm_model
    if not (settings.fpt_enabled and model_id):
        raise SystemExit(
            "cần AIC_FPT_ENABLED=true + AIC_FPT_FAST_LLM_MODEL để dịch. "
            "Chạy với AIC_ENV_FILE=.env.fpt.local, hoặc bỏ --translate-clip "
            "(nhưng khi đó số CLIP KHÔNG so được với production)."
        )
    return TranslatingTextEncoder(
        encoder,
        FptQueryTranslator(
            FptClient.from_settings(settings),
            model_id=model_id,
            cache_dir=settings.query_translation_cache_dir,
        ),
    )


async def build_visual(repo, data_root: Path, embedding_name: str, model_path: str,
                       *, translate: bool = False) -> DenseRetriever:
    """Một nhánh dense_visual đọc ĐÚNG một embedding_name.

    Dùng `build_frame_vector_rows_by_index` chứ không `build_frame_vector_rows`:
    hàm sau lấy ref ĐẦU TIÊN của mỗi keyframe, nên khi export mang cả CLIP lẫn
    jina nó sẽ luôn trả CLIP và thí nghiệm này lặng lẽ đo cùng một model hai
    lần.
    """

    by_index = await build_frame_vector_rows_by_index(
        repo, data_root, embedding_names=[embedding_name]
    )
    rows = by_index.get(embedding_name, [])
    if not rows:
        raise SystemExit(
            f"export không có vector nào tên {embedding_name!r}. Dựng bằng:\n"
            f"  python -m scripts.embed_export_keyframes --kind jina "
            f"--model-path storage/models/jina-clip-v2 --embedding-name {embedding_name}"
        )
    encoder = LocalTextEncoder(model_path)
    # Warmup NGOÀI vòng đo: lần encode đầu nuốt thời gian nạp model, vượt
    # deadline nhánh, và `_raise_if_all_degraded` giết cả lượt chạy. Đã xảy ra
    # thật ở DENSE-TEXT-01 với E5. Warmup TRƯỚC khi bọc dịch: warmup chỉ để nạp
    # trọng số, không cần tốn một lời gọi LLM.
    encoder.warmup()
    if translate:
        encoder = wrap_translation(encoder)
    return DenseRetriever(
        encoder,
        InMemoryVectorStore(rows),
        # branch_id RIÊNG: hai nhánh visual dùng chung id thì fusion coi chúng
        # là một, phiếu của nhánh sau ghi đè nhánh trước.
        branch_id=f"dense_visual_{embedding_name}",
        backend_kind="vector",
    )


def _rebuild(svc) -> None:
    from online.services.registry import RetrieverRegistry
    from online.services.retrieval_orchestrator import RetrievalOrchestrator

    svc.registry = RetrieverRegistry(svc.retrievers)
    svc.orchestrator = RetrievalOrchestrator(svc.retrievers)


async def main_async(args: argparse.Namespace) -> None:
    repo = await JsonlSceneRepository.load(args.metadata)
    scenes = await repo.all()
    fps = load_fps(args.metadata)
    gold = [item for item in load_gold(args.gold) if item.task == TaskType.TRAKE]
    data_root = Path(os.getenv("AIC_DATA_ROOT", "storage")).resolve()

    clip = await build_visual(repo, data_root, args.clip_name, str(args.clip_model),
                              translate=args.translate_clip)
    # jina-clip-v2 KHÔNG bọc dịch: text tower của nó đa ngữ sẵn (đo được: phân
    # biệt tiếng Việt 0.260 so với tiếng Anh 0.262, và ghép đúng cặp dịch 0.820).
    # Bọc dịch cho nó chỉ thêm một lời gọi LLM và một nguồn lỗi, không thêm gì.
    jina = await build_visual(repo, data_root, args.jina_name, str(args.jina_model))
    print(f"CLIP dịch VI→EN: {args.translate_clip} | fusion: {args.fusion_method}")

    variants = (
        "A baseline",
        "B chi CLIP",
        "C chi jina-clip-v2",
        "D baseline thay jina",
        "E baseline + ca hai",
    )
    out: dict[str, dict] = {}
    print("=== VISUAL-01 tren Stage B that ===")
    for name in variants:
        svc = await build_service(
            "fusion", repo, backend="local", use_rules=False,
            use_expansion=False, use_query_prep=False, candidate_limit=100,
        )
        # Nhánh dense_visual mà build_service tự dựng đọc ref ĐẦU TIÊN (CLIP).
        # Bỏ nó ra rồi cắm lại tường minh, để mọi variant nói rõ nó đang chạy
        # model nào thay vì phụ thuộc thứ tự ref trong export.
        others = [r for r in svc.retrievers if getattr(r, "branch_id", "") != "dense_visual"]
        if name.startswith("A"):
            svc.retrievers = [*others, clip]
        elif name.startswith("B"):
            svc.retrievers = [clip]
        elif name.startswith("C"):
            svc.retrievers = [jina]
        elif name.startswith("D"):
            svc.retrievers = [*others, jina]
        else:
            svc.retrievers = [*others, clip, jina]
        # `build_service` KHÔNG truyền fusion_method nên nó luôn dựng với mặc
        # định `rrf` của SearchService — còn cấu hình chốt của dự án là
        # `norm_max` (docs/20 § Phase D, 09/08). Không đặt lại ở đây thì thí
        # nghiệm đo một hệ khác với hệ đang phục vụ, mà chẳng có gì báo.
        svc.fusion_method = args.fusion_method
        _rebuild(svc)
        out[name] = report(name, await measure(svc, scenes, fps, gold))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VISUAL-01: CLIP vs jina-clip-v2")
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_multivideo/scenes.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("examples/gold_all3.jsonl"))
    parser.add_argument("--clip-name", default="clip_vit_l14_v1")
    parser.add_argument("--clip-model", type=Path,
                        default=Path("storage/models/clip-vit-large-patch14"))
    parser.add_argument("--jina-name", default="jina_clip_v2")
    parser.add_argument("--jina-model", type=Path, default=Path("storage/models/jina-clip-v2"))
    parser.add_argument("--translate-clip", action="store_true",
                        help="Bọc TranslatingTextEncoder cho CLIP như container. "
                             "KHÔNG bật = số CLIP không so được với production.")
    parser.add_argument("--fusion-method", default="norm_max",
                        choices=("rrf", "norm_sum", "norm_max", "margin_sum", "entropy_sum"),
                        help="Mặc định norm_max = cấu hình đã chốt cho KIS/TRAKE/AVS")
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/evaluation/visual_01_clip_vs_jina.json"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
