"""DENSE-TEXT-01: dựng caption document + index dense text bằng E5.

Vì sao cần nhánh này (docs/20_EXPERIMENT_LOG.md § TRAKE):

    gold_region_recall@100 của Stage B = 21/35
    query có ĐỦ mọi event retrieve được = 1/8

Nghĩa là 7/8 query TRAKE có ít nhất một bước mà vùng gold không bao giờ vào
pool. Hệ thống hiện có 4 nhánh BM25 lexical + 1 nhánh CLIP **ảnh**; không có
nhánh dense nào trên **text**. Mô tả một bước rất ngắn ("cột nước được quay
từ xa") nên khớp BM25 token quá giòn.

Document schema `caption_dense_v1` — CỐ Ý chưa có OCR/ASR:
ROUTE-01 cho thấy OCR/ASR có giá trị thật trong corpus này, nhưng trộn ngay
vào document dense thì không tách được gain đến từ semantic caption hay từ
lower-third bản tin. Thêm sau, ở một thí nghiệm riêng.

Hai encoder, chọn bằng `--encoder`. Cả hai đều bất đối xứng query-vs-passage,
chỉ khác cách khai, và bỏ qua phần bất đối xứng là mất phần lớn chất lượng —
lỗi im lặng, model vẫn chạy và vẫn trả số:

    e5       prefix chuỗi `query: ` / `passage: `
    jina_v3  LoRA adapter `retrieval.query` / `retrieval.passage`

Manifest ghi `encoder_kind` để phía online chặn được ca dùng nhầm họ. Chốt đó
KHÔNG thừa: cả hai model đều ra vector 1024 chiều nên `assert_dimension` cho
qua sạch.

Chạy::

    python -m scripts.build_caption_dense_index \\
        --metadata storage/exports_l21/scenes.jsonl \\
        --model storage/models/multilingual-e5-large \\
        --out storage/indexes_l21/caption_dense

    python -m scripts.build_caption_dense_index \\
        --encoder jina_v3 \\
        --model jinaai/jina-embeddings-v3 \\
        --model-id jinaai/jina-embeddings-v3 \\
        --out storage/indexes_multivideo/caption_dense_jina
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from online.adapters.json_metadata import JsonlSceneRepository

# Định nghĩa CHUNG với nhánh online. Nhân đôi encoder/prefix/schema ở đây là
# đường ngắn nhất tới lệch pooling giữa index và truy vấn — lỗi im lặng, model
# vẫn chạy và vẫn trả số. Re-export để script cũ import từ đây vẫn chạy.
from online.adapters.dense_text import (  # noqa: F401  (re-export có chủ đích)
    DOCUMENT_SCHEMA,
    ENCODER_KINDS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    E5Encoder,
    JinaV3Encoder,
    build_document_text,
    build_text_encoder,
    prefixes_for,
)


async def main_async(args: argparse.Namespace) -> None:
    repository = await JsonlSceneRepository.load(args.metadata)
    scenes = await repository.all()
    documents = [(scene.scene_id, build_document_text(scene)) for scene in scenes]
    usable = [(scene_id, text) for scene_id, text in documents if text.strip()]
    print(f"{len(usable)}/{len(documents)} scene có text dùng được")
    if not usable:
        raise SystemExit("không scene nào có caption/tag — chạy enrichment trước")

    # `for_passages=True`: phía dựng index luôn là passage side. Container dựng
    # cùng encoder với `for_passages=False`. Một factory duy nhất giữ hai phía
    # không bao giờ lệch cơ chế bất đối xứng.
    encoder = build_text_encoder(
        args.encoder, str(args.model), device=args.device, for_passages=True
    )
    query_prefix, passage_prefix = prefixes_for(args.encoder)
    print(f"encode {len(usable)} document ({args.encoder}, dim={encoder.dim}) …")
    matrix = encoder.encode([passage_prefix + text for _sid, text in usable], args.batch_size)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", matrix)
    (args.out / "scene_ids.json").write_text(
        json.dumps([scene_id for scene_id, _text in usable], ensure_ascii=False), encoding="utf-8"
    )
    fingerprint = hashlib.sha256(matrix.tobytes()).hexdigest()[:16]
    manifest = {
        "model_id": args.model_id,
        "model_path": str(args.model),
        "document_schema": DOCUMENT_SCHEMA,
        # Chốt chống lệch họ encoder ở phía online. Xem
        # CaptionDenseRetriever.assert_encoder_kind — e5 và jina_v3 cùng 1024
        # chiều nên chốt theo chiều không bắt được ca này.
        "encoder_kind": args.encoder,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        # Jina bất đối xứng bằng LoRA adapter chứ không bằng prefix; ghi lại để
        # đọc manifest là biết index dựng ở phía nào.
        "encoder_task": getattr(encoder, "task", None),
        "pooling": "mean",
        "normalized": True,
        "embedding_dim": int(matrix.shape[1]),
        "scene_count": int(matrix.shape[0]),
        "max_length": encoder.max_length,
        # `as_posix()` để thông báo lệch-corpus của CaptionDenseRetriever đọc
        # được trên mọi nền (Windows lưu `\\` làm chuỗi trong JSON rất khó đọc).
        "metadata_source": Path(args.metadata).as_posix(),
        "index_fingerprint": fingerprint,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"\n-> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build caption dense index (DENSE-TEXT-01)")
    # Mặc định trỏ corpus ĐANG PHỤC VỤ (3 video). Index dựng cho một export
    # khác vẫn nạp được và vẫn trả điểm hợp lệ, chỉ là không bao giờ đề xuất
    # nổi scene của video không có trong index — `CaptionDenseRetriever
    # .assert_covers` chặn ca đó lúc khởi động.
    parser.add_argument("--metadata", type=Path,
                        default=Path("storage/exports_multivideo/scenes.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("storage/models/multilingual-e5-large"))
    parser.add_argument("--model-id", default="intfloat/multilingual-e5-large")
    parser.add_argument("--encoder", choices=ENCODER_KINDS, default="e5",
                        help="e5 (prefix chuỗi) | jina_v3 (LoRA adapter theo task)")
    parser.add_argument("--out", type=Path,
                        default=Path("storage/indexes_multivideo/caption_dense"))
    parser.add_argument("--device", default="cpu", help="cpu | cuda — E5-large trên CPU vẫn chạy được")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
