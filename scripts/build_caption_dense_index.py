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

E5 yêu cầu prefix bất đối xứng: `query: ` cho truy vấn, `passage: ` cho tài
liệu. Bỏ prefix là mất phần lớn chất lượng — đây là lỗi im lặng, model vẫn
chạy và vẫn trả số.

Chạy::

    python -m scripts.build_caption_dense_index \\
        --metadata storage/exports_l21/scenes.jsonl \\
        --model storage/models/multilingual-e5-large \\
        --out storage/indexes_l21/caption_dense
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from online.adapters.json_metadata import JsonlSceneRepository
from online.domain.models import SceneDocument

DOCUMENT_SCHEMA = "caption_dense_v1"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def build_document_text(scene: SceneDocument) -> str:
    """Một chuỗi text tìm kiếm được cho mỗi scene.

    Thứ tự cố ý: caption trước (mang nhiều ngữ nghĩa nhất), rồi tới tag. Không
    lặp field để giả trọng số — trọng số là việc của tầng fusion.
    """

    parts: list[str] = []
    if scene.captions:
        parts.append(" ".join(scene.captions))
    for values in (scene.object_labels, scene.action_tags, scene.keywords):
        if values:
            parts.append(", ".join(dict.fromkeys(values)))
    return " | ".join(part for part in parts if part.strip())


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).float()
    return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)


class E5Encoder:
    """Encoder E5 chạy local, luôn L2-normalize để cosine = dot product."""

    def __init__(self, model_path: str, device: str = "cpu", max_length: int = 320) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            pooled = mean_pool(hidden, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype="float32")


async def main_async(args: argparse.Namespace) -> None:
    repository = await JsonlSceneRepository.load(args.metadata)
    scenes = await repository.all()
    documents = [(scene.scene_id, build_document_text(scene)) for scene in scenes]
    usable = [(scene_id, text) for scene_id, text in documents if text.strip()]
    print(f"{len(usable)}/{len(documents)} scene có text dùng được")
    if not usable:
        raise SystemExit("không scene nào có caption/tag — chạy enrichment trước")

    encoder = E5Encoder(args.model, device=args.device)
    print(f"encode {len(usable)} document (dim={encoder.dim}) …")
    matrix = encoder.encode([PASSAGE_PREFIX + text for _sid, text in usable], args.batch_size)

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
        "query_prefix": QUERY_PREFIX,
        "passage_prefix": PASSAGE_PREFIX,
        "pooling": "mean",
        "normalized": True,
        "embedding_dim": int(matrix.shape[1]),
        "scene_count": int(matrix.shape[0]),
        "max_length": encoder.max_length,
        "metadata_source": str(args.metadata),
        "index_fingerprint": fingerprint,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"\n-> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build caption dense index (DENSE-TEXT-01)")
    parser.add_argument("--metadata", type=Path, default=Path("storage/exports_l21/scenes.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("storage/models/multilingual-e5-large"))
    parser.add_argument("--model-id", default="intfloat/multilingual-e5-large")
    parser.add_argument("--out", type=Path, default=Path("storage/indexes_l21/caption_dense"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
