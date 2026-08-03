"""Sinh embedding pack (visual dense) cho keyframe bằng CLIP local trên CPU.

Không cần FPT/GPU worker — đúng Finding 3 của kế hoạch PR-12..18: `.venv` đã
có `torch`+`transformers`, CLIP ViT-L/14 chạy được trên CPU cho một video.

Đọc `storage/packs/keyframe/manifests/keyframe_manifest.jsonl` (đã có
`frame_idx` THẬT từ PR trước), suy `image_path` tương đối `AIC_DATA_ROOT`, embed
từng keyframe theo batch, ghi:

  - `{data_root}/processed/embeddings/{video_id}/frame_{frame_idx:06d}.json`
    (vector JSON, dùng bởi `offline/indexing.py::frame_rows` qua `embedding_refs`)
  - `storage/packs/embedding/manifests/embedding_manifest.jsonl` (đúng
    `contracts/stage_pack.schema.json::embeddingRow`, có cả `vector` inline
    lẫn `embedding_refs` — `vector` inline để `offline/assemble.py` pool clip
    segment qua `PackEmbeddingReader`, `embedding_refs` để dựng frame index)

    python -m scripts.embed_keyframes_local
    python -m scripts.embed_keyframes_local --model-path storage/models/clip-vit-large-patch14

Máy dev này bị chặn SSL khi `transformers.from_pretrained` tự tải qua Python
`requests` (lỗi OpenSSL 3.x "Basic Constraints of CA cert not marked
critical" trên chứng chỉ chặn giữa của máy — `curl.exe`/Windows Schannel vẫn
tải được bình thường). Vì vậy weights đã được tải sẵn bằng `curl` vào
`storage/models/clip-vit-large-patch14/` — dùng `--model-path` để nạp từ đó
mà không cần gọi lại HF Hub. `--model-id` vẫn ghi "openai/clip-vit-large-
patch14" vào provenance vì đó là danh tính model thật, bất kể tải từ đâu.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasection.exporter import atomic_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _write_pack(pack_dir: Path, manifest_name: str, rows: list[dict], *, model: str, device: str) -> None:
    (pack_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (pack_dir / "_SUCCESS.json").write_text(
        json.dumps({"status": "success", "stage": "embedding", "count": len(rows)}, ensure_ascii=False),
        encoding="utf-8",
    )
    (pack_dir / "model_info.json").write_text(
        json.dumps(
            {"component": "embedding", "model": model, "pack_version": "clip-vit-l14-v1", "device": device},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    atomic_jsonl(pack_dir / "manifests" / manifest_name, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keyframes", type=Path, default=ROOT / "storage" / "packs" / "keyframe")
    parser.add_argument("--out", type=Path, default=ROOT / "storage" / "packs" / "embedding")
    parser.add_argument("--data-root", type=Path, default=Path(os.getenv("AIC_DATA_ROOT", ROOT / "storage")))
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14", help="Danh tính model ghi vào provenance")
    parser.add_argument(
        "--model-path", default=os.getenv("AIC_VISUAL_EMBEDDING_MODEL", "openai/clip-vit-large-patch14"),
        help="Đường dẫn/local dir hoặc HF repo id thực sự dùng để nạp model",
    )
    parser.add_argument("--embedding-name", default="clip_vit_l14_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    manifest_path = args.keyframes / "manifests" / "keyframe_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"keyframe manifest rỗng: {manifest_path}")

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Nạp CLIP từ {args.model_path!r} (device={device})...")
    model = CLIPModel.from_pretrained(args.model_path).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model_path)

    out_rows: list[dict] = []
    total = len(rows)
    for start in range(0, total, args.batch_size):
        batch = rows[start : start + args.batch_size]
        images = [Image.open(args.data_root / row["image_path"]).convert("RGB") for row in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            vectors = model.get_image_features(**inputs)
            vectors = vectors / vectors.norm(p=2, dim=-1, keepdim=True)
        vectors = vectors.detach().cpu().float().tolist()
        for row, vector in zip(batch, vectors):
            video_id = str(row["video_id"])
            frame_idx = int(row["frame_idx"])
            relative = f"processed/embeddings/{video_id}/frame_{frame_idx:06d}.json"
            _atomic_write_json(args.data_root / relative, vector)
            out_rows.append({
                "video_id": video_id,
                "frame_idx": frame_idx,
                "vector": vector,
                "embedding_refs": [{
                    "embedding_name": args.embedding_name,
                    "modality": "image",
                    "model_name": args.model_id,
                    "model_revision": None,
                    "dimension": len(vector),
                    "normalized": True,
                    "storage_locations": [{
                        "backend": "file",
                        "vector_id": f"{video_id}_F{frame_idx:06d}",
                        "index_name": args.embedding_name,
                        "vector_uri": relative,
                    }],
                }],
            })
        print(f"  {min(start + args.batch_size, total)}/{total} keyframe đã embed")

    _write_pack(args.out, "embedding_manifest.jsonl", out_rows, model=args.model_id, device=device)
    print(f"embedding: {len(out_rows)} dòng -> {args.out}")


if __name__ == "__main__":
    main()
