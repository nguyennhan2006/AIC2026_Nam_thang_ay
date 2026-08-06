"""Sinh embedding CLIP cho keyframe của một EXPORT (không phải stage pack).

Khác `scripts/embed_keyframes_local.py`: script đó đọc
`storage/packs/keyframe/manifests/keyframe_manifest.jsonl` — đúng cho luồng
offline đầy đủ. Ở đây đầu vào là export đã dựng sẵn
(`scenes.jsonl` + `keyframes.jsonl`), vì video distractor L21_V002/V003 chỉ có
ảnh + CSV chứ không đi qua pipeline stage pack.

Ghi ra đúng hai thứ mà tầng online cần:

1. `{data_root}/processed/embeddings/{video}/frame_{idx:06d}.json` — vector thô
2. `embedding_refs` trong `keyframes.jsonl` VÀ trong `scene["keyframes"]` của
   `scenes.jsonl`

Điểm dễ sai: `JsonlSceneRepository` đọc keyframe **lồng trong scene**, không
đọc `keyframes.jsonl`. Cập nhật một file mà quên file kia thì
`build_frame_vector_rows` vẫn báo 0 vector và nhánh dense im lặng bỏ qua video
mới — đúng kiểu hỏng âm thầm đã gặp nhiều lần trong dự án này.

Bỏ qua keyframe đã có `embedding_refs`, nên chạy lại là rẻ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EMBEDDING_NAME = "clip_vit_l14_v1"


def load_model(model_path: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_path)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_path)
    return torch, model, processor


def embed_images(torch, model, processor, paths: list[Path], batch_size: int) -> list[list[float]]:
    from PIL import Image

    vectors: list[list[float]] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # Chuẩn hoá L2: index dùng cosine similarity, và text tower cũng được
        # chuẩn hoá — lệch một bên là điểm số vô nghĩa mà vẫn chạy.
        features = features / features.norm(dim=-1, keepdim=True)
        vectors.extend(features.tolist())
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return vectors


def make_ref(video_id: str, frame_idx: int, dimension: int, model_id: str) -> dict:
    uri = f"processed/embeddings/{video_id}/frame_{frame_idx:06d}.json"
    return {
        "dimension": dimension,
        "embedding_name": EMBEDDING_NAME,
        "modality": "image",
        "model_name": model_id,
        "model_revision": None,
        "normalized": True,
        "storage_locations": [
            {
                "backend": "file",
                "index_name": EMBEDDING_NAME,
                "vector_id": f"{video_id}_F{frame_idx:06d}",
                "vector_uri": uri,
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed keyframe của một export")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--model-path", default="storage/models/clip-vit-large-patch14")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--video", action="append", default=[],
                        help="Chỉ embed các video này; bỏ trống = mọi keyframe còn thiếu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    keyframes_path = args.export / "keyframes.jsonl"
    scenes_path = args.export / "scenes.jsonl"
    keyframes = [json.loads(line) for line in keyframes_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    wanted = set(args.video)
    todo = [
        row for row in keyframes
        if not row.get("embedding_refs")
        and (not wanted or row["video_id"] in wanted)
        and (args.data_root / row["image_path"]).exists()
    ]
    if not todo:
        print("không có keyframe nào cần embed")
        return
    print(f"cần embed {len(todo)} keyframe")

    torch, model, processor = load_model(args.model_path)
    vectors = embed_images(
        torch, model, processor,
        [args.data_root / row["image_path"] for row in todo],
        args.batch_size,
    )

    refs_by_id: dict[str, dict] = {}
    for row, vector in zip(todo, vectors, strict=True):
        out_path = args.data_root / "processed" / "embeddings" / row["video_id"] / \
            f"frame_{row['frame_idx']:06d}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(vector), encoding="utf-8")
        ref = make_ref(row["video_id"], row["frame_idx"], len(vector), args.model_id)
        row["embedding_refs"] = [ref]
        refs_by_id[row["keyframe_id"]] = ref

    keyframes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keyframes),
        encoding="utf-8",
    )

    # Bắt buộc: repository đọc keyframe LỒNG trong scene. Quên bước này thì
    # vector vẫn nằm trên đĩa mà nhánh dense không thấy video mới.
    scenes = [json.loads(line) for line in scenes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    touched = 0
    for scene in scenes:
        for keyframe in scene.get("keyframes", []):
            ref = refs_by_id.get(keyframe.get("keyframe_id"))
            if ref is not None:
                keyframe["embedding_refs"] = [ref]
                touched += 1
    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes),
        encoding="utf-8",
    )
    print(f"đã ghi {len(todo)} vector, cập nhật {touched} keyframe lồng trong scene")


if __name__ == "__main__":
    main()
