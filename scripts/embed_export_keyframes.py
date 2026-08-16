"""Sinh embedding ẢNH cho keyframe của một EXPORT (không phải stage pack).

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

Bỏ qua keyframe đã có ref MANG ĐÚNG `--embedding-name`, nên chạy lại là rẻ, và
chạy model thứ hai không đụng gì tới model thứ nhất.

Hai họ model (`--kind`):

    clip  CLIPModel.get_image_features(**processor(images=...))
    jina  AutoModel(trust_remote_code=True).encode_image([PIL, ...])

`jina` ở đây là jina-clip (v1/v2) — model ĐA PHƯƠNG THỨC có image tower, khác
hẳn `jina-embeddings-v3` vốn chỉ có text và không index được ảnh. Hai thứ tên
gần giống nhau, cắm vào hai slot khác nhau.

Vector của mỗi embedding_name nằm ở THƯ MỤC RIÊNG
(`processed/embeddings/{embedding_name}/…`, trừ `clip_vit_l14_v1` giữ đường cũ
để không phá export đang phục vụ). Dùng chung đường dẫn thì model chạy sau ghi
đè vector của model chạy trước, `embedding_refs` vẫn trỏ đúng tên, và cosine
vẫn ra số — hỏng hoàn toàn im lặng.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EMBEDDING_NAME = "clip_vit_l14_v1"
KINDS = ("clip", "jina")


def load_model(model_path: str, kind: str):
    import torch

    if kind == "jina":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        return torch, model, None

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_path)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_path)
    return torch, model, processor


def embed_images(torch, model, processor, paths: list[Path], batch_size: int,
                 kind: str = "clip") -> list[list[float]]:
    from PIL import Image

    vectors: list[list[float]] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch]
        with torch.no_grad():
            if kind == "jina":
                # jina-clip tự lo preprocessing; `encode_image` trả numpy.
                raw = model.encode_image(images, batch_size=len(images))
                features = torch.as_tensor(raw, dtype=torch.float32)
            else:
                inputs = processor(images=images, return_tensors="pt")
                features = model.get_image_features(**inputs)
        # Chuẩn hoá L2: index dùng cosine similarity, và text tower cũng được
        # chuẩn hoá — lệch một bên là điểm số vô nghĩa mà vẫn chạy. Chuẩn hoá
        # lại kể cả khi thư viện nói đã chuẩn hoá: bất biến này phải đúng bất
        # kể mặc định của bên thứ ba đổi ra sao.
        features = features / features.norm(dim=-1, keepdim=True)
        vectors.extend(features.tolist())
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return vectors


def vector_uri(video_id: str, frame_idx: int, embedding_name: str) -> str:
    """Đường dẫn vector, TÁCH theo embedding_name.

    `clip_vit_l14_v1` giữ nguyên đường cũ vì export hiện tại đã trỏ vào đó;
    đổi sẽ làm mọi ref đang có thành đường dẫn chết.
    """

    if embedding_name == EMBEDDING_NAME:
        return f"processed/embeddings/{video_id}/frame_{frame_idx:06d}.json"
    return f"processed/embeddings/{embedding_name}/{video_id}/frame_{frame_idx:06d}.json"


def make_ref(video_id: str, frame_idx: int, dimension: int, model_id: str,
             embedding_name: str) -> dict:
    return {
        "dimension": dimension,
        "embedding_name": embedding_name,
        "modality": "image",
        "model_name": model_id,
        "model_revision": None,
        "normalized": True,
        "storage_locations": [
            {
                "backend": "file",
                "index_name": embedding_name,
                "vector_id": f"{video_id}_F{frame_idx:06d}",
                "vector_uri": vector_uri(video_id, frame_idx, embedding_name),
            }
        ],
    }


def merge_ref(existing: list[dict] | None, ref: dict) -> list[dict]:
    """Thêm/thay ref theo `embedding_name`, GIỮ các ref của model khác.

    `build_frame_vector_rows_by_index` dựng một vector store cho mỗi tên, nên
    nhiều ref cùng tồn tại là đúng thiết kế. Gán đè cả danh sách (bản cũ làm
    vậy) sẽ xoá vector CLIP ngay khi chạy model thứ hai.
    """

    kept = [item for item in (existing or []) if item.get("embedding_name") != ref["embedding_name"]]
    return [*kept, ref]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed keyframe của một export")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--model-path", default="storage/models/clip-vit-large-patch14")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--kind", choices=KINDS, default="clip")
    parser.add_argument("--embedding-name", default=EMBEDDING_NAME)
    parser.add_argument("--video", action="append", default=[],
                        help="Chỉ embed các video này; bỏ trống = mọi keyframe còn thiếu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=64,
                        help="Ghi export sau mỗi ngần ấy keyframe; 0 = chỉ ghi ở cuối")
    args = parser.parse_args()

    keyframes_path = args.export / "keyframes.jsonl"
    scenes_path = args.export / "scenes.jsonl"
    keyframes = [json.loads(line) for line in keyframes_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def has_this_embedding(row: dict) -> bool:
        return any(
            ref.get("embedding_name") == args.embedding_name
            for ref in (row.get("embedding_refs") or [])
        )

    wanted = set(args.video)
    todo = [
        row for row in keyframes
        # Lọc theo TÊN chứ không theo "có ref nào chưa": bản cũ lọc kiểu sau nên
        # keyframe đã có CLIP sẽ bị bỏ qua, và model thứ hai không embed nổi
        # dòng nào — script kết thúc sạch sẽ với "không có keyframe nào cần
        # embed" mà chẳng làm gì.
        if not has_this_embedding(row)
        and (not wanted or row["video_id"] in wanted)
        and (args.data_root / row["image_path"]).exists()
    ]
    if not todo:
        print(f"không có keyframe nào cần embed cho {args.embedding_name!r}")
        return
    print(f"cần embed {len(todo)} keyframe ({args.kind}, {args.embedding_name})")

    torch, model, processor = load_model(args.model_path, args.kind)

    refs_by_id: dict[str, dict] = {}

    def flush() -> None:
        """Ghi cả hai file export. Gọi định kỳ, không chỉ ở cuối.

        jina-clip-v2 trên CPU mất ~9.6s/ảnh, tức hơn hai tiếng cho 855 keyframe.
        Chỉ ghi ở cuối thì một lần tắt máy hay OOM là mất sạch. Vector từng ảnh
        đã nằm trên đĩa ngay khi encode xong, nên flush chỉ cần đồng bộ phần
        `embedding_refs` — và vì `todo` lọc theo ref đã có, chạy lại sau khi
        gián đoạn sẽ tiếp đúng chỗ dừng thay vì làm lại từ đầu.
        """

        keyframes_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keyframes),
            encoding="utf-8",
        )
        scenes = [json.loads(line) for line in
                  scenes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for scene in scenes:
            for keyframe in scene.get("keyframes", []):
                ref = refs_by_id.get(keyframe.get("keyframe_id"))
                if ref is not None:
                    keyframe["embedding_refs"] = merge_ref(keyframe.get("embedding_refs"), ref)
        scenes_path.write_text(
            "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes),
            encoding="utf-8",
        )

    done = 0
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        vectors = embed_images(
            torch, model, processor,
            [args.data_root / row["image_path"] for row in chunk],
            args.batch_size,
            args.kind,
        )
        for row, vector in zip(chunk, vectors, strict=True):
            out_path = args.data_root / vector_uri(
                row["video_id"], row["frame_idx"], args.embedding_name
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(vector), encoding="utf-8")
            ref = make_ref(row["video_id"], row["frame_idx"], len(vector), args.model_id,
                           args.embedding_name)
            row["embedding_refs"] = merge_ref(row.get("embedding_refs"), ref)
            refs_by_id[row["keyframe_id"]] = ref
        done += len(chunk)
        if args.checkpoint_every and done % args.checkpoint_every < args.batch_size:
            flush()
            print(f"  [checkpoint] {done}/{len(todo)}", flush=True)
    # `flush()` đã cập nhật keyframe LỒNG trong scene — bắt buộc, vì repository
    # đọc scene chứ không đọc keyframes.jsonl. Quên bước đó thì vector vẫn nằm
    # trên đĩa mà nhánh dense không thấy gì.
    flush()
    print(f"đã ghi {len(todo)} vector cho {args.embedding_name!r} "
          f"({args.model_id}), cập nhật cả keyframes.jsonl lẫn scenes.jsonl")


if __name__ == "__main__":
    main()
