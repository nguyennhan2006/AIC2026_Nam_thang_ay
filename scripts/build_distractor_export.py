"""EVAL-MULTIVIDEO-01 — dựng export nhiều video để có DISTRACTOR thật.

Vì sao cần: mọi con số đo được tới giờ đều trên MỘT video. `correct_video_rate`
luôn bằng 1.000 vì không có video nào khác để nhầm. Cuộc thi thật có ~800k
keyframe. Phần khó nhất của bài toán — phân biệt giữa các video khác nhau —
hiện chưa hề được thử thách, nên chưa biết bao nhiêu phần trong điểm số hiện
tại là thật và bao nhiêu là do không có đối thủ.

L21_V002/V003 chỉ có ảnh keyframe + CSV mapping (`n, pts_time, fps, frame_idx`),
KHÔNG có scene manifest và KHÔNG có ASR — khác V001. Nên scene ở đây được suy
ra từ chính lưới keyframe: mỗi keyframe mở một scene kéo tới keyframe kế tiếp.

Đó là xấp xỉ, và phải nói rõ: nó KHÔNG phải scene ngữ nghĩa do detector cắt.
Nhưng với vai trò distractor thì vừa đủ — cái cần là candidate cạnh tranh có
thật trong index, không phải ranh giới scene chính xác. Gold query đều thuộc
V001 nên V002/V003 không bao giờ là đáp án đúng.

Chạy::

    python -m scripts.build_distractor_export --video L21_V002 --video L21_V003 \\
        --base storage/exports_l21_enriched --out storage/exports_multivideo
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

SCHEMA_VERSION = "1.0.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_keyframe_grid(csv_path: Path) -> list[dict]:
    """Đọc `n, pts_time, fps, frame_idx` từ CSV mapping của BTC."""

    rows: list[dict] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "n": int(row["n"]),
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "frame_idx": int(row["frame_idx"]),
            })
    rows.sort(key=lambda item: item["frame_idx"])
    return rows


def copy_images(video_id: str, source_dir: Path, data_root: Path, grid: list[dict]) -> dict[int, str]:
    """Chép ảnh sang đúng quy ước `processed/keyframes/<video>/frame_%06d.jpg`.

    Giữ cùng quy ước với V001 để `image_path` trong keyframes.jsonl phân giải
    được bằng cùng một `data_root` — nếu không, VLM rerank và UI sẽ không mở
    được ảnh của riêng nhóm video mới.
    """

    target_dir = data_root / "processed" / "keyframes" / video_id
    target_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, str] = {}
    for entry in grid:
        # BTC đánh số ảnh theo `n` (1-based), không theo frame_idx.
        source = source_dir / f"{entry['n']:03d}.jpg"
        if not source.exists():
            continue
        name = f"frame_{entry['frame_idx']:06d}.jpg"
        destination = target_dir / name
        if not destination.exists():
            shutil.copy2(source, destination)
        mapping[entry["frame_idx"]] = f"processed/keyframes/{video_id}/{name}"
    return mapping


def image_size(path: Path) -> tuple[int, int]:
    """Kích thước thật của ảnh. Schema keyframe đòi > 0, không nhận 0 giả."""

    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.width, image.height
    except Exception:  # noqa: BLE001 - thiếu Pillow hoặc ảnh lạ
        return 1280, 720


def build_records(
    video_id: str,
    grid: list[dict],
    images: dict[int, str],
    total_frames: int,
    data_root: Path,
) -> tuple[list[dict], list[dict], dict]:
    scenes: list[dict] = []
    keyframes: list[dict] = []
    created = _now()
    fps = grid[0]["fps"] if grid else 30.0

    for index, entry in enumerate(grid):
        frame_idx = entry["frame_idx"]
        if frame_idx not in images:
            continue
        end_exclusive = (
            grid[index + 1]["frame_idx"] if index + 1 < len(grid) else total_frames
        )
        scene_id = f"{video_id}_S{index:04d}"
        keyframe_id = f"{scene_id}_F{frame_idx:06d}"

        width, height = image_size(data_root / images[frame_idx])
        keyframe = {
            "schema_version": SCHEMA_VERSION,
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "scene_id": scene_id,
            "frame_idx": frame_idx,
            "timestamp_sec": entry["pts_time"],
            "image_path": images[frame_idx],
            "width": width,
            "height": height,
            "selection_score": None,
            "quality": {},
            "roles": ["representative"],
            "captions": [],
            "ocr_instances": [],
            "objects": [],
            "action_tags": [],
            "color": {},
            "embedding_refs": [],
            "source_checksum": None,
            "created_at": created,
            "extensions": {},
        }
        keyframes.append(keyframe)

        scenes.append({
            "schema_version": SCHEMA_VERSION,
            "scene_id": scene_id,
            "video_id": video_id,
            "scene_idx": index,
            "start_frame": frame_idx,
            "end_frame_exclusive": max(end_exclusive, frame_idx + 1),
            "start_sec": entry["pts_time"],
            "end_sec": entry["pts_time"] + max(end_exclusive - frame_idx, 1) / fps,
            "keyframes": [keyframe],
            "captions": [],
            "keywords": [],
            "action_tags": [],
            "asr_segments": [],
            "embedding_refs": [],
            # Enum, không nhận None. "unknown" là giá trị đúng cho trường
            # hợp không có detector chuyển cảnh.
            "transition_in": "unknown",
            "transition_out": "unknown",
            "boundary_confidence_in": None,
            "boundary_confidence_out": None,
            "scene_clip_path": None,
            "scene_clip_checksum": None,
            # `model_name` ghi rõ scene này SUY RA từ lưới keyframe chứ không
            # do detector cắt — để sau này không ai đọc nhầm là ranh giới ngữ
            # nghĩa thật. Provenance có schema cố định, không nhận field tự chế.
            "segmentation_provenance": {
                "created_at": created,
                "device": "unknown",
                "model_name": "csv_keyframe_grid:scene-fallback",
                "model_revision": "btc-mapping-v1",
                "parameters": {},
                "pipeline_version": "aic-v1.0.0",
                "prompt_version": None,
            },
            "created_at": created,
            "extensions": {},
        })

    video = {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "fps": fps,
        "frame_count": total_frames,
        "duration_sec": total_frames / fps if fps else 0.0,
        "audio_present": False,
        "codec": None,
        "width": max((k["width"] for k in keyframes), default=1280),
        "height": max((k["height"] for k in keyframes), default=720),
        "source_checksum": None,
        # `JsonlSceneRepository` đọc trường này để phục vụ /v1/media, nên thiếu
        # nó là KeyError ngay lúc nạp — không phải trường trang trí.
        "source_path": f"raw/videos/{video_id}.mp4",
        "probe_provenance": {
            "created_at": created,
            "device": "unknown",
            "model_name": "csv_grid:build_distractor_export",
            "model_revision": "keyframe-grid-v1",
            "parameters": {},
            "pipeline_version": "aic-v1.0.0",
            "prompt_version": None,
        },
        "clips": [],
        "events": [],
        "created_at": created,
        "extensions": {},
    }
    return scenes, keyframes, video


def rewrite_manifest(out: Path) -> dict | None:
    """Tính lại `dataset_manifest.json` từ chính các file vừa ghi.

    Vì sao đây là lỗi thật, không phải chuyện hiển thị: `build_id` trong
    manifest được `/v1/health` trả ra dưới tên `dataset_version`, và nó tồn
    tại ĐÚNG để phân biệt "server cũ trỏ data mới" với "server mới trỏ data
    cũ". Copy manifest của export gốc làm nó nói dối, tức nó phản tác dụng
    hoàn toàn.

    Đo được trước khi sửa: manifest ghi `video_count 1, scene_count 217,
    keyframe_count 307, dataset_id l21-v001-real` trong khi export thật có 3
    video, 765 scene, 855 keyframe. `/v1/health` vì thế trộn số đúng
    (`scene_count` đọc từ repository) với số cũ trong cùng một phản hồi — tệ
    hơn là sai đều, vì nhìn vẫn hợp lý.
    """

    path = out / "dataset_manifest.json"
    if not path.exists():
        return None

    manifest = json.loads(path.read_text(encoding="utf-8"))
    scenes = [
        json.loads(line)
        for line in (out / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    videos = sorted({scene["video_id"] for scene in scenes})
    manifest.update({
        "dataset_id": f"multivideo-{len(videos)}",
        "build_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "video_count": len(videos),
        "scene_count": len(scenes),
        "keyframe_count": sum(len(scene.get("keyframes") or []) for scene in scenes),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {len(videos)} video, {len(scenes)} scene, "
          f"{manifest['keyframe_count']} keyframe, build_id={manifest['build_id']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng export nhiều video cho eval distractor")
    parser.add_argument("--video", action="append", required=True, help="vd L21_V002")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--base", type=Path, default=Path("storage/exports_l21_enriched"),
                        help="Export gốc chứa video có gold; được COPY sang out rồi nối thêm")
    parser.add_argument("--out", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # `dataset_manifest.json` được COPY từ export gốc rồi mới thêm video, nên
    # nó mô tả bản gốc chứ không mô tả bản vừa dựng. `rewrite_manifest` ở cuối
    # hàm sửa lại; xem docstring của nó để biết vì sao đây là lỗi thật chứ
    # không phải chuyện hiển thị.
    # Bắt đầu từ export gốc để video có gold giữ NGUYÊN caption/embedding đã có.
    # Dựng lại từ đầu là đánh mất toàn bộ enrichment và làm hỏng phép so sánh.
    for name in ("scenes.jsonl", "keyframes.jsonl", "videos.jsonl"):
        source = args.base / name
        if source.exists():
            shutil.copy2(source, args.out / name)
    for extra in ("dataset_manifest.json", "events.jsonl"):
        source = args.base / extra
        if source.exists():
            shutil.copy2(source, args.out / extra)

    added_scenes = added_keyframes = 0
    for video_id in args.video:
        csv_path = args.input_dir / f"{video_id}.csv"
        image_dir = args.input_dir / video_id
        if not csv_path.exists() or not image_dir.is_dir():
            raise SystemExit(f"thiếu {csv_path} hoặc {image_dir}")

        grid = read_keyframe_grid(csv_path)
        images = copy_images(video_id, image_dir, args.data_root, grid)
        # Không có metadata độ dài video: lấy frame cuối + một bước lưới.
        step = grid[1]["frame_idx"] - grid[0]["frame_idx"] if len(grid) > 1 else 90
        total_frames = grid[-1]["frame_idx"] + step

        scenes, keyframes, video = build_records(
            video_id, grid, images, total_frames, args.data_root
        )
        with (args.out / "scenes.jsonl").open("a", encoding="utf-8") as handle:
            for scene in scenes:
                handle.write(json.dumps(scene, ensure_ascii=False) + "\n")
        with (args.out / "keyframes.jsonl").open("a", encoding="utf-8") as handle:
            for keyframe in keyframes:
                handle.write(json.dumps(keyframe, ensure_ascii=False) + "\n")
        with (args.out / "videos.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(video, ensure_ascii=False) + "\n")

        added_scenes += len(scenes)
        added_keyframes += len(keyframes)
        print(f"{video_id}: {len(scenes)} scene, {len(keyframes)} keyframe, "
              f"{len(images)} ảnh đã chép")

    rewrite_manifest(args.out)

    print(f"\n-> {args.out}  (+{added_scenes} scene, +{added_keyframes} keyframe)")
    print("Bước tiếp: sinh embedding cho video mới, rồi caption nếu muốn chúng "
          "cạnh tranh cả ở nhánh BM25.")


if __name__ == "__main__":
    main()
