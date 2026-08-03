"""Dựng stage pack (video/scene/keyframe/asr) từ dữ liệu thật L21_V001 ở `input/`.

`input/` hiện có (do người dùng cung cấp, không phải notebook Kaggle):

    input/scene_manifest.jsonl      336 scene (TransNetV2), end_frame inclusive
    input/asr_segments.jsonl        264 segment (faster-whisper large-v3)
    input/mapping_L21_V001.csv      n,pts_time,fps,frame_idx — map keyframe ordinal -> frame_idx THẬT
    input/L21_V001/001.jpg..307.jpg keyframe, đặt tên theo ordinal `n` trong mapping
    input/L21_V001.mp4               video gốc

`mapping_L21_V001.csv` là mảnh còn thiếu đã chặn việc dùng dữ liệu này: 307 file
jpg không có EXIF/frame_idx nhúng vào tên, và không map 1:1 với 336 scene
(217 scene có keyframe, 119 scene không có — sẽ bị `assemble` quarantine, đúng
hành vi "báo rõ thay vì âm thầm bỏ qua").

Script này CHỈ dịch định dạng — không suy đoán, không sinh nội dung caption/
OCR (chờ FPT VLM ở PR-13). Chạy xong sẽ có 4/9 stage pack (thiếu caption/ocr/
object/color/embedding), đủ để `offline assemble` dựng canonical dataset THẬT
với `frame_idx` đúng — kiểm chứng được toàn bộ đường ống trước khi có key FPT.

    python -m scripts.build_l21_stage_packs
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from datasection.exporter import atomic_jsonl

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
STORAGE = ROOT / "storage"
PACKS = STORAGE / "packs"

VIDEO_ID = "L21_V001"
# Từ examples/AIC2026_L21_V001_query_schema.json — nguồn đáng tin theo thứ tự
# ưu tiên §1 của AIC2026_FPT_API_SINGLE_VIDEO_TEST_TUNING_GUIDE.md (gold
# benchmark đã version hóa, chỉ sau luật BTC). scene_manifest.jsonl (max
# end_frame + 1 = 37849) khớp đúng total_frames ở đây — hai nguồn xác nhận
# lẫn nhau, không phải suy đoán một chiều.
VIDEO_FPS = 30.0
VIDEO_FRAME_COUNT = 37849
VIDEO_DURATION_SEC = 1261.725896
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


def _write_pack(stage: str, manifest_name: str, rows: list[dict], *, model: str) -> Path:
    pack_dir = PACKS / stage
    (pack_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (pack_dir / "_SUCCESS.json").write_text(
        json.dumps({"status": "success", "stage": stage, "count": len(rows)}, ensure_ascii=False),
        encoding="utf-8",
    )
    (pack_dir / "model_info.json").write_text(
        json.dumps({"component": stage, "model": model, "pack_version": "l21-input-v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    atomic_jsonl(pack_dir / "manifests" / manifest_name, rows)
    print(f"  {stage}: {len(rows)} dòng -> {pack_dir}")
    return pack_dir


def build_video_pack() -> None:
    row = {
        "video_id": VIDEO_ID,
        "source_path": "raw/videos/L21_V001.mp4",
        "fps": VIDEO_FPS,
        "frame_count": VIDEO_FRAME_COUNT,
        "duration_sec": VIDEO_DURATION_SEC,
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "codec": "h264",
        "audio_present": True,
    }
    _write_pack("video", "video_manifest.jsonl", [row], model="ffprobe")

    target = STORAGE / "raw" / "videos" / f"{VIDEO_ID}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = INPUT / f"{VIDEO_ID}.mp4"
    if source.exists() and not target.exists():
        shutil.copy2(source, target)
        print(f"  video file -> {target}")


def build_scene_pack() -> None:
    rows = [
        json.loads(line)
        for line in (INPUT / "scene_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Đúng contracts/stage_pack.schema.json: cần scene_index (input đã có
    # scene_index đúng tên), start_frame, end_frame (inclusive). Giữ nguyên,
    # không cần dịch field.
    for row in rows:
        assert row["video_id"] == VIDEO_ID
    _write_pack("scene", "scene_manifest.jsonl", rows, model="transnetv2_pytorch")


def build_asr_pack() -> None:
    rows = [
        json.loads(line)
        for line in (INPUT / "asr_segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_pack("asr", "asr_segments.jsonl", rows, model="faster-whisper:large-v3")


def build_keyframe_pack() -> None:
    """Đọc mapping_L21_V001.csv để lấy frame_idx THẬT, copy jpg sang vị trí
    canonical `processed/keyframes/{video_id}/frame_{frame_idx:06d}.jpg` —
    đây trở thành `image_path` chính thức, không phải tên file gốc `NNN.jpg`
    (đó chỉ là thứ tự trích xuất, không phải frame_idx — xem docstring trên)."""

    mapping_path = INPUT / f"mapping_{VIDEO_ID}.csv"
    with mapping_path.open(encoding="utf-8") as handle:
        mapping = list(csv.DictReader(handle))

    keyframe_dir = INPUT / VIDEO_ID
    target_dir = STORAGE / "processed" / "keyframes" / VIDEO_ID
    target_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing_source: list[str] = []
    for entry in mapping:
        ordinal = int(entry["n"])
        frame_idx = int(entry["frame_idx"])
        source = keyframe_dir / f"{ordinal:03d}.jpg"
        if not source.exists():
            missing_source.append(source.name)
            continue
        relative = f"processed/keyframes/{VIDEO_ID}/frame_{frame_idx:06d}.jpg"
        target = STORAGE / relative
        if not target.exists():
            shutil.copy2(source, target)
        rows.append({
            "video_id": VIDEO_ID,
            "frame_idx": frame_idx,
            "timestamp_sec": float(entry["pts_time"]),
            "image_path": relative,
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "roles": ["representative"],
        })
    if missing_source:
        raise FileNotFoundError(
            f"mapping tham chiếu {len(missing_source)} file jpg không tồn tại: {missing_source[:5]}"
        )
    rows.sort(key=lambda item: item["frame_idx"])
    _write_pack("keyframe", "keyframe_manifest.jsonl", rows, model="ffmpeg-scene-representative")


def main() -> None:
    print(f"Dựng stage pack cho {VIDEO_ID} từ {INPUT} -> {PACKS}")
    build_video_pack()
    build_scene_pack()
    build_asr_pack()
    build_keyframe_pack()
    print(
        "\nXong 4/9 stage pack (video/scene/asr/keyframe). Thiếu caption/ocr/object/"
        "color/embedding — chờ PR-13 (FPT VLM) và embedding local. `offline assemble`"
        " vẫn chạy được với 4 pack này (chỉ cảnh báo thiếu, không lỗi)."
    )


if __name__ == "__main__":
    main()
