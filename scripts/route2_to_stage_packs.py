"""Chuyển đầu ra Route2 (VastAI) thành stage pack cho `offline assemble`.

**Chỉ dịch định dạng.** Mọi kiến thức về contract — sinh `scene_id`/
`keyframe_id`, provenance, gom keyframe vào scene, cắt ASR vào biên, quarantine
scene không có keyframe — nằm ở `offline/assemble.py`, là đường đã dựng ra
L21_V001 và đang chạy ổn định. Script này không được lặp lại chúng.

Bản đầu tiên của tôi đã viết một exporter tự chứa dựng thẳng `scenes.jsonl`.
Nó chạy được, nhưng nhân bản contract ra chỗ thứ hai: sáu ràng buộc (enum
`caption_type`, `evidence_keyframe_ids`, bbox pixel-vs-normalized, biên ASR,
`scene_id` 4 hay 5 chữ số, và `start_sec <= timestamp_sec < end_sec`) đều phải
tự cài lại — và cái thứ sáu chỉ lộ ra khi pydantic của `datasection` từ chối,
sau khi JSON Schema đã báo sạch. Bản này bỏ hết chỗ đó cho `assemble`.

Stage pack là JSONL phẳng, khoá join `(video_id, frame_idx)`:

    caption   {"video_id", "frame_idx", "captions":[{caption_type, language, text}]}
    ocr       {"video_id", "frame_idx", "instances":[{bbox, language, text}]}
    object    {"video_id", "frame_idx", "objects":[{bbox, confidence, label}]}
    color     {"video_id", "frame_idx", "mean_hsv", "hsv_histogram", "dominant_hex"}

    python -m scripts.route2_to_stage_packs \
        --route2 out/scene_index_ready.json \
        --map-keyframes input/map-keyframes \
        --packs storage/packs
    python -m offline assemble --packs storage/packs --out storage/exports_full
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PACK_VERSION = "route2-v1"

# Enum của caption mức KEYFRAME. `visual` là của scene — `assemble` tự sinh
# caption scene, ở đây tuyệt đối không dùng nó.
KEYFRAME_CAPTION_TYPES = {"short", "detailed", "tags", "crop"}


def load_map_keyframes(directory: Path | None) -> dict[str, dict[int, int]]:
    """`video_id -> {n -> frame_idx}`.

    Nguồn DUY NHẤT đúng cho `frame_idx`. Tên file ảnh (`001.jpg`) là thứ tự
    keyframe trong video, không phải frame thật — suy từ đó là nộp sai frame.
    """

    if not directory or not directory.exists():
        return {}
    out: dict[str, dict[int, int]] = {}
    for path in sorted(directory.glob("*.csv")):
        rows: dict[int, int] = {}
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows[int(row["n"])] = int(row["frame_idx"])
                except (KeyError, TypeError, ValueError):
                    continue
        if rows:
            out[path.stem] = rows
    return out


def normalise_bbox(bbox: Any, width: int, height: int) -> dict | None:
    """Đưa bbox về [0,1]; `None` nếu không dùng được.

    VLM trả toạ độ PIXEL dù prompt yêu cầu normalized. Bản đầu kẹp bằng
    `min(max(v,0),1)` và điều đó phá sạch mọi bbox — mọi giá trị > 1 thành 1.0,
    tức mọi khung dồn về cạnh phải/dưới. Phải PHÁT HIỆN rồi CHIA.
    """

    if isinstance(bbox, dict):
        try:
            values = [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            values = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
    else:
        return None

    x1, y1, x2, y2 = values
    if max(values) > 1.0:
        if width <= 0 or height <= 0:
            return None
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if not all(-0.01 <= v <= 1.01 for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    clamp = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return {"x1": clamp(x1), "y1": clamp(y1), "x2": clamp(x2), "y2": clamp(y2)}


def map_hsv(hsv: dict) -> dict | None:
    """Đưa `extract_hsv_features` của Route2 về đúng shape mà `assemble` đọc.

    Hai bên không khớp, và sai chỗ này thì nhánh `color_search` rỗng **im
    lặng** — đúng lỗi A3 trong `docs/27` (0/855 keyframe có màu).

        Route2 trả:  hue_hist, sat_hist, val_hist (ba mảng RỜI),
                     dominant_hue_deg, mean_saturation, mean_brightness
        assemble đọc: hsv_histogram (MỘT mảng) hoặc dominant_colors,
                      cộng mean_hsv (bộ ba)

    Điều kiện bật màu ở `offline/assemble.py` là
    `if color_row.get("dominant_colors") or color_row.get("hsv_histogram")`.
    Chỉ có `mean_hsv` và `dominant_hex` thì bị bỏ qua hoàn toàn.

    `mean_hsv` chuẩn hoá về [0,1]: hue chia 360, sat/val chia 255 (OpenCV lưu
    8-bit). Route2 đã nhân 2 để đổi hue từ [0,180) của OpenCV về độ thật.
    """

    histogram = (
        list(hsv.get("hue_hist") or [])
        + list(hsv.get("sat_hist") or [])
        + list(hsv.get("val_hist") or [])
    ) or list(hsv.get("hsv_histogram") or hsv.get("histogram") or [])
    if not histogram and not hsv.get("dominant_colors"):
        return None

    mean_hsv = hsv.get("mean_hsv")
    if not mean_hsv and hsv.get("dominant_hue_deg") is not None:
        mean_hsv = [
            round(float(hsv["dominant_hue_deg"]) / 360.0, 4),
            round(float(hsv.get("mean_saturation") or 0.0) / 255.0, 4),
            round(float(hsv.get("mean_brightness") or 0.0) / 255.0, 4),
        ]
    return {
        "hsv_histogram": histogram,
        "mean_hsv": mean_hsv,
        "dominant_hex": hsv.get("dominant_hex") or [],
        "dominant_colors": hsv.get("dominant_colors") or [],
    }


def write_pack(root: Path, stage: str, manifest: str, rows: list[dict], model: str) -> None:
    """Ghi một stage pack đúng bố cục mà `offline.stagepack` chờ đợi."""

    pack = root / stage
    (pack / "manifests").mkdir(parents=True, exist_ok=True)
    with (pack / "manifests" / manifest).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (pack / "model_info.json").write_text(json.dumps({
        "component": stage, "model": model, "pack_version": PACK_VERSION,
    }, ensure_ascii=False), encoding="utf-8")
    (pack / "_SUCCESS.json").write_text(json.dumps({
        "status": "success", "stage": stage, "count": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"  {stage:8s} {len(rows):>6} dòng -> {pack}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route2", type=Path, required=True,
                        help="scene_index_ready.json do Route2 xuất")
    parser.add_argument("--map-keyframes", type=Path,
                        help="thư mục CSV map-keyframes (nguồn đúng của frame_idx)")
    parser.add_argument("--packs", type=Path, default=Path("storage/packs"))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--default-size", default="1280x720",
                        help="kích thước ảnh khi Route2 không ghi, để quy bbox pixel")
    args = parser.parse_args()

    default_w, _, default_h = args.default_size.partition("x")
    payload = json.loads(args.route2.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("rows") or []
    maps = load_map_keyframes(args.map_keyframes)

    captions, ocr, objects, colors = [], [], [], []
    skipped_no_frame = skipped_bbox = 0

    for row in rows:
        if row.get("record_type") != "keyframe":
            continue          # scene_context do `assemble` tự tổng hợp lại
        video_id = row.get("video_id")
        if not video_id:
            continue

        frame_idx = row.get("frame_idx")
        if frame_idx is None:
            n = row.get("keyframe_n")
            frame_idx = maps.get(video_id, {}).get(n) if n is not None else None
        if frame_idx is None:
            skipped_no_frame += 1
            continue
        frame_idx = int(frame_idx)
        key = {"video_id": video_id, "frame_idx": frame_idx}
        width = int(row.get("width") or default_w)
        height = int(row.get("height") or default_h)

        items = []
        for caption_type, field in (("short", "short_caption_vi"),
                                    ("detailed", "detailed_caption_vi")):
            text = (row.get(field) or "").strip()
            if text:
                items.append({"caption_type": caption_type, "language": "vi", "text": text})
        tags = [str(t).strip() for t in (row.get("keywords_vi") or []) if str(t).strip()]
        if tags:
            items.append({"caption_type": "tags", "language": "vi", "text": ", ".join(tags)})
        if items:
            captions.append({**key, "captions": items})

        instances = []
        for region in row.get("ocr_regions") or []:
            text = (region.get("text") or region.get("text_vi") or "").strip()
            bbox = normalise_bbox(region.get("bbox"), width, height)
            if not text:
                continue
            if bbox is None:
                skipped_bbox += 1
                continue
            instances.append({"bbox": bbox, "language": region.get("language") or "vi",
                              "text": text})
        if instances:
            ocr.append({**key, "instances": instances})

        detected = []
        for entity in row.get("entities") or []:
            label = (entity.get("label_vi") or entity.get("label") or "").strip()
            bbox = normalise_bbox(entity.get("bbox"), width, height)
            if not label:
                continue
            if bbox is None:
                skipped_bbox += 1
                continue
            detected.append({"bbox": bbox, "label": label,
                             "confidence": float(entity.get("confidence") or 0.9)})
        if detected:
            objects.append({**key, "objects": detected})

        color_row = map_hsv(row.get("hsv_features") or {})
        if color_row:
            colors.append({**key, **color_row})

    print(f"đọc {len(rows)} dòng Route2:")
    write_pack(args.packs, "caption", "caption_manifest.jsonl", captions, args.model)
    write_pack(args.packs, "ocr", "ocr_manifest.jsonl", ocr, args.model)
    write_pack(args.packs, "object", "object_manifest.jsonl", objects, args.model)
    if colors:
        write_pack(args.packs, "color", "color_manifest.jsonl", colors, "opencv-hsv")
    else:
        print("  color        0 dòng — Route2 chưa kèm hsv_features, "
              "nhánh color_search sẽ rỗng")

    if skipped_no_frame:
        print(f"\nBỎ {skipped_no_frame} keyframe không suy được frame_idx. "
              f"Truyền --map-keyframes nếu Route2 chỉ ghi số thứ tự ảnh.", file=sys.stderr)
    if skipped_bbox:
        print(f"BỎ {skipped_bbox} vùng OCR/object vì bbox không hợp lệ.", file=sys.stderr)

    print(f"\nBước tiếp — `assemble` lo phần contract, đừng tự dựng scenes.jsonl:")
    print(f"    python -m offline assemble --packs {args.packs} --out storage/exports_full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
