"""Bù `color` và `quality` cho keyframe — CPU thuần, không gọi API.

Hai trường này có đường đọc đầy đủ trong online nhưng dữ liệu trống 100%:

    color               0/855   -> nhánh `color_search` rỗng ở MỌI truy vấn
    quality.*           0/855   -> `safe_frame` mất sharpness/brightness/black-frame

Cả hai chỉ cần đọc ảnh keyframe đã có sẵn trên đĩa. Không cần GPU, không cần
model, không tốn một đồng API — đây là phần rẻ nhất của toàn bộ tầng dữ liệu và
nó bị bỏ trống chỉ vì chưa ai chạy.

Cách đặt tên màu tái dùng NGUYÊN VẸN `TransformersGpuEngine._name_pixels`
(offline/gpu_engine.py), không viết lại: `color_search` khớp CHÍNH XÁC theo
chuỗi giữa `COLOR_LEXICON` và `dominant_colors[].name`, nên một bộ tên thứ hai
lệch dù một chữ là nhánh đó lại rỗng, lần này còn khó phát hiện hơn.

`selection_score` CỐ Ý để nguyên `None`. Khác với `quality` — vốn là phép đo vật
lý khách quan trên ảnh — `selection_score` là một phán đoán "frame này đại diện
tốt đến đâu". `safe_frame._quality` cộng thẳng nó vào điểm chọn frame, nên bịa ra
một công thức ở đây là nhét một tín hiệu xếp hạng chưa từng được đo vào đường
chấm điểm. Muốn có thì phải thiết kế rồi đo riêng.

LƯU Ý: `quality` cũng đi vào `safe_frame`, nên lượt bù này CÓ thể đổi frame mà
KIS chọn. Đó là lý do phải đo lại 4 task trước/sau, không phải chạy xong là xong.

    python -m scripts.backfill_color_quality --dry-run
    python -m scripts.backfill_color_quality
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

MARKER = "color-quality-v1"

# Ngưỡng khớp `online/services/safe_frame.py`: sharpness được chuẩn hoá trong
# khoảng [40, 300]. In ra phân bố thật để biết thang đo có hợp lý với corpus này
# hay không, thay vì tin vào hằng số đặt sẵn.
SHARPNESS_FLOOR = 40.0
SHARPNESS_CEILING = 300.0

# Pixel tối hơn mức này tính là "đen". 8-bit, nên 16/255 ~ 6% độ sáng.
BLACK_LEVEL = 16


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _count_names(hue, sat, val) -> dict[str, int]:
    """Đếm pixel theo tên màu — cùng LUẬT với `TransformersGpuEngine._name_pixels`.

    Vì sao không gọi thẳng hàm gốc: nó dựng `numpy.empty(shape, dtype=object)` rồi
    gán chuỗi Python cho từng pixel. Trên ảnh 1280x720 (~1M pixel) và gọi 4 lần
    mỗi ảnh (toàn khung + 3 dải) thì mất **1,9 giây/ảnh** — 27 phút cho 855
    keyframe, và ~139 GIỜ nếu mở rộng lên 876 video. Bản này đếm bằng mặt nạ
    boolean nên không bao giờ tạo mảng object.

    Kết quả PHẢI trùng tuyệt đối: `color_search` khớp chính xác theo chuỗi giữa
    `COLOR_LEXICON` và `dominant_colors[].name`. `_assert_matches_reference()`
    kiểm điều đó trên ảnh thật trước khi chạy, không tin vào việc đọc code.
    """

    import numpy

    from offline.gpu_engine import TransformersGpuEngine

    counts: dict[str, int] = {}
    low_sat = sat < 0.15
    for name, mask in (
        ("black", low_sat & (val < 0.2)),
        ("white", low_sat & (val >= 0.85)),
        ("gray", low_sat & (val >= 0.2) & (val < 0.85)),
    ):
        total = int(mask.sum())
        if total:
            counts[name] = counts.get(name, 0) + total
    colored = ~low_sat
    for name, low, high in TransformersGpuEngine._HUE_NAMES:
        total = int((colored & (hue >= low) & (hue < high)).sum())
        if total:
            counts[name] = counts.get(name, 0) + total
    return counts


def _assert_matches_reference(path: Path) -> None:
    """Chứng minh bản nhanh cho ra ĐÚNG kết quả của hàm gốc, trên một ảnh thật."""

    import numpy
    from PIL import Image

    from offline.gpu_engine import TransformersGpuEngine

    with Image.open(path) as handle:
        hsv = numpy.asarray(handle.convert("HSV"), dtype=numpy.float32)
    hue = hsv[..., 0].ravel() / 255.0 * 360.0
    sat = hsv[..., 1].ravel() / 255.0
    val = hsv[..., 2].ravel() / 255.0

    names = TransformersGpuEngine._name_pixels(hue, sat, val)
    unique, tallies = numpy.unique(names, return_counts=True)
    reference = {str(name): int(count) for name, count in zip(unique, tallies)}
    fast = _count_names(hue, sat, val)
    if reference != fast:
        raise SystemExit(
            "ban nhanh LECH ban goc — dung lai thay vi ghi du lieu sai:\n"
            f"  goc  : {sorted(reference.items())}\n"
            f"  nhanh: {sorted(fast.items())}"
        )


def _provenance(task: str) -> dict[str, Any]:
    return {
        "created_at": _now(),
        "device": "cpu",
        "model_name": f"pillow-numpy:{task}",
        "model_revision": MARKER,
        "parameters": {},
        "pipeline_version": "aic-v1.0.0",
    }


def measure(path: Path, hist_bins: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(color, quality)` cho một ảnh. Đọc ảnh đúng MỘT lần cho cả hai."""

    import numpy
    from PIL import Image

    with Image.open(path) as handle:
        image = handle.convert("RGB")
        hsv = numpy.asarray(image.convert("HSV"), dtype=numpy.float32)
        gray = numpy.asarray(image.convert("L"), dtype=numpy.float32)

    hue = hsv[..., 0].ravel() / 255.0 * 360.0
    sat = hsv[..., 1].ravel() / 255.0
    val = hsv[..., 2].ravel() / 255.0

    raw_hist, _ = numpy.histogram(hue, bins=hist_bins, range=(0.0, 360.0))
    total = float(raw_hist.sum())
    histogram = (raw_hist / total).tolist() if total > 0 else [0.0] * hist_bins

    counts = _count_names(hue, sat, val)
    pixels = float(hue.size)
    dominant = sorted(
        (
            {"name": name, "ratio": round(count / pixels, 4)}
            for name, count in counts.items()
        ),
        key=lambda item: -item["ratio"],
    )[:8]

    height = hsv.shape[0]
    band = max(1, height // 3)
    regions: dict[str, list[str]] = {}
    for region, rows in (
        ("upper", slice(0, band)),
        ("center", slice(band, 2 * band)),
        ("lower", slice(2 * band, height)),
    ):
        band_counts = _count_names(
            hsv[rows, :, 0].ravel() / 255.0 * 360.0,
            hsv[rows, :, 1].ravel() / 255.0,
            hsv[rows, :, 2].ravel() / 255.0,
        )
        if band_counts:
            regions[region] = [max(band_counts.items(), key=lambda item: item[1])[0]]

    color = {
        "dominant_hex": [],
        "dominant_colors": dominant,
        "mean_hsv": [
            round(float(hue.mean()), 2),
            round(float(sat.mean()), 4),
            round(float(val.mean()), 4),
        ],
        "hsv_histogram": [round(value, 6) for value in histogram],
        "regions": regions,
        "provenance": _provenance("color"),
    }

    # Phương sai Laplacian — thước đo độ nét tiêu chuẩn. Dùng nhân 4 lân cận
    # thay vì gọi scipy/cv2 để không thêm phụ thuộc cho một phép tính 5 dòng.
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    quality = {
        "sharpness": round(float(laplacian.var()), 3),
        "brightness": round(float(gray.mean() / 255.0), 4),
        "contrast": round(float(gray.std() / 255.0), 4),
        "black_frame_ratio": round(float((gray < BLACK_LEVEL).mean()), 4),
        # `duplicate_score` cần so sánh GIỮA các frame, không đo được từ một ảnh
        # đơn lẻ. Để None thay vì điền 0.0 — 0.0 nghĩa là "chắc chắn không trùng",
        # một khẳng định ta chưa kiểm tra.
        "duplicate_score": None,
    }
    return color, quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Bù color + quality cho keyframe (CPU)")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--hist-bins", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--redo", action="store_true", help="Lam lai ca keyframe da bu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenes_path = args.export / "scenes.jsonl"
    keyframes_path = args.export / "keyframes.jsonl"
    scenes = [
        json.loads(line)
        for line in scenes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    todo: list[dict[str, Any]] = []
    missing_image = 0
    for scene in scenes:
        for frame in scene.get("keyframes") or []:
            done = (frame.get("extensions") or {}).get("color_quality") == MARKER
            if done and not args.redo:
                continue
            if not (args.data_root / frame["image_path"]).is_file():
                missing_image += 1
                continue
            todo.append(frame)
    if args.limit:
        todo = todo[: args.limit]

    print(f"can bu: {len(todo)} keyframe" + (f"  (thieu anh: {missing_image})" if missing_image else ""), flush=True)
    if not todo:
        return
    _assert_matches_reference(args.data_root / todo[0]["image_path"])
    print("kiem chung: ban nhanh khop ban goc _name_pixels", flush=True)
    if args.dry_run:
        print("(--dry-run: khong ghi gi)")
        return

    sharpness_values: list[float] = []
    for index, frame in enumerate(todo, start=1):
        color, quality = measure(args.data_root / frame["image_path"], args.hist_bins)
        frame["color"] = color
        frame["quality"] = quality
        frame.setdefault("extensions", {})["color_quality"] = MARKER
        sharpness_values.append(quality["sharpness"])
        if index % 200 == 0:
            print(f"  {index}/{len(todo)}", flush=True)

    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes),
        encoding="utf-8",
    )

    patched = {
        (scene["video_id"], int(frame["frame_idx"])): frame
        for scene in scenes
        for frame in scene.get("keyframes") or []
        if (frame.get("extensions") or {}).get("color_quality") == MARKER
    }
    rows = [
        json.loads(line)
        for line in keyframes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    synced = 0
    for row in rows:
        source = patched.get((row["video_id"], int(row["frame_idx"])))
        if source is not None:
            row["color"] = source["color"]
            row["quality"] = source["quality"]
            row.setdefault("extensions", {})["color_quality"] = MARKER
            synced += 1
    keyframes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    sharpness_values.sort()
    def percentile(fraction: float) -> float:
        return sharpness_values[min(len(sharpness_values) - 1, int(len(sharpness_values) * fraction))]

    print(f"xong: {len(todo)} keyframe, {synced} dong bo sang keyframes.jsonl")
    # Thang [40, 300] của safe_frame được chọn TRƯỚC khi có dữ liệu thật. In phân
    # bố ra để thấy ngay nếu corpus này nằm lệch hẳn khỏi thang đó — lúc ấy mọi
    # frame sẽ bị kẹp về 0 hoặc 1 và tín hiệu sharpness thành vô dụng.
    print(
        "sharpness  p10={:.0f}  p50={:.0f}  p90={:.0f}   (safe_frame chuan hoa trong [{:.0f}, {:.0f}])".format(
            percentile(0.10), percentile(0.50), percentile(0.90),
            SHARPNESS_FLOOR, SHARPNESS_CEILING,
        )
    )
    below = sum(1 for value in sharpness_values if value < SHARPNESS_FLOOR)
    above = sum(1 for value in sharpness_values if value > SHARPNESS_CEILING)
    print(f"           duoi san: {below}   tren tran: {above}   (bi kep, mat phan biet)")


if __name__ == "__main__":
    main()
