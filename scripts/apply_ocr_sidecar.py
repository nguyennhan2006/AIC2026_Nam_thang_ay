"""Ghép một file OCR SIDECAR vào export canonical, theo kiểu STREAM.

Vì sao là sidecar chứ không bắt lượt OCR ghi thẳng `keyframes.jsonl`: export
thi đấu nặng 515MB keyframes + 650MB scenes, mỗi dòng còn mang caption/object/
embedding_refs. Bắt máy chạy OCR (Kaggle/VastAI) mang cả khối đó đi rồi mang
về là chép 1.1GB qua mạng cho vài chục MB chữ. Sidecar chỉ mang
`keyframe_id -> chữ`, tải về là ghép được ngay.

Định dạng sidecar — JSONL, MỖI KEYFRAME MỘT DÒNG, kể cả khi không có chữ:

    {"keyframe_id": "L21_V001_S0000_F000000",
     "texts": ["HTV9 HD", "60 giay"],
     "boxes": [[0.78, 0.04, 0.95, 0.12], [0.40, 0.42, 0.60, 0.55]],
     "confs": [0.99, 0.87]}

`keyframe_id` là khoá chính; thiếu thì dùng `image_path` (đúng nguyên văn chuỗi
trong `keyframes.jsonl`). `boxes`/`confs` tuỳ chọn, nhưng nếu có thì phải cùng
độ dài và cùng thứ tự với `texts`.

**Nên gửi kèm `boxes` thật.** Bộ lọc lớp phủ ở `online/adapters/bm25.py` loại
logo/đồng hồ/chữ chạy bằng VỊ TRÍ — đo trên dữ liệu thật thì 84% chuỗi OCR là
lớp phủ và chúng nằm đúng hai dải cố định trên khung hình. Không có bbox thì
chỉ còn lọc theo tần suất, yếu hơn hẳn. Toạ độ CHUẨN HOÁ 0..1 (chia cho
width/height của chính keyframe đó), x1<x2, y1<y2. Khung đúng bằng [0,0,1,1]
được hiểu là "không biết vị trí" (quy ước của `scripts/ocr_backfill.py`), nên
đừng dùng nó cho chữ có toạ độ thật.

Chạy:
    python scripts/apply_ocr_sidecar.py \
        --sidecar storage/incoming/ocr_competition.jsonl \
        --export  storage/exports_competition \
        --model   paddleocr:PP-OCRv4-vi
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterator

MARKER = "ocr_sidecar_v1"
_FULL_FRAME = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Đọc từng dòng. Không `read_text()` cả file: scenes.jsonl 650MB nạp một
    lượt thành object Python là vài GB RAM, máy local không chịu nổi."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _norm_box(raw: Any) -> dict[str, float] | None:
    """Chấp nhận [x1,y1,x2,y2] hoặc {"x1": ...}. Sai định dạng -> None."""

    if isinstance(raw, dict):
        try:
            vals = [float(raw["x1"]), float(raw["y1"]), float(raw["x2"]), float(raw["y2"])]
        except (KeyError, TypeError, ValueError):
            return None
    elif isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            vals = [float(v) for v in raw]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not all(-0.001 <= v <= 1.001 for v in vals):
        # Toạ độ pixel lọt vào đây thì bộ lọc dải lớp phủ hiểu sai hoàn toàn —
        # thà bỏ bbox còn hơn tin một con số sai đơn vị.
        return None
    x1, y1, x2, y2 = vals
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return {"x1": round(x1, 6), "y1": round(y1, 6), "x2": round(x2, 6), "y2": round(y2, 6)}


def _instance(text: str, box: Any, conf: Any, model: str) -> dict[str, Any]:
    parsed = _norm_box(box)
    return {
        "text": text,
        "normalized_text": None,
        "language": "vi",
        "confidence": float(conf) if isinstance(conf, (int, float)) else 0.0,
        "bbox": parsed or dict(_FULL_FRAME),
        "provenance": {
            "created_at": _now(),
            "device": "unknown",
            "model_name": model if parsed else f"{model}:ocr-textonly",
            "model_revision": MARKER,
            "parameters": {},
            "pipeline_version": "aic-v1.0.0",
            "prompt_version": None,
        },
    }


def load_sidecar(path: Path, model: str) -> tuple[dict[str, list], dict[str, list]]:
    """-> (theo keyframe_id, theo image_path).

    Dòng không có chữ vẫn giữ dưới dạng danh sách rỗng: "không có chữ" là kết
    quả hợp lệ và cần được đánh dấu đã xử lý, khác hẳn "chưa chạy tới".
    """

    by_id: dict[str, list] = {}
    by_path: dict[str, list] = {}
    for row in _iter_jsonl(path):
        texts = [str(t).strip() for t in (row.get("texts") or []) if str(t).strip()]
        boxes = row.get("boxes") or []
        confs = row.get("confs") or []
        instances = [
            _instance(
                text,
                boxes[i] if i < len(boxes) else None,
                confs[i] if i < len(confs) else None,
                model,
            )
            for i, text in enumerate(texts)
        ]
        if row.get("keyframe_id"):
            by_id[str(row["keyframe_id"])] = instances
        elif row.get("image_path"):
            by_path[str(row["image_path"]).replace("\\", "/")] = instances
    return by_id, by_path


def _patch(
    row: dict[str, Any],
    by_id: dict[str, list],
    by_path: dict[str, list],
    *,
    skip_existing: bool = False,
) -> bool:
    # Mặc định GHI ĐÈ, kể cả bằng danh sách rỗng — trên export thi đấu
    # `ocr_instances` đang rỗng 100% nên không mất gì, còn khi chạy lại thì
    # lượt mới phải thắng lượt cũ. `--skip-existing` để giữ OCR có bbox thật
    # của một lượt enrich trước.
    if skip_existing and row.get("ocr_instances"):
        return False
    key = row.get("keyframe_id")
    if key in by_id:
        instances = by_id[key]
    else:
        path = str(row.get("image_path") or "").replace("\\", "/")
        if path not in by_path:
            return False
        instances = by_path[path]
    row["ocr_instances"] = instances
    row.setdefault("extensions", {})["ocr_backfill"] = MARKER
    return True


def _rewrite(path: Path, patch_fn) -> None:
    """Ghi ra file tạm rồi mới thay: đứt giữa chừng thì export gốc còn nguyên,
    chứ không để lại một keyframes.jsonl cụt 200MB."""

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for row in _iter_jsonl(path):
            patch_fn(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ghep sidecar OCR vao export canonical")
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--export", type=Path, default=Path("storage/exports_competition"))
    parser.add_argument("--model", default="ocr")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chi bao do phu, khong ghi file nao")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Giu nguyen keyframe da co ocr_instances")
    args = parser.parse_args()

    keyframes_path = args.export / "keyframes.jsonl"
    scenes_path = args.export / "scenes.jsonl"
    by_id, by_path = load_sidecar(args.sidecar, args.model)

    every = list(by_id.values()) + list(by_path.values())
    with_text = sum(1 for v in every if v)
    strings = sum(len(v) for v in every)
    with_box = sum(1 for v in every for item in v if item["bbox"] != _FULL_FRAME)
    print(f"sidecar: {len(every)} keyframe, {with_text} co chu, "
          f"{strings} chuoi, {with_box} chuoi co bbox that")

    if args.dry_run:
        seen = sum(
            1
            for row in _iter_jsonl(keyframes_path)
            if row.get("keyframe_id") in by_id
            or str(row.get("image_path") or "").replace("\\", "/") in by_path
        )
        print(f"dry-run: khop {seen} keyframe trong {keyframes_path.name}")
        return

    matched = [0]

    def patch_flat(row: dict[str, Any]) -> None:
        if _patch(row, by_id, by_path, skip_existing=args.skip_existing):
            matched[0] += 1

    _rewrite(keyframes_path, patch_flat)

    # Runtime CHỈ đọc scenes.jsonl (AIC_METADATA_JSONL) và lấy keyframe LỒNG
    # trong đó. Quên bước này thì OCR nằm trong file mà `bm25_ocr`/`ocr_fuzzy`
    # không thấy một chữ nào — đây là cái bẫy duy nhất của quy trình.
    synced = [0]

    def patch_scene(scene: dict[str, Any]) -> None:
        for nested in scene.get("keyframes", []):
            if _patch(nested, by_id, by_path, skip_existing=args.skip_existing):
                synced[0] += 1

    _rewrite(scenes_path, patch_scene)
    print(f"xong: {matched[0]} keyframe, {synced[0]} ban long trong scene")
    print("nho chay: python scripts/sync_dataset_manifest.py (cap nhat checksum manifest)")


if __name__ == "__main__":
    main()
