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

Chạy — `--sidecar` nhận nhiều file hoặc cả THƯ MỤC, gộp trong MỘT lượt ghi:

    python scripts/apply_ocr_sidecar.py \
        --sidecar C:/Users/ASUS/Downloads/OCR_QWEN_25VL7B \
        --export  storage/exports_competition \
        --model   Qwen2.5-VL-7B-Instruct \
        --missing-out outputs/ocr_con_thieu.jsonl \
        --dry-run

Bỏ `--dry-run` để ghi thật. Keyframe không có trong sidecar được GIỮ NGUYÊN,
nên chạy nhiều đợt (L21-L30 trước, phần còn lại của L25 sau) là an toàn.
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


def expand_sources(items: list[Path]) -> list[Path]:
    """Nhận file lẻ hoặc THƯ MỤC. Lượt OCR chia shard (một file mỗi nhóm video,
    L26 tách năm mảnh) nên gộp hết trong MỘT lượt: mỗi lần chạy là một lần ghi
    lại 1.1GB export, chạy 15 lần là 15 lần như thế."""

    out: list[Path] = []
    for item in items:
        if item.is_dir():
            out.extend(sorted(item.glob("*.jsonl")))
        else:
            out.append(item)
    return out


def load_sidecar(paths: list[Path]) -> tuple[dict[str, tuple], dict[str, tuple], dict[str, int]]:
    """-> (theo keyframe_id, theo image_path, thống kê).

    Giữ dạng thô `(texts, boxes, confs)` chứ chưa dựng `ocr_instances`: mỗi
    instance mang một khối `provenance` riêng, nhân với ~300k chuỗi là vài trăm
    MB nằm không trong RAM suốt lượt ghi. Dựng lúc vá thì mỗi dòng chỉ sống một
    khoảnh khắc.

    Dòng không có chữ vẫn giữ dưới dạng danh sách rỗng: "không có chữ" là kết
    quả hợp lệ và cần được đánh dấu đã xử lý, khác hẳn "chưa chạy tới".

    Trùng key giữa các shard: GIỮ BẢN CÓ NHIỀU CHỮ HƠN. Đo trên bộ Qwen 15
    shard có 344 key trùng, 17 trong đó lệch nội dung và gần như luôn là một
    bên rỗng — tức lượt hỏng/timeout, không phải "frame này thật sự không có
    chữ". Lấy theo thứ tự file thì kết quả phụ thuộc tên file, thứ không ai
    kiểm soát.
    """

    by_id: dict[str, tuple] = {}
    by_path: dict[str, tuple] = {}
    stats = {"dong": 0, "trung": 0, "trung_lech": 0, "khong_khoa": 0}
    for path in paths:
        for row in _iter_jsonl(path):
            stats["dong"] += 1
            texts = [str(t).strip() for t in (row.get("texts") or []) if str(t).strip()]
            payload = (texts, row.get("boxes") or [], row.get("confs") or [])
            if row.get("keyframe_id"):
                bucket, key = by_id, str(row["keyframe_id"])
            elif row.get("image_path"):
                bucket, key = by_path, str(row["image_path"]).replace("\\", "/")
            else:
                stats["khong_khoa"] += 1
                continue
            old = bucket.get(key)
            if old is not None:
                stats["trung"] += 1
                if old[0] != texts:
                    stats["trung_lech"] += 1
                if len(old[0]) >= len(texts):
                    continue
            bucket[key] = payload
    return by_id, by_path, stats


def build_instances(payload: tuple, model: str) -> list[dict[str, Any]]:
    texts, boxes, confs = payload
    return [
        _instance(
            text,
            boxes[i] if i < len(boxes) else None,
            confs[i] if i < len(confs) else None,
            model,
        )
        for i, text in enumerate(texts)
    ]


def _patch(
    row: dict[str, Any],
    by_id: dict[str, tuple],
    by_path: dict[str, tuple],
    model: str,
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
        payload = by_id[key]
    else:
        path = str(row.get("image_path") or "").replace("\\", "/")
        if path not in by_path:
            # Keyframe chưa được OCR: GIỮ NGUYÊN, không ghi rỗng. Ghi rỗng ở
            # đây là nói dối rằng đã xử lý, và lượt bù sau sẽ bỏ qua nó.
            return False
        payload = by_path[path]
    row["ocr_instances"] = build_instances(payload, model)
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
    parser.add_argument("--sidecar", type=Path, nargs="+", required=True,
                        help="File .jsonl hoac THU MUC chua nhieu shard")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_competition"))
    parser.add_argument("--model", default="ocr")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chi bao do phu, khong ghi file nao")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Giu nguyen keyframe da co ocr_instances")
    parser.add_argument("--missing-out", type=Path, default=None,
                        help="Ghi danh sach keyframe CHUA duoc OCR ra file (lam job cho luot sau)")
    parser.add_argument("--merge-out", type=Path, default=None,
                        help="Gop cac shard thanh MOT sidecar roi dung lai, khong dung toi export")
    args = parser.parse_args()

    keyframes_path = args.export / "keyframes.jsonl"
    scenes_path = args.export / "scenes.jsonl"
    sources = expand_sources(args.sidecar)
    by_id, by_path, stats = load_sidecar(sources)

    every = list(by_id.values()) + list(by_path.values())
    with_text = sum(1 for texts, _, _ in every if texts)
    strings = sum(len(texts) for texts, _, _ in every)
    with_box = sum(
        1
        for texts, boxes, _ in every
        for i in range(len(texts))
        if i < len(boxes) and _norm_box(boxes[i]) is not None
    )
    print(f"{len(sources)} shard -> {stats['dong']} dong, {len(every)} keyframe duy nhat")
    print(f"  {with_text} co chu, {strings} chuoi, {with_box} chuoi co bbox that")
    if stats["trung"]:
        print(f"  trung key: {stats['trung']} ({stats['trung_lech']} lech noi dung, "
              f"giu ban nhieu chu hon)")
    if stats["khong_khoa"]:
        print(f"  BO QUA {stats['khong_khoa']} dong khong co keyframe_id lan image_path")

    # Đối chiếu với export TRƯỚC khi ghi: một lượt OCR có thể thiếu cả một nhóm
    # video mà không ai nhận ra, vì file sidecar nào cũng "trông đầy đủ".
    seen = 0
    missing: list[dict[str, Any]] = []
    for row in _iter_jsonl(keyframes_path):
        if row.get("keyframe_id") in by_id or str(
            row.get("image_path") or ""
        ).replace("\\", "/") in by_path:
            seen += 1
        else:
            missing.append(row)
    print(f"export: khop {seen}/{seen + len(missing)} keyframe, thieu {len(missing)}")
    if missing:
        groups: dict[str, int] = {}
        for row in missing:
            prefix = str(row.get("video_id", "?")).split("_")[0]
            groups[prefix] = groups.get(prefix, 0) + 1
        print(f"  thieu theo nhom: {dict(sorted(groups.items()))}")
    if args.missing_out and missing:
        with args.missing_out.open("w", encoding="utf-8") as out:
            for row in missing:
                out.write(json.dumps(
                    {k: row[k] for k in ("keyframe_id", "video_id", "frame_idx",
                                         "image_path", "width", "height") if k in row},
                    ensure_ascii=False,
                ) + "\n")
        print(f"  danh sach con thieu -> {args.missing_out}")

    if args.merge_out:
        # Một sidecar DUY NHẤT, sắp theo khoá. Sắp xếp không phải để đẹp: đợt
        # sau còn bù L25, và thứ tự ổn định thì `diff` giữa hai lần gộp đọc
        # được — đổi tên shard hay đổi thứ tự glob không làm cả file xáo lên.
        #
        # Gộp rồi vẫn dùng lại được chính script này: luật khử trùng "giữ bản
        # nhiều chữ hơn" là idempotent, nên
        #     --sidecar merged.jsonl shard_L25_moi.jsonl --merge-out merged_v2.jsonl
        # cho đúng kết quả như gộp lại từ đầu 15 shard cũ + shard mới.
        args.merge_out.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with args.merge_out.open("w", encoding="utf-8") as out:
            for field, bucket in (("keyframe_id", by_id), ("image_path", by_path)):
                for key in sorted(bucket):
                    texts, boxes, confs = bucket[key]
                    record = {field: key, "texts": texts}
                    # Chỉ ghi khi có thật: Qwen không trả bbox, và một cột rỗng
                    # lặp 148k lần chỉ làm file to ra chứ không mang tin gì.
                    if boxes:
                        record["boxes"] = boxes
                    if confs:
                        record["confs"] = confs
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
        size_mb = args.merge_out.stat().st_size / 1e6
        print(f"gop -> {args.merge_out} ({written} dong, {size_mb:.1f} MB)")
        return

    if args.dry_run:
        print("dry-run: khong ghi gi")
        return

    matched = [0]

    def patch_flat(row: dict[str, Any]) -> None:
        if _patch(row, by_id, by_path, args.model, skip_existing=args.skip_existing):
            matched[0] += 1

    _rewrite(keyframes_path, patch_flat)

    # Runtime CHỈ đọc scenes.jsonl (AIC_METADATA_JSONL) và lấy keyframe LỒNG
    # trong đó. Quên bước này thì OCR nằm trong file mà `bm25_ocr`/`ocr_fuzzy`
    # không thấy một chữ nào — đây là cái bẫy duy nhất của quy trình.
    synced = [0]

    def patch_scene(scene: dict[str, Any]) -> None:
        for nested in scene.get("keyframes", []):
            if _patch(nested, by_id, by_path, args.model, skip_existing=args.skip_existing):
                synced[0] += 1

    _rewrite(scenes_path, patch_scene)
    print(f"xong: {matched[0]} keyframe, {synced[0]} ban long trong scene")
    print("nho chay: python scripts/sync_dataset_manifest.py (cap nhat checksum manifest)")


if __name__ == "__main__":
    main()
