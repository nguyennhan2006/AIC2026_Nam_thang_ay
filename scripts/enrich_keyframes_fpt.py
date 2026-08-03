"""Sinh caption + OCR + object cho keyframe thật bằng FPT VLM (PR-13).

Đọc `storage/packs/keyframe/manifests/keyframe_manifest.jsonl` (đã có
`frame_idx` THẬT), gọi FPT VLM (`AIC_FPT_VLM_MODEL`, vd Qwen2.5-VL-7B-Instruct)
một lần cho mỗi keyframe, ghi thẳng 3 stage pack `caption`/`ocr`/`object` đúng
`contracts/stage_pack.schema.json` — không tự đặt `keyframe_id` (assemble.py
làm việc đó).

Cache theo `sha256(ảnh + prompt + model)` NGAY TỪ ĐẦU: PR-14 sẽ tune prompt
qua nhiều vòng, không cache thì mỗi vòng ablation trả tiền lại từ đầu cho cả
phần không đổi.

Không bịa OCR/object không nhìn thấy: prompt yêu cầu để mảng rỗng nếu không
chắc, và mọi bbox không hợp lệ (x2<=x1 hoặc y2<=y1 sau khi chuẩn hoá — thường
do model ảo giác toạ độ) bị LOẠI THẲNG, không cố "sửa" theo suy đoán.

    python -m scripts.enrich_keyframes_fpt --env-file .env.fpt.local --limit 5   # smoke test
    python -m scripts.enrich_keyframes_fpt --env-file .env.fpt.local            # full 307 keyframe
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from datasection.exporter import atomic_jsonl
from online.adapters.fpt_client import FptClient, FptUsage, image_to_data_url
from online.config import Settings

ROOT = Path(__file__).resolve().parents[1]

_ENRICH_PROMPT = (
    "Phân tích khung hình video này và trả lời DUY NHẤT một JSON object đúng schema sau, "
    "không thêm markdown hay giải thích:\n"
    '{"caption": "...", "ocr_instances": [{"text": "...", "bbox_2d": [x1,y1,x2,y2]}], '
    '"objects": [{"label": "...", "confidence": 0.0, "bbox_2d": [x1,y1,x2,y2]}]}\n\n'
    "Yêu cầu:\n"
    "- caption: mô tả khách quan, cụ thể bằng tiếng Việt (1-2 câu) những gì nhìn thấy trong "
    "khung hình: người, hành động, bối cảnh, vật thể nổi bật.\n"
    "- ocr_instances: CHỈ liệt kê chữ/văn bản THỰC SỰ đọc được rõ trong ảnh (biển hiệu, banner, "
    "phụ đề, chữ trên giấy tờ, tiêu đề...), giữ nguyên chính xác từng ký tự kể cả dấu tiếng Việt "
    "và số. KHÔNG bịa chữ không có trong ảnh. Nếu ảnh không có chữ nào, để mảng rỗng [].\n"
    "- MỖI phần tử của ocr_instances và objects PHẢI là một JSON object đầy đủ 2 khoá "
    '(vd {"text": "60 giây", "bbox_2d": [10,20,200,60]}) — TUYỆT ĐỐI KHÔNG được rút gọn '
    "thành chuỗi trần (vd KHÔNG được viết \"60 giây\" một mình trong mảng).\n"
    "- objects: CHỈ liệt kê vật thể/người THỰC SỰ nhìn thấy rõ trong ảnh, không suy đoán vật thể "
    "ngoài khung hình hoặc bị che khuất. Nếu không chắc, bỏ qua thay vì đoán.\n"
    "- bbox_2d là toạ độ PIXEL thực trên ảnh gốc theo thứ tự [x1, y1, x2, y2], x1<x2 và y1<y2, "
    "BẮT BUỘC có mặt ở mọi phần tử của ocr_instances lẫn objects."
)


def _load_env_file(path: Path) -> None:
    """Nạp KEY=VALUE từ file vào os.environ — biến đã có sẵn trong shell/env
    THẬT giữ nguyên (ưu tiên như `_env()` của scripts/caption_qwen3vl.py)."""

    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _normalize_bbox(bbox_2d: Any, width: int, height: int) -> dict[str, float] | None:
    """Pixel xyxy -> normalized [0,1]. Trả None nếu box suy biến (ảo giác)."""

    if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox_2d)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    nx1, nx2 = sorted((max(0.0, min(1.0, x1 / width)), max(0.0, min(1.0, x2 / width))))
    ny1, ny2 = sorted((max(0.0, min(1.0, y1 / height)), max(0.0, min(1.0, y2 / height))))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return {"x1": nx1, "y1": ny1, "x2": nx2, "y2": ny2}


def _cache_key(image_path: Path, prompt: str, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    digest.update(prompt.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    return digest.hexdigest()


def _write_pack(pack_dir: Path, manifest_name: str, rows: list[dict], *, model: str) -> None:
    (pack_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (pack_dir / "_SUCCESS.json").write_text(
        json.dumps({"status": "success", "stage": pack_dir.name, "count": len(rows)}, ensure_ascii=False),
        encoding="utf-8",
    )
    (pack_dir / "model_info.json").write_text(
        json.dumps({"component": pack_dir.name, "model": model, "pack_version": "fpt-enrich-v1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    atomic_jsonl(pack_dir / "manifests" / manifest_name, rows)


def _process_one(
    client: FptClient, model: str, cache_dir: Path, row: dict, data_root: Path
) -> tuple[dict, dict[str, Any] | None, bool, FptUsage | None]:
    """Trả `(row_gốc, parsed_json hoặc None nếu lỗi, cache_hit, usage)`.

    Bọc TOÀN BỘ trong try/except: một frame lỗi (ảnh hỏng, JSON không sửa
    được, provider lỗi) không được phép làm hỏng cả batch 307 frame.
    """

    image_path = data_root / row["image_path"]
    try:
        cache_key = _cache_key(image_path, _ENRICH_PROMPT, model)
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return row, json.loads(cache_path.read_text(encoding="utf-8")), True, None

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": _ENRICH_PROMPT},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }]
        # 800 tokens từng không đủ cho vài frame có nhiều object/OCR — JSON bị
        # cắt giữa chừng (thấy thật ở 3/307 frame khi chạy full lần đầu, luôn
        # là do mảng objects/ocr_instances dài). 1400 đủ dư cho trường hợp đó.
        result = client.chat_completion(messages, model=model, temperature=0.0, max_tokens=1400)
        parsed = _extract_json(result.text)
        if parsed is None:
            return row, {"_error": f"malformed JSON: {result.text[:300]!r}"}, False, result.usage
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        return row, parsed, False, result.usage
    except Exception as exc:  # noqa: BLE001 - một frame lỗi không được sập cả batch
        return row, {"_error": f"{type(exc).__name__}: {exc}"}, False, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--keyframes", type=Path, default=ROOT / "storage" / "packs" / "keyframe")
    parser.add_argument("--out", type=Path, default=ROOT / "storage" / "packs")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "storage" / "cache" / "fpt_enrich")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N keyframe đầu (smoke test)")
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    if args.env_file:
        _load_env_file(args.env_file)

    settings = Settings.from_env()
    if not settings.fpt_enabled:
        raise SystemExit("AIC_FPT_ENABLED=false — bật trong .env.fpt.local trước khi chạy enrichment")
    vlm_model = settings.fpt_vlm_model
    if not vlm_model:
        raise SystemExit("AIC_FPT_VLM_MODEL trống — cần model VLM (vd Qwen2.5-VL-7B-Instruct)")

    data_root = args.data_root or settings.data_root
    concurrency = args.concurrency or settings.fpt_max_concurrency

    manifest_path = args.keyframes / "manifests" / "keyframe_manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"keyframe manifest rỗng: {manifest_path}")

    client = FptClient.from_settings(settings)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Enrich {len(rows)} keyframe qua FPT VLM ({vlm_model}, concurrency={concurrency})...")

    caption_rows: list[dict] = []
    ocr_rows: list[dict] = []
    object_rows: list[dict] = []
    failures: list[dict] = []
    cache_hits = 0
    total_input_tokens = 0
    total_output_tokens = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_process_one, client, vlm_model, args.cache_dir, row, data_root): row
            for row in rows
        }
        done = 0
        for future in as_completed(futures):
            source_row, parsed, cache_hit, usage = future.result()
            done += 1
            video_id = str(source_row["video_id"])
            frame_idx = int(source_row["frame_idx"])
            if cache_hit:
                cache_hits += 1
            if usage is not None:
                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens

            if parsed is None or "_error" in parsed:
                reason = (parsed or {}).get("_error", "unknown")
                failures.append({"video_id": video_id, "frame_idx": frame_idx, "reason": reason})
                print(f"  [{done}/{len(rows)}] LỖI frame_idx={frame_idx}: {reason}")
                continue

            caption_text = parsed.get("caption")
            if isinstance(caption_text, str) and caption_text.strip():
                caption_rows.append({
                    "video_id": video_id, "frame_idx": frame_idx,
                    "captions": [{"text": caption_text.strip(), "language": "vi", "caption_type": "detailed"}],
                })

            width = int(source_row.get("width") or 1280)
            height = int(source_row.get("height") or 720)
            # Model 7B đôi khi bỏ qua schema và trả phần tử là chuỗi trần thay
            # vì object {"text":..., "bbox_2d":...} — ocrRow/objectRow schema
            # BẮT BUỘC có bbox nên các trường hợp đó bị loại (không suy đoán
            # bbox), chỉ giữ item đúng dạng object với bbox hợp lệ.
            ocr_instances = []
            for item in parsed.get("ocr_instances", []) or []:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                bbox = _normalize_bbox(item.get("bbox_2d"), width, height)
                if isinstance(text, str) and text.strip() and bbox is not None:
                    ocr_instances.append({"text": text.strip(), "bbox": bbox, "language": "vi"})
            if ocr_instances:
                ocr_rows.append({"video_id": video_id, "frame_idx": frame_idx, "instances": ocr_instances})

            objects = []
            for item in parsed.get("objects", []) or []:
                if not isinstance(item, dict):
                    continue
                label = item.get("label")
                bbox = _normalize_bbox(item.get("bbox_2d"), width, height)
                if isinstance(label, str) and label.strip() and bbox is not None:
                    entry: dict[str, Any] = {"label": label.strip(), "bbox": bbox}
                    if isinstance(item.get("confidence"), (int, float)):
                        entry["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
                    objects.append(entry)
            if objects:
                object_rows.append({"video_id": video_id, "frame_idx": frame_idx, "objects": objects})

            print(f"  [{done}/{len(rows)}] frame_idx={frame_idx} OK (cache={'hit' if cache_hit else 'miss'})")

    _write_pack(args.out / "caption", "caption_manifest.jsonl", caption_rows, model=vlm_model)
    _write_pack(args.out / "ocr", "ocr_manifest.jsonl", ocr_rows, model=vlm_model)
    _write_pack(args.out / "object", "object_manifest.jsonl", object_rows, model=vlm_model)

    print(f"\nXong: {len(rows)} keyframe, cache_hit={cache_hits}, lỗi={len(failures)}")
    print(f"caption: {len(caption_rows)} dòng, ocr: {len(ocr_rows)} dòng, object: {len(object_rows)} dòng")
    print(f"token: input={total_input_tokens}, output={total_output_tokens} (chỉ tính call thật, không tính cache hit)")
    if failures:
        failed_path = args.out / "fpt_enrich_failures.jsonl"
        atomic_jsonl(failed_path, failures)
        print(f"CẢNH BÁO: {len(failures)} frame lỗi -> {failed_path}")


if __name__ == "__main__":
    main()
