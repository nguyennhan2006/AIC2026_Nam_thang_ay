"""Cân bằng dữ liệu distractor: sinh object + OCR + action cho keyframe export.

PR-4C. Vấn đề đang giải: L21_V002/V003 CHỈ có caption, nên trong 10 nhánh
retrieval chỉ 2 nhánh có cạnh tranh thật (`dense_visual`, `bm25_caption`). Sáu
nhánh còn lại — `bm25_keyword`, `bm25_ocr`, `ocr_fuzzy`, `bm25_object`,
`bm25_action`, `event_search` — về mặt cấu trúc không thể trả về gì ngoài
L21_V001. Mọi phép đo "hệ khoẻ khi có distractor" vì thế dễ hơn thực tế.

`bm25_asr` KHÔNG cân bằng được: không có audio cho V002/V003. Nó sẽ vĩnh viễn
là nhánh chỉ-V001, nên phép đo đa video phải tắt nó (`--disable-branch
bm25_asr`) và ghi rõ đó là giới hạn của bộ dữ liệu.

Một lệnh gọi VLM cho MỖI keyframe trả về cả bốn thứ, thay vì bốn lượt riêng —
ảnh phải mã hoá base64 và gửi lại mỗi lần, nên gộp là khác biệt lớn về chi phí
lẫn thời gian dưới trần 50 RPM của FPT.

`keywords` KHÔNG gọi model: chúng được suy từ nhãn object, đúng cách
`offline/assemble.py` làm (`sources: ["object"]`).

Bbox sai (x2<=x1 hoặc y2<=y1) bị LOẠI THẲNG chứ không "sửa" theo suy đoán —
model ảo giác toạ độ là chuyện thường và một bbox bịa còn tệ hơn không có.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import time

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.adapters.rerank import _image_data_url
from online.config import Settings

PROMPT = """Phân tích khung hình này. Trả về DUY NHẤT một object JSON:

{"caption": "<một câu tiếng Việt tả những gì NHÌN THẤY>",
 "objects": [{"label": "<tên vật thể tiếng Việt>", "bbox": [x1, y1, x2, y2]}],
 "ocr": [{"text": "<chữ NGUYÊN VĂN thấy trên màn hình>", "bbox": [x1, y1, x2, y2]}],
 "actions": ["<hành động tiếng Anh, vd standing, walking, driving>"]}

Quy tắc:
- Toạ độ bbox là tỉ lệ 0..1 so với chiều rộng/cao ảnh.
- objects: tối đa 8 vật thể NỔI BẬT. Không liệt kê chi tiết vụn vặt.
- ocr: chép NGUYÊN VĂN, không dịch, không diễn giải. Không thấy chữ thì để mảng rỗng.
- CẤM bịa: không chắc thì để mảng rỗng. Mảng rỗng luôn tốt hơn phỏng đoán.
- caption: bỏ qua logo kênh và đồng hồ trên màn hình."""


ENRICH_MARKER = "export_enrich_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _provenance(model: str, suffix: str) -> dict:
    return {
        "created_at": _now(),
        "device": "unknown",
        "model_name": f"{model}:{suffix}",
        "model_revision": "export-enrich-v1",
        "parameters": {},
        "pipeline_version": "aic-v1.0.0",
        "prompt_version": "export_enrich_v1",
    }


def _bbox(raw, width: int, height: int) -> dict | None:
    """Bbox chuẩn hoá về [0,1], hoặc None nếu không hợp lệ.

    Model trả toạ độ theo PIXEL chứ không theo tỉ lệ, dù prompt yêu cầu 0..1 —
    quan sát thật: `[0, 267, 1287, 493]` trên ảnh 1288x756. Kẹp thẳng về [0,1]
    biến mọi bbox thành `[0,1,1,1]` và bị loại hết vì `y2 <= y1`. Phát hiện
    theo giá trị: có số nào > 1 thì coi là pixel và chia cho kích thước ảnh.
    """

    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v > 1.0 for v in values) and width > 0 and height > 0:
        values = [
            values[0] / width, values[1] / height,
            values[2] / width, values[3] / height,
        ]
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in values)
    if x2 <= x1 or y2 <= y1:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def parse_payload(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def build_fields(payload: dict, model: str, width: int, height: int) -> dict:
    objects = []
    for item in (payload.get("objects") or [])[:8]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        bbox = _bbox(item.get("bbox"), width, height)
        if not label or bbox is None:
            continue
        objects.append({
            "attributes": {},
            "bbox": bbox,
            "confidence": 0.8,
            "label": label,
            "provenance": _provenance(model, "object"),
        })

    ocr_instances = []
    for item in (payload.get("ocr") or [])[:12]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        bbox = _bbox(item.get("bbox"), width, height)
        if not text or bbox is None:
            continue
        ocr_instances.append({
            "bbox": bbox,
            "confidence": 0.0,
            "language": "vi",
            "normalized_text": None,
            "text": text,
            "provenance": _provenance(model, "ocr"),
        })

    actions = [
        str(item).strip().lower()
        for item in (payload.get("actions") or [])[:6]
        if isinstance(item, str) and str(item).strip()
    ]
    return {
        "caption": str(payload.get("caption") or "").strip(),
        "objects": objects,
        "ocr_instances": ocr_instances,
        "actions": sorted(set(actions)),
    }


def _merge_objects(old: list[dict], new: list[dict]) -> list[dict]:
    """Gộp theo NHÃN — cùng nhãn thì giữ bản cũ (đã kiểm chứng)."""

    seen = {item.get("label") for item in old}
    return old + [item for item in new if item.get("label") not in seen]


def _merge_ocr(old: list[dict], new: list[dict]) -> list[dict]:
    """Gộp theo chuỗi chữ đã chuẩn hoá khoảng trắng/hoa-thường."""

    def key(item: dict) -> str:
        return " ".join(str(item.get("text", "")).split()).casefold()

    seen = {key(item) for item in old}
    return old + [item for item in new if key(item) not in seen]


def keyword_block(text: str) -> dict:
    """Keyword suy từ nhãn object — KHÔNG gọi model.

    Giữ nguyên hình dạng mà `offline/assemble.py` sinh ra (`sources: ["object"]`)
    để `bm25_keyword` index được y như với L21_V001.
    """

    return {
        "confidence": None,
        "language": None,
        "normalized_text": text,
        "sources": ["object"],
        "text": text,
        "provenance": {
            "created_at": _now(),
            "device": None,
            "model_name": "assemble:keyword",
            "model_revision": None,
            "parameters": {},
            "pipeline_version": "aic-v1.0.0",
            "prompt_version": None,
        },
    }


async def main_async(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not (settings.fpt_enabled and settings.fpt_vlm_model):
        raise SystemExit("cần AIC_FPT_ENABLED=true và AIC_FPT_VLM_MODEL")
    client = FptClient.from_settings(settings)
    model = settings.fpt_vlm_model
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    keyframes_path = args.export / "keyframes.jsonl"
    scenes_path = args.export / "scenes.jsonl"
    keyframes = [json.loads(l) for l in keyframes_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    wanted = set(args.video)
    todo = [
        row for row in keyframes
        if (not wanted or row["video_id"] in wanted)
        # Đánh dấu tường minh thay vì suy từ nội dung: model hoàn toàn có thể
        # trả về mảng rỗng hợp lệ (ảnh không có chữ, không có hành động), và
        # dùng "rỗng" làm dấu hiệu chưa xử lý sẽ lặp lại chúng mãi mãi.
        and row.get("extensions", {}).get("enriched") != ENRICH_MARKER
        and (args.data_root / row["image_path"]).exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("không có keyframe nào cần enrich")
        return
    print(f"cần enrich {len(todo)} keyframe (model={model}, {args.rpm} RPM)")

    semaphore = asyncio.Semaphore(args.concurrency)
    min_interval = 60.0 / max(args.rpm, 1)
    rate_lock = threading.Lock()
    next_slot = [0.0]
    done = failed = 0

    def wait_for_slot() -> None:
        with rate_lock:
            start = max(time.monotonic(), next_slot[0])
            next_slot[0] = start + min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def call(image_path: Path) -> dict | None:
        digest = hashlib.sha256()
        digest.update(image_path.read_bytes())
        digest.update(PROMPT.encode("utf-8"))
        digest.update(model.encode("utf-8"))
        cached = args.cache_dir / f"{digest.hexdigest()}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        wait_for_slot()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
            ],
        }]
        text = client.chat_completion(
            messages, model=model, temperature=0.0, max_tokens=700,
            response_format={"type": "json_object"},
        ).text
        payload = parse_payload(text)
        if payload is None:
            return None
        cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    async def one(row: dict) -> None:
        nonlocal done, failed
        async with semaphore:
            try:
                payload = await asyncio.to_thread(call, args.data_root / row["image_path"])
            except (ProviderError, OSError) as exc:
                failed += 1
                print(f"  bỏ qua {row['keyframe_id']}: {str(exc)[:110]}", flush=True)
                return
            if payload is None:
                failed += 1
                return
            fields = build_fields(
                payload, model, int(row.get("width") or 0), int(row.get("height") or 0)
            )
            # HỢP NHẤT, không ghi đè. Lý do: L21_V001 đã được enrich bằng prompt
            # tinh chỉnh riêng ở CAPTION-ENRICH-01, và một số gold query là dạng
            # OCR ("bảng hiệu có chữ ..."). Thay dữ liệu đã kiểm chứng bằng dữ
            # liệu của một prompt tổng quát có thể làm hỏng đúng những truy vấn
            # đang dùng để đo — tức tự tạo ra một "cải thiện" hoặc "suy giảm"
            # không liên quan gì tới hệ thống.
            #
            # Hợp nhất giữ được dữ liệu cũ VÀ nâng độ phủ lên ngang các video
            # mới, nên bàn cân đối xứng mà không mất gì.
            row["objects"] = _merge_objects(row.get("objects") or [], fields["objects"])
            row["ocr_instances"] = _merge_ocr(
                row.get("ocr_instances") or [], fields["ocr_instances"]
            )
            row["action_tags"] = sorted(
                set(row.get("action_tags") or []) | set(fields["actions"])
            )
            row.setdefault("extensions", {})["enriched"] = ENRICH_MARKER
            if fields["caption"] and not row.get("captions"):
                row["captions"] = [{
                    "caption_type": "detailed",
                    "confidence": None,
                    "crop_bbox": None,
                    "language": "vi",
                    "text": fields["caption"],
                    "provenance": _provenance(model, "caption"),
                }]
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(row) for row in todo))

    by_id = {row["keyframe_id"]: row for row in keyframes}
    keyframes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keyframes), encoding="utf-8"
    )

    # Repository đọc keyframe LỒNG trong scene; scene còn có `keywords` và
    # `action_tags` riêng mà `bm25_keyword`/`bm25_action` index trên đó.
    scenes = [json.loads(l) for l in scenes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    updated = 0
    for scene in scenes:
        labels: list[str] = []
        actions: list[str] = []
        for nested in scene.get("keyframes", []):
            source = by_id.get(nested.get("keyframe_id"))
            if source is None or not source.get("objects"):
                continue
            nested["objects"] = source["objects"]
            nested["ocr_instances"] = source["ocr_instances"]
            nested["action_tags"] = source["action_tags"]
            if source.get("captions"):
                nested["captions"] = source["captions"]
            labels.extend(item["label"] for item in source["objects"])
            actions.extend(source["action_tags"])
            updated += 1
        if labels:
            existing = {
                str(item.get("text") or "") for item in (scene.get("keywords") or [])
            }
            scene["keywords"] = (scene.get("keywords") or []) + [
                keyword_block(text) for text in sorted(set(labels) - existing)
            ]
        if actions:
            scene["action_tags"] = sorted(set(scene.get("action_tags") or []) | set(actions))
    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes), encoding="utf-8"
    )
    print(f"xong: {done} thành công, {failed} hỏng, {updated} keyframe cập nhật trong scene")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinh object/OCR/action/keyword cho export")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=Path("storage/cache/export_enrich"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
