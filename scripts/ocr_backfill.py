"""Bù OCR cho keyframe mà lượt enrich tổng quát bỏ sót.

Đo được trước khi làm: 150/765 scene (20%) không có OCR instance nào — độ phủ
84% / 80% / 79% cho V001 / V002 / V003. Sáu truy vấn gold khai
`required_modalities: ["ocr"]` rơi đúng vào phần thiếu này, tức chúng không thể
ăn điểm dù hệ thống hoàn hảo.

**Chữ có thật, chỉ là không trích được.** Thử prompt chuyên OCR trên 6 frame
"không có OCR": tìm thấy chữ ở **6/6**, gồm cả `HTV9`, đồng hồ, và tiêu đề bản
tin đầy đủ. Nên đây là lỗi trích xuất, không phải frame trắng chữ.

Nguyên nhân: prompt enrich tổng quát đòi JSON kèm `bbox` cho từng chuỗi. Bắt
model vừa đọc chữ vừa định vị toạ độ làm nó bỏ sót hẳn — cùng ảnh, prompt chỉ
đòi CHỮ thì ra đầy đủ. Nên ở đây **ưu tiên chữ, hy sinh bbox**.

`bbox` là trường bắt buộc của schema nên phải điền; điền khung toàn ảnh và ghi
`model_name: "<model>:ocr-textonly"` để không ai đọc nhầm đó là toạ độ thật.
Không nhánh retrieval nào đọc bbox (`bm25_ocr` và `ocr_fuzzy` chỉ dùng `text`),
nó chỉ phục vụ hiển thị — nên đánh đổi này rẻ.

Model: `gemma-4-31B-it`. Dò thực tế cho thấy nó nhận ảnh và đọc chữ chính xác
hơn `gemma-3-27b-it` (`"HTV9 HD"` liền một chuỗi, `"nhiều lĩnh vực"` đúng chính
tả thay vì `"nghiêu lĩnh vực"`). `GLM-5.2` và `gpt-oss-120b` không đọc được
ảnh. Dùng model khác `Qwen2.5-VL` còn để tránh đụng trần RPM của nó.
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

MARKER = "ocr_backfill_v2_cropped"

# Vùng lớp phủ, suy TỪ DỮ LIỆU chứ không đoán: thống kê bbox THẬT của 2028
# chuỗi OCR do lượt enrich đầu sinh ra cho thấy phân bố lưỡng cực rõ rệt theo
# trục dọc — 762 chuỗi ở y~0.1 (trung bình 1.6 từ, x>0.75: logo + đồng hồ) và
# 963 chuỗi ở y>0.85 (trung bình 9.3 từ: thanh chữ chạy). Chỉ 323 chuỗi nằm
# giữa, và đó mới là nội dung.
#
# Cắt các vùng này KHỎI ẢNH trước khi gọi model, thay vì lọc chữ sau khi đọc:
# lọc sau không áp dụng được cho phần bù (bbox khung-toàn-ảnh), còn cắt trước
# thì mọi chữ đọc ra đều chắc chắn là nội dung.
BANNER_TOP = 0.86     # mọi thứ dưới mốc này là thanh chữ chạy
CORNER_Y = 0.26       # góc trên...
CORNER_X = 0.74       # ...phải: logo kênh và đồng hồ
NO_TEXT = "KHONG-CO-CHU"

PROMPT = """Đọc TOÀN BỘ chữ nhìn thấy trong ảnh này.

Bao gồm cả: logo kênh, đồng hồ, chữ chạy, tên người, tiêu đề bản tin, biển
hiệu, phụ đề, chữ trên đồ vật.

Chép NGUYÊN VĂN, mỗi chuỗi một dòng. Không dịch, không diễn giải, không thêm
lời dẫn.

Nếu thật sự không có chữ nào, trả về đúng một dòng: KHONG-CO-CHU"""

# Dòng model hay thêm dù prompt cấm.
_NOISE_PREFIX = ("đây là", "day la", "trong ảnh", "trong anh", "các chữ", "cac chu",
                 "chữ trong", "chu trong", "văn bản", "van ban", "text:", "ocr:")


def crop_overlay(path: Path, work_dir: Path) -> Path:
    """Ảnh đã xoá hai vùng lớp phủ. Thiếu Pillow thì trả ảnh gốc."""

    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / f"{path.stem}_c.jpg"
    if target.exists():
        return target
    try:
        from PIL import Image, ImageDraw

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            # Tô đen thay vì cắt bỏ: giữ nguyên tỉ lệ khung hình, và model
            # không phải suy diễn về một bức ảnh bị xén lệch.
            draw.rectangle([0, int(height * BANNER_TOP), width, height], fill=(0, 0, 0))
            draw.rectangle(
                [int(width * CORNER_X), 0, width, int(height * CORNER_Y)], fill=(0, 0, 0)
            )
            image.save(target, format="JPEG", quality=88)
        return target
    except Exception:  # noqa: BLE001 - thiếu Pillow hoặc ảnh lạ
        return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_lines(text: str) -> list[str]:
    """Tách từng chuỗi chữ, bỏ lời dẫn và dấu liệt kê."""

    if NO_TEXT in text.upper():
        return []
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•").strip()
        line = line.strip('"').strip()
        if not line or len(line) < 2:
            continue
        low = line.lower()
        if any(low.startswith(p) for p in _NOISE_PREFIX):
            continue
        if line.upper() == NO_TEXT:
            continue
        if line not in out:
            out.append(line)
    return out[:20]


def ocr_instance(text: str, model: str) -> dict:
    return {
        # Khung TOÀN ẢNH: prompt này cố tình không hỏi toạ độ (hỏi thì model bỏ
        # sót chữ). `model_name` ghi rõ để không ai đọc nhầm là vị trí thật.
        "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        "confidence": 0.0,
        "language": "vi",
        "normalized_text": None,
        "text": text,
        "provenance": {
            "created_at": _now(),
            "device": "unknown",
            "model_name": f"{model}:ocr-textonly",
            "model_revision": MARKER,
            "parameters": {},
            "pipeline_version": "aic-v1.0.0",
            "prompt_version": "ocr_textonly_v1",
        },
    }


async def main_async(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    if not (settings.fpt_enabled and settings.fpt_api_key):
        raise SystemExit("cần AIC_FPT_ENABLED=true và AIC_FPT_API_KEY")
    client = FptClient.from_settings(settings)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    keyframes_path = args.export / "keyframes.jsonl"
    scenes_path = args.export / "scenes.jsonl"
    keyframes = [
        json.loads(line)
        for line in keyframes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Chỉ những keyframe CHƯA có OCR và chưa qua lượt bù này.
    todo = [
        row for row in keyframes
        if (not row.get("ocr_instances") or args.replace or
            (args.redo_backfilled and row.get("extensions", {}).get("ocr_backfill")))
        and row.get("extensions", {}).get("ocr_backfill") != MARKER
        and (args.data_root / row["image_path"]).exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("không có keyframe nào cần bù OCR")
        return
    print(f"bù OCR cho {len(todo)} keyframe (model={args.model}, {args.rpm} RPM)")

    semaphore = asyncio.Semaphore(args.concurrency)
    min_interval = 60.0 / max(args.rpm, 1)
    lock = threading.Lock()
    next_slot = [0.0]
    done = found = failed = 0

    def slot() -> None:
        with lock:
            start = max(time.monotonic(), next_slot[0])
            next_slot[0] = start + min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def call(path: Path) -> list[str] | None:
        if args.crop_overlay:
            path = crop_overlay(path, args.work_dir)
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        digest.update(PROMPT.encode("utf-8"))
        digest.update(args.model.encode("utf-8"))
        cached = args.cache_dir / f"{digest.hexdigest()}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        slot()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
            ],
        }]
        text = client.chat_completion(
            messages, model=args.model, temperature=0.0, max_tokens=500
        ).text
        lines = parse_lines(text)
        cached.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
        return lines

    async def one(row: dict) -> None:
        nonlocal done, found, failed
        async with semaphore:
            try:
                lines = await asyncio.to_thread(call, args.data_root / row["image_path"])
            except (ProviderError, OSError) as exc:
                failed += 1
                print(f"  bỏ qua {row['keyframe_id']}: {str(exc)[:100]}", flush=True)
                return
            if lines is None:
                failed += 1
                return
            # Ghi ĐÈ, kể cả khi rỗng: lượt trước đã nhét lớp phủ vào đây,
            # giữ lại là giữ đúng thứ đang gây hại.
            row["ocr_instances"] = [ocr_instance(t, args.model) for t in lines]
            if lines:
                found += 1
            # Đánh dấu kể cả khi KHÔNG có chữ, để lần chạy sau không thử lại —
            # "không có chữ" là kết quả hợp lệ, không phải chưa xử lý.
            row.setdefault("extensions", {})["ocr_backfill"] = MARKER
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)} (có chữ: {found})", flush=True)

    await asyncio.gather(*(one(row) for row in todo))

    by_id = {row["keyframe_id"]: row for row in keyframes}
    keyframes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keyframes),
        encoding="utf-8",
    )

    # Repository đọc keyframe LỒNG trong scene — quên bước này thì OCR nằm
    # trong file mà `bm25_ocr` không thấy.
    scenes = [
        json.loads(line)
        for line in scenes_path.read_text(encoding="utf-8").splitlines()    
        if line.strip()
    ]
    synced = 0
    for scene in scenes:
        for nested in scene.get("keyframes", []):
            source = by_id.get(nested.get("keyframe_id"))
            if source is not None and source.get("extensions", {}).get("ocr_backfill") == MARKER:
                nested["ocr_instances"] = source["ocr_instances"]
                nested.setdefault("extensions", {})["ocr_backfill"] = MARKER
                synced += 1
    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes),
        encoding="utf-8",
    )
    print(f"xong: {done} xử lý, {found} có chữ, {failed} hỏng, {synced} đồng bộ vào scene")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bù OCR cho keyframe còn thiếu")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--model", default="gemma-4-31B-it")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=40)
    parser.add_argument("--redo-backfilled", action="store_true",
                        help="Chi lam lai nhung keyframe da bu o luot truoc (giu nguyen phan co bbox that)")
    parser.add_argument("--crop-overlay", action="store_true",
                        help="To den vung logo/dong ho va thanh chu chay truoc khi OCR")
    parser.add_argument("--replace", action="store_true",
                        help="Xu ly ca keyframe DA co OCR (de thay bang ban da cat lop phu)")
    parser.add_argument("--work-dir", type=Path, default=Path("storage/cache/ocr_crop"))
    parser.add_argument("--cache-dir", type=Path, default=Path("storage/cache/ocr_backfill"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
