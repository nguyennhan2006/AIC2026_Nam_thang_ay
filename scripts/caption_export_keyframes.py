"""Sinh caption tiếng Việt cho keyframe của một EXPORT bằng FPT VLM.

Vì sao cần riêng: `scripts/enrich_keyframes_fpt.py` đọc stage pack và ghi ra
stage pack — đúng cho luồng offline đầy đủ. Video distractor L21_V002/V003 chỉ
có ảnh + CSV nên không có pack; ở đây đọc/ghi thẳng trên export.

Vì sao distractor CẦN caption: không có caption thì `bm25_caption`,
`bm25_keyword` và text rerank không bao giờ nhìn thấy video mới, nên chúng chỉ
là distractor cho nhánh thị giác. Đo với distractor nửa vời sẽ báo cáo một
mức khó thấp hơn thực tế.

Chỉ sinh **caption**, không sinh OCR/object: caption là thứ nuôi các nhánh
lexical, còn OCR/object cần bbox và kiểm chứng riêng — thêm chúng vào đây là
mở rộng phạm vi mà chưa cần.

Cache theo `sha256(ảnh + prompt + model)` để chạy lại rẻ và để đổi prompt
không phải trả tiền cho phần không đổi.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.adapters.rerank import _image_data_url
from online.config import Settings

PROMPT = """Mô tả khung hình này bằng MỘT câu tiếng Việt.

Chỉ tả những gì NHÌN THẤY: người, vật thể, hành động, bối cảnh. Không suy đoán
nguyên nhân hay cảm xúc. Bỏ qua logo kênh và đồng hồ trên màn hình.

Chỉ trả về câu mô tả, không kèm gì khác."""


def cache_key(image_path: Path, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(image_path.read_bytes())
    digest.update(PROMPT.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    return digest.hexdigest()


def _provenance(model: str, suffix: str) -> dict:
    return {
        # Phải là datetime hợp lệ — chuỗi rỗng làm hỏng validation lúc NẠP,
        # tức sau khi đã trả tiền cho toàn bộ lệnh gọi VLM.
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": "unknown",
        "model_name": f"{model}:{suffix}",
        "model_revision": "distractor-v1",
        "parameters": {},
        "pipeline_version": "aic-v1.0.0",
        "prompt_version": "export_caption_v1",
    }


def keyframe_caption(text: str, model: str) -> dict:
    """Caption ở tầng KEYFRAME.

    Schema khác hẳn tầng scene: `caption_type` là enum
    `short|detailed|tags|crop` (KHÔNG có "visual"), có `crop_bbox`, và KHÔNG
    nhận `evidence_keyframe_ids`. Dùng nhầm một dạng cho cả hai thì file ghi
    xong mới hỏng lúc nạp.
    """

    return {
        "caption_type": "detailed",
        "confidence": None,
        "crop_bbox": None,
        "language": "vi",
        "text": text,
        "provenance": _provenance(model, "caption"),
    }


def scene_caption(text: str, keyframe_id: str, model: str) -> dict:
    """Caption ở tầng SCENE — `caption_type` là "visual" và có evidence."""

    return {
        "caption_type": "visual",
        "confidence": None,
        "evidence_keyframe_ids": [keyframe_id],
        "language": "vi",
        "text": text,
        "provenance": _provenance(model, "scene-caption"),
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
        if not row.get("captions")
        and (not wanted or row["video_id"] in wanted)
        and (args.data_root / row["image_path"]).exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("không có keyframe nào cần caption")
        return
    print(f"cần caption {len(todo)} keyframe (model={model})")

    semaphore = asyncio.Semaphore(args.concurrency)
    done = 0
    # FPT giới hạn 50 RPM cho Qwen2.5-VL-7B-Instruct. Chỉ hạ `concurrency` là
    # không đủ: nó chặn số lệnh gọi ĐỒNG THỜI chứ không chặn TỐC ĐỘ, nên vẫn
    # vượt hạn mức khi mỗi lệnh trả về nhanh. Backoff của FptClient cũng không
    # cứu được vì hạn mức tính theo phút. Đo thực tế: concurrency=6 làm hỏng
    # 233/545 keyframe vì 429.
    min_interval = 60.0 / max(args.rpm, 1)
    rate_lock = threading.Lock()
    next_slot = [0.0]

    def wait_for_slot() -> None:
        with rate_lock:
            now = time.monotonic()
            start = max(now, next_slot[0])
            next_slot[0] = start + min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def call(image_path: Path) -> str:
        key = cache_key(image_path, model)
        cached = args.cache_dir / f"{key}.txt"
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        wait_for_slot()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
            ],
        }]
        text = client.chat_completion(messages, model=model, temperature=0.0, max_tokens=200).text
        text = text.strip().strip('"')
        cached.write_text(text, encoding="utf-8")
        return text

    async def one(row: dict) -> None:
        nonlocal done
        async with semaphore:
            try:
                text = await asyncio.to_thread(call, args.data_root / row["image_path"])
            except (ProviderError, OSError) as exc:
                print(f"  bỏ qua {row['keyframe_id']}: {exc}", flush=True)
                return
            if text:
                row["captions"] = [keyframe_caption(text, model)]
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(row) for row in todo))

    by_id = {row["keyframe_id"]: row.get("captions", []) for row in keyframes}
    keyframes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in keyframes), encoding="utf-8"
    )

    # Repository đọc keyframe LỒNG trong scene, và scene có `captions` riêng mà
    # `bm25_caption` index trên đó. Cập nhật cả hai chỗ, nếu không caption nằm
    # trong file mà không nhánh nào thấy.
    scenes = [json.loads(l) for l in scenes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    touched = 0
    for scene in scenes:
        texts: list[str] = []
        for keyframe in scene.get("keyframes", []):
            captions = by_id.get(keyframe.get("keyframe_id")) or []
            if captions:
                keyframe["captions"] = captions
                texts.extend(block["text"] for block in captions)
                touched += 1
        if texts and not scene.get("captions"):
            scene["captions"] = [
                scene_caption(" ".join(texts), scene["keyframes"][0]["keyframe_id"], model)
            ]
    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes), encoding="utf-8"
    )
    print(f"đã caption {done}/{len(todo)}, cập nhật {touched} keyframe trong scene")


def main() -> None:
    parser = argparse.ArgumentParser(description="Caption keyframe của export bằng FPT VLM")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=40,
                        help="Trần lệnh gọi mỗi phút. FPT cho 50 RPM với VLM; để 40 cho an toàn")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("storage/cache/export_caption"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
