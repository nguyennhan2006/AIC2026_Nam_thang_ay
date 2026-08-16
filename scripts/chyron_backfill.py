"""Bù OCR cho CHYRON (dải chữ giới thiệu người nói) mà bộ lấy keyframe bỏ lỡ.

Vấn đề cụ thể, đo được trên L21_V001: chyron chứa **tên + chức vụ + tỉnh** của
người được phỏng vấn — đúng loại danh từ riêng mà KIS/QA bám vào — nhưng nó chỉ
hiện 2-3 giây, trong khi mỗi scene chỉ được lấy MỘT keyframe đại diện. Kết quả::

    64 cửa sổ chyron >= 1s trong V001
       có keyframe rơi vào : 37
       BỎ LỠ hoàn toàn     : 27  (42%)

Và phần lớn hụt cực sát — 2, 3, 2, 3 frame, tức dưới 0,1 giây. Ví dụ thật::

    chyron  frame 10360-10430  "Ông DƯƠNG PHÚ XUÂN / TRƯỞNG PHÒNG KINH TẾ
                                TP. HỒNG NGỰ, TỈNH ĐỒNG THÁP"
    keyframe f10355 = 345.2s   -> sớm hơn đúng 5 frame, không thấy chyron

Hệ quả: hỏi "người đàn ông được phỏng vấn tên gì" thì 20/20 đáp án QA là danh
từ chung ("người", "áo sơ mi"), vì cái tên chưa từng vào index.

Cách làm: quét video tìm dải chyron, trích frame GIỮA mỗi cửa sổ, OCR riêng dải
đó, rồi **gắn thêm** (không ghi đè) vào `ocr_instances` của keyframe gần nhất
trong cùng scene.

Ba lựa chọn có chủ ý:

1. **Gắn thêm chứ không ghi đè.** OCR sẵn có của keyframe vẫn đúng với những gì
   frame đó hiển thị; ta chỉ bổ sung chữ từ một frame lân cận. Khác
   `scripts/ocr_backfill.py` — ở đó ghi đè là đúng vì bản cũ chứa lớp phủ gây hại.

2. **Không thêm keyframe mới.** Thêm frame vào export sẽ đổi tập frame được phép
   nộp, và một frame chỉ-có-chữ gần như chắc chắn là đáp án KIS tệ hơn frame đại
   diện thật. Đây là bù *tín hiệu tìm kiếm*, không phải bù *ứng viên nộp bài*.

3. **bbox là dải thật đã cắt**, không phải khung toàn ảnh. Ta biết chính xác cắt
   ở đâu nên khai đúng chỗ đó; `provenance.parameters.source_frame_idx` ghi frame
   gốc để truy ngược được.

GIỚI HẠN: bộ dò dựa vào dải ĐỎ ở phần ba dưới — đúng với style HTV9 "60 giây"
của corpus hiện tại, KHÔNG tổng quát cho đài khác. Với corpus mới phải đo lại
`--min-red` hoặc đổi sang bộ dò khác. Script in ra số cửa sổ tìm được để phát
hiện ngay khi ngưỡng sai (0 cửa sổ trên một video 20 phút là dấu hiệu sai ngưỡng,
không phải video không có chyron).

    python -m scripts.chyron_backfill --dry-run          # chỉ dò, không gọi API
    python -m scripts.chyron_backfill --video L21_V001
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.adapters.rerank import _image_data_url
from online.config import Settings

MARKER = "chyron-v1"

# Dải DÒ và dải CẮT ĐỂ OCR là hai thứ khác nhau, và tách ra là bắt buộc.
#
# Chyron nằm NGAY TRÊN thanh chữ chạy, cả hai đều nền đỏ. Lần đầu tôi dò trên
# dải rộng 0.62-0.84 và nó hỏng theo hai cách cùng lúc: phần lớn dải là nội dung
# nên tín hiệu loãng còn 6-7%, mà thanh chữ chạy lại lọt vào nên 37% frame BẤT KỲ
# cũng vượt ngưỡng. Đo lại theo từng hàng cho tín hiệu sạch (V001, 41 mẫu):
#
#     y = 0.78-0.86   có chyron 32-44%   không chyron 0.0-0.1%   <- dùng dải này
#     y = 0.88-0.97   có chyron  5%      không chyron 0-3%       <- thanh chữ chạy
#
# Cách nhau ~300 lần, nên ngưỡng đặt ở đâu trong khoảng 5-30 cũng được.
DETECT_TOP = 0.78
DETECT_BOTTOM = 0.86

# Cắt rộng hơn một chút để lấy trọn hai dòng chữ của chyron, nhưng vẫn dừng
# TRƯỚC thanh chữ chạy (0.88) — chữ chạy là tin của bản tin KHÁC, đưa vào OCR là
# bơm nội dung sai vào frame này.
CROP_TOP = 0.76
CROP_BOTTOM = 0.88

NO_TEXT = "KHONG-CO-CHU"

PROMPT = """Đây là dải chữ giới thiệu (chyron) cắt từ một bản tin truyền hình.

Chép NGUYÊN VĂN mọi chữ nhìn thấy, mỗi dòng một chuỗi. Giữ đúng dấu tiếng Việt
và đúng chữ hoa/thường như trên ảnh. Thường có dạng: tên người ở dòng trên,
chức vụ và nơi công tác ở dòng dưới.

Không dịch, không diễn giải, không thêm lời dẫn.

Nếu không có chữ nào, trả về đúng một dòng: KHONG-CO-CHU"""

_NOISE_PREFIX = ("đây là", "ảnh ", "trong ảnh", "chuỗi ", "dòng ", "văn bản")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def detect_chyron_windows(
    video: Path, ffmpeg: str, work_dir: Path, *, stride: int, min_red: float, min_frames: int
) -> list[tuple[int, int]]:
    """Các khoảng `(frame_bắt_đầu, frame_kết_thúc)` có dải chyron.

    Trích sẵn cả dải xuống đĩa rồi đo, thay vì decode hai lần: lượt sau còn cần
    chính những frame này để OCR.
    """

    import numpy
    from PIL import Image

    work_dir.mkdir(parents=True, exist_ok=True)
    for stale in work_dir.glob("*.jpg"):
        stale.unlink()
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(video),
            "-vf", f"select='not(mod(n\\,{stride}))',"
                   f"crop=iw:ih*{DETECT_BOTTOM - DETECT_TOP:.4f}:0:ih*{DETECT_TOP:.4f},scale=200:-1",
            "-vsync", "0", "-q:v", "6", str(work_dir / "b%06d.jpg"), "-y",
        ],
        check=True,
    )

    windows: list[tuple[int, int]] = []
    current: list[int] | None = None
    for index, path in enumerate(sorted(work_dir.glob("b*.jpg"))):
        pixels = numpy.asarray(Image.open(path).convert("RGB")).astype(int)
        red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
        ratio = ((red > 120) & (red - green > 60) & (red - blue > 60)).mean() * 100
        frame_idx = index * stride
        if ratio > min_red:
            current = [frame_idx, frame_idx] if current is None else [current[0], frame_idx]
        else:
            if current and current[1] - current[0] >= min_frames:
                windows.append((current[0], current[1]))
            current = None
    if current and current[1] - current[0] >= min_frames:
        windows.append((current[0], current[1]))
    return windows


def extract_band(video: Path, frame_idx: int, ffmpeg: str, target: Path) -> Path | None:
    """Cắt riêng dải chyron của MỘT frame, ở độ phân giải gốc để đọc được chữ."""

    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(video),
            "-vf", f"select='eq(n\\,{frame_idx})',"
                   f"crop=iw:ih*{CROP_BOTTOM - CROP_TOP:.4f}:0:ih*{CROP_TOP:.4f}",
            "-vsync", "0", "-frames:v", "1", "-q:v", "2", str(target), "-y",
        ],
        capture_output=True,
    )
    return target if result.returncode == 0 and target.exists() else None


def parse_lines(text: str) -> list[str]:
    if NO_TEXT in text.upper():
        return []
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•").strip().strip('"').strip()
        if len(line) < 2 or line.upper() == NO_TEXT:
            continue
        if any(line.lower().startswith(prefix) for prefix in _NOISE_PREFIX):
            continue
        if line not in out:
            out.append(line)
    return out[:8]


def ocr_instance(text: str, model: str, source_frame_idx: int) -> dict:
    return {
        # bbox là DẢI THẬT đã cắt, không phải khung toàn ảnh: ta biết chính xác
        # vùng nào được đưa cho model nên khai đúng vùng đó.
        "bbox": {"x1": 0.0, "y1": CROP_TOP, "x2": 1.0, "y2": CROP_BOTTOM},
        "confidence": 0.0,
        "language": "vi",
        "normalized_text": None,
        "text": text,
        "provenance": {
            "created_at": _now(),
            "device": "unknown",
            "model_name": f"{model}:ocr-chyron",
            "model_revision": MARKER,
            # Frame gốc KHÁC frame được gắn vào — ghi lại để truy ngược được
            # chữ này thật sự đến từ đâu.
            "parameters": {"source_frame_idx": source_frame_idx},
            "pipeline_version": "aic-v1.0.0",
            "prompt_version": "ocr_chyron_v1",
        },
    }


def _scene_for_frame(scenes: list[dict], video_id: str, frame_idx: int) -> dict | None:
    for scene in scenes:
        if scene["video_id"] != video_id:
            continue
        if scene["start_frame"] <= frame_idx < scene["end_frame_exclusive"]:
            return scene
    return None


async def main_async(args: argparse.Namespace) -> None:
    scenes_path = args.export / "scenes.jsonl"
    keyframes_path = args.export / "keyframes.jsonl"
    scenes = [
        json.loads(line)
        for line in scenes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    video_ids = (
        [args.video]
        if args.video
        else sorted({scene["video_id"] for scene in scenes})
    )

    plan: list[tuple[str, int, dict, dict]] = []  # (video_id, frame, scene, keyframe)
    for video_id in video_ids:
        video = args.data_root / "raw" / "videos" / f"{video_id}.mp4"
        if not video.is_file():
            print(f"{video_id}: bỏ qua — không có {video}")
            continue
        windows = detect_chyron_windows(
            video, args.ffmpeg, args.work_dir / video_id / "_scan",
            stride=args.stride, min_red=args.min_red, min_frames=args.min_frames,
        )
        print(f"{video_id}: {len(windows)} cửa sổ chyron")
        if not windows:
            print("   -> 0 cửa sổ: nhiều khả năng --min-red sai với đài này, không phải video không có chyron")
            continue
        for start, end in windows:
            middle = (start + end) // 2
            scene = _scene_for_frame(scenes, video_id, middle)
            if scene is None:
                continue
            frames = scene.get("keyframes") or []
            if not frames:
                continue
            nearest = min(frames, key=lambda item: abs(int(item["frame_idx"]) - middle))
            done = (nearest.get("extensions") or {}).get("chyron_backfill") or []
            if middle in done:
                continue
            plan.append((video_id, middle, scene, nearest))

    if not plan:
        print("không có cửa sổ chyron nào cần bù")
        return
    print(f"cần OCR {len(plan)} dải chyron")
    if args.limit:
        plan = plan[: args.limit]
    if args.dry_run:
        for video_id, frame, scene, nearest in plan[:20]:
            offset = frame - int(nearest["frame_idx"])
            print(f"   {video_id} f{frame} -> {scene['scene_id']} "
                  f"keyframe f{nearest['frame_idx']} (lệch {offset:+d})")
        print("(--dry-run: không gọi API)")
        return

    settings = Settings.from_env()
    if not (settings.fpt_enabled and settings.fpt_api_key):
        raise SystemExit("cần AIC_FPT_ENABLED=true và AIC_FPT_API_KEY")
    client = FptClient.from_settings(settings)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    min_interval = 60.0 / max(args.rpm, 1)
    lock = threading.Lock()
    next_slot = [0.0]

    def slot() -> None:
        with lock:
            start = max(time.monotonic(), next_slot[0])
            next_slot[0] = start + min_interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def call(video_id: str, frame_idx: int) -> list[str] | None:
        video = args.data_root / "raw" / "videos" / f"{video_id}.mp4"
        band = extract_band(
            video, frame_idx, args.ffmpeg,
            args.work_dir / video_id / f"chyron_{frame_idx:06d}.jpg",
        )
        if band is None:
            return None
        digest = hashlib.sha256()
        digest.update(band.read_bytes())
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
                {"type": "image_url", "image_url": {"url": _image_data_url(band)}},
            ],
        }]
        lines = parse_lines(
            client.chat_completion(
                messages, model=args.model, temperature=0.0, max_tokens=300
            ).text
        )
        cached.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
        return lines

    semaphore = asyncio.Semaphore(args.concurrency)
    counters = {"done": 0, "found": 0, "failed": 0}

    async def one(video_id: str, frame_idx: int, nearest: dict) -> None:
        async with semaphore:
            try:
                lines = await asyncio.to_thread(call, video_id, frame_idx)
            except (ProviderError, OSError) as exc:
                counters["failed"] += 1
                print(f"  bỏ qua f{frame_idx}: {str(exc)[:100]}", flush=True)
                return
            if lines is None:
                counters["failed"] += 1
                return
            existing = {item["text"] for item in nearest.get("ocr_instances") or []}
            added = [
                ocr_instance(text, args.model, frame_idx)
                for text in lines
                if text not in existing
            ]
            if added:
                nearest.setdefault("ocr_instances", []).extend(added)
                counters["found"] += 1
            # Đánh dấu kể cả khi không có chữ: "đã thử, không có" là kết quả hợp
            # lệ, lần chạy sau không nên trả tiền lại cho nó.
            marks = nearest.setdefault("extensions", {}).setdefault("chyron_backfill", [])
            if frame_idx not in marks:
                marks.append(frame_idx)
            counters["done"] += 1
            if counters["done"] % 10 == 0:
                print(f"  {counters['done']}/{len(plan)} (có chữ: {counters['found']})", flush=True)

    await asyncio.gather(*(one(vid, frame, nearest) for vid, frame, _scene, nearest in plan))

    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes),
        encoding="utf-8",
    )

    # `keyframes.jsonl` là bản phẳng của cùng dữ liệu. Không đồng bộ thì hai file
    # lệch nhau và lần assemble/kiểm tra sau sẽ khó lần ra vì sao.
    patched = {
        (scene["video_id"], int(frame["frame_idx"])): frame
        for scene in scenes
        for frame in scene.get("keyframes") or []
        if (frame.get("extensions") or {}).get("chyron_backfill")
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
            row["ocr_instances"] = source["ocr_instances"]
            row.setdefault("extensions", {})["chyron_backfill"] = \
                source["extensions"]["chyron_backfill"]
            synced += 1
    keyframes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        f"xong: {counters['done']} xử lý, {counters['found']} có chữ mới, "
        f"{counters['failed']} hỏng, {synced} đồng bộ sang keyframes.jsonl"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bù OCR chyron bị bộ lấy keyframe bỏ lỡ")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--data-root", type=Path, default=Path("storage"))
    parser.add_argument("--video", default=None, help="Chỉ một video, vd L21_V001")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--model", default="gemma-4-31B-it")
    parser.add_argument("--stride", type=int, default=10, help="Lấy mẫu mỗi N frame khi dò")
    parser.add_argument("--min-red", type=float, default=20.0, help="%% điểm ảnh đỏ trong dải 0.78-0.86 để coi là chyron")
    parser.add_argument("--min-frames", type=int, default=30, help="Cửa sổ ngắn hơn thì bỏ")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rpm", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true", help="Chỉ dò và in kế hoạch")
    parser.add_argument("--work-dir", type=Path, default=Path("storage/cache/chyron"))
    parser.add_argument("--cache-dir", type=Path, default=Path("storage/cache/chyron_ocr"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
