"""Chạy thử bộ prompt caption v2 trên FPT API trước khi đốt giờ Kaggle.

Mẫu thử KHÔNG lấy ngẫu nhiên: phần lớn là những frame đã biết là đáp án của
các câu hỏi thi thật (docs/41), nên đọc kết quả là trả lời được đúng câu hỏi
"thẻ mới có chứa đủ chi tiết để khớp câu hỏi không".

    python scripts/pilot_caption_v2.py                 # T1 + SLIDE + T2 + T3
    python scripts/pilot_caption_v2.py --only t1
    python scripts/pilot_caption_v2.py --no-asr        # bỏ lời thuyết minh

Chi phí: mặc định 12 lời gọi. Cổng tốc độ 40 RPM (hạn mức đo được là 50).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from online.adapters.fpt_client import FptClient, image_to_data_url  # noqa: E402
from online.config import Settings  # noqa: E402
from prompts_caption_v2 import (  # noqa: E402
    ASR_CONTEXT_SUFFIX,
    GENRE_SUFFIX,
    GROUP_GENRE,
    FIELDS_KEYFRAME,
    FIELDS_ROLLUP,
    FIELDS_SHOT,
    FIELDS_SLIDE,
    OCR_HINT_SUFFIX,
    PROMPT_KEYFRAME_CARD,
    PROMPT_SHOT_WINDOW,
    PROMPT_SLIDE_CARD,
    PROMPT_VIDEO_ROLLUP,
    dedupe_list_fields,
    missing_fields,
    parse_card,
)

KEYFRAMES = ROOT / "storage/exports_competition/keyframes.jsonl"
SCENES = ROOT / "storage/exports_competition/scenes.jsonl"
DATA_ROOT = ROOT / "storage"
OUT = ROOT / "outputs/caption_v2_pilot"

# (video, frame, prompt, câu hỏi thi mà frame này là đáp án)
SAMPLE: list[tuple[str, int, str, str]] = [
    ("L26_V171", 5998, "t1",
     "P1-8/14: đặt miếng dạng thanh và lát cắt hình hoa vào đĩa đang hấp"),
    ("L29_V013", 11276, "t1",
     "P1-10: cắt chùm nho bằng kéo ĐEN, có dây XANH DƯƠNG buộc cuống"),
    ("L23_V023", 4125, "t1",
     "P1-11: vạch đích đua xe, thứ tự nhất/nhì/ba theo màu áo-quần"),
    ("L21_V010", 19500, "t1",
     "P1-15 (QA): bản đồ động đất, đếm số vị trí cấp độ 4 ngoài bảng chú giải"),
    ("L25_V060", 28500, "slide",
     "P1-23: giáo viên nam + slide sơ đồ 3 tầng, khối hộp cam/xanh"),
]

# Cửa sổ T2: ba keyframe liên tiếp quanh đáp án P1-10 (hành động cắt nho).
WINDOW_VIDEO = "L29_V013"
WINDOW_ANCHOR = 11276
# T3: video đủ ngắn để một lời gọi text-only nuốt hết danh sách cảnh.
ROLLUP_VIDEO = "L23_V023"

RPM = 40


def load_env(path: Path = ROOT / ".env.fpt.local") -> None:
    """Nạp KEY=VALUE; biến đã có trong shell thật giữ nguyên (như các script khác)."""

    if not path.exists():
        raise FileNotFoundError(f"không thấy {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class RateGate:
    """Cổng TỐC ĐỘ, không phải cổng đồng thời.

    Hạ concurrency không chặn được 429 vì nó giới hạn số lệnh gọi SONG SONG
    chứ không giới hạn số lệnh gọi mỗi phút — bài học D2 của docs/27.
    """

    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self._interval:
            time.sleep(self._interval - gap)
        self._last = time.monotonic()


def collect_keyframes(wanted: dict[str, set[int]]) -> dict[tuple[str, int], dict]:
    """Một lượt quét duy nhất qua keyframes.jsonl (744 MB) cho mọi frame cần."""

    found: dict[tuple[str, int], dict] = {}
    with KEYFRAMES.open(encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"video_id": "') + 13
            vid = line[i:i + 8]
            if vid not in wanted:
                continue
            record = json.loads(line)
            if record["frame_idx"] in wanted[vid]:
                found[(vid, record["frame_idx"])] = record
    return found


def neighbour_frames(video: str, anchor: int, count: int = 3) -> list[int]:
    """`count` keyframe liên tiếp bắt đầu từ keyframe gần `anchor` nhất."""

    frames: list[int] = []
    with KEYFRAMES.open(encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"video_id": "') + 13
            if line[i:i + 8] != video:
                continue
            j = line.find('"frame_idx": ', i) + 13
            frames.append(int(line[j:line.find(",", j)]))
    frames.sort()
    if not frames:
        raise SystemExit(f"{video}: không có keyframe nào")
    start = min(range(len(frames)), key=lambda k: abs(frames[k] - anchor))
    start = min(start, max(0, len(frames) - count))
    return frames[start:start + count]


def asr_windows(wanted: dict[str, set[int]], radius_sec: float = 20.0) -> dict[tuple[str, int], str]:
    """Lời thuyết minh quanh mỗi keyframe cần, ±`radius_sec`.

    Cắt theo CỬA SỔ THỜI GIAN chứ không lấy cả scene: có scene dài 845-1417s
    với 249 segment ASR (L25_V060_S0075), nhét cả vào prompt thì lời dẫn át
    hết phần thị giác và model bắt đầu tả thứ nó chỉ nghe thấy.
    """

    stamps: dict[tuple[str, int], float] = {}
    segments: dict[str, list[tuple[float, float, str]]] = {}
    with KEYFRAMES.open(encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"video_id": "') + 13
            vid = line[i:i + 8]
            if vid not in wanted:
                continue
            record = json.loads(line)
            if record["frame_idx"] in wanted[vid]:
                stamps[(vid, record["frame_idx"])] = record["timestamp_sec"]
    with SCENES.open(encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"video_id": "') + 13
            vid = line[i:i + 8]
            if vid not in wanted or '"asr_segments": []' in line:
                continue
            scene = json.loads(line)
            for seg in scene.get("asr_segments", []):
                text = (seg.get("text") or "").strip()
                if text:
                    segments.setdefault(vid, []).append(
                        (float(seg.get("start_sec", 0.0)), float(seg.get("end_sec", 0.0)), text))
    out: dict[tuple[str, int], str] = {}
    for key, ts in stamps.items():
        rows = segments.get(key[0], [])
        near = [t for start, end, t in sorted(rows)
                if end >= ts - radius_sec and start <= ts + radius_sec]
        out[key] = " ".join(near)[:900]
    return out


def scene_lines(video: str, limit: int = 60) -> list[str]:
    """Danh sách cảnh cho T3, dựng từ caption ĐANG CÓ — không gọi VLM."""

    rows: list[tuple[float, float, str]] = []
    with SCENES.open(encoding="utf-8") as fh:
        for line in fh:
            i = line.find('"video_id": "') + 13
            if line[i:i + 8] != video:
                continue
            scene = json.loads(line)
            texts = [
                caption["text"]
                for frame in scene.get("keyframes", [])
                for caption in frame.get("captions", [])
            ]
            if not texts:
                continue
            rows.append((scene["start_sec"], scene["end_sec"], texts[0]))
    rows.sort()
    out = []
    for start, end, text in rows[:limit]:
        out.append(f"{int(start)//60:02d}:{int(start)%60:02d}-"
                   f"{int(end)//60:02d}:{int(end)%60:02d} | {text}")
    return out


def call_vlm(client: FptClient, model: str, prompt: str, images: list[Path],
             gate: RateGate, max_tokens: int, temperature: float = 0.2) -> tuple[str, object]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for path in images:
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(path)}})
    gate.wait()
    result = client.chat_completion(
        [{"role": "user", "content": content}],
        model=model, temperature=temperature, max_tokens=max_tokens,
    )
    return result.text, result.usage


def report(tag: str, note: str, raw: str, fields: tuple[str, ...],
           required: tuple[str, ...], usage) -> dict:
    card = dedupe_list_fields(parse_card(raw, fields))
    gaps = missing_fields(card, required)
    print(f"\n{'=' * 78}\n{tag} — {note}")
    print(f"{'-' * 78}")
    for name in fields:
        value = card[name] or "(rỗng)"
        print(f"  {name:14} {value}")
    print(f"{'-' * 78}")
    print(f"  thiếu trường bắt buộc: {gaps or 'không'}")
    print(f"  token vào/ra: {usage.input_tokens}/{usage.output_tokens}"
          f"  · {usage.latency_ms} ms · retry {usage.retry_count}")
    return {"tag": tag, "note": note, "card": card, "missing": gaps, "raw": raw,
            "tokens_in": usage.input_tokens, "tokens_out": usage.output_tokens,
            "latency_ms": usage.latency_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["t1", "slide", "t2", "t3"], default=None)
    parser.add_argument("--no-ocr-hint", action="store_true",
                        help="bỏ mồi OCR sidecar (mặc định CÓ nối vào prompt)")
    parser.add_argument("--no-asr", action="store_true",
                        help="bỏ lời thuyết minh ASR (mặc định CÓ nối vào prompt)")
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    load_env()
    settings = Settings.from_env()
    model = settings.fpt_vlm_model
    if not model:
        raise SystemExit("AIC_FPT_VLM_MODEL rỗng")
    client = FptClient.from_settings(settings)
    gate = RateGate(RPM)
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    wanted: dict[str, set[int]] = {}
    for video, frame, _, _ in SAMPLE:
        wanted.setdefault(video, set()).add(frame)
    window = neighbour_frames(WINDOW_VIDEO, WINDOW_ANCHOR) if args.only in (None, "t2") else []
    if window:
        wanted.setdefault(WINDOW_VIDEO, set()).update(window)
    records = collect_keyframes(wanted)
    asr = {} if args.no_asr else asr_windows(wanted)

    for video, frame, kind, note in SAMPLE:
        if args.only not in (None, kind):
            continue
        record = records.get((video, frame))
        if record is None:
            print(f"!! {video}/{frame} không có trong keyframes.jsonl — bỏ")
            continue
        image = DATA_ROOT / record["image_path"]
        if not image.exists():
            print(f"!! thiếu ảnh {image} — bỏ")
            continue
        if kind == "slide":
            prompt, fields, required = PROMPT_SLIDE_CARD, FIELDS_SLIDE, ("TOANVAN",)
        else:
            prompt, fields = PROMPT_KEYFRAME_CARD, FIELDS_KEYFRAME
            required = ("MOTA", "DACTRUNG", "TUKHOA")
            if not args.no_ocr_hint:
                hint = "; ".join(o["text"] for o in record.get("ocr_instances", []))
                prompt += OCR_HINT_SUFFIX.format(ocr_hint=hint or "(không có)")
            genre = GROUP_GENRE.get(video[:3])
            if genre:
                prompt += GENRE_SUFFIX.format(genre=genre)
            spoken = asr.get((video, frame), "")
            if spoken:
                prompt += ASR_CONTEXT_SUFFIX.format(asr=spoken)
                print()
                print(f"### ASR ±20s ({len(spoken.split())} từ): {spoken[:160]}")
        raw, usage = call_vlm(client, model, prompt, [image], gate,
                              args.max_tokens, args.temperature)
        old = " | ".join(c["text"] for c in record.get("captions", []))
        print(f"\n### CAPTION CŨ ({len(old.split())} từ): {old[:200]}")
        results.append(report(f"{kind.upper()} {video}/{frame}", note, raw,
                              fields, required, usage))

    if args.only in (None, "t2") and window:
        images = []
        for frame in window:
            record = records.get((WINDOW_VIDEO, frame))
            if record:
                images.append(DATA_ROOT / record["image_path"])
        if len(images) >= 2:
            prompt = PROMPT_SHOT_WINDOW.format(n=len(images))
            raw, usage = call_vlm(client, model, prompt, images, gate,
                                  args.max_tokens, args.temperature)
            results.append(report(f"T2 {WINDOW_VIDEO} {window}",
                                  "cửa sổ 3 keyframe liên tiếp quanh cảnh cắt nho",
                                  raw, FIELDS_SHOT, ("TOMTAT", "HANHDONG", "MAYQUAY"), usage))

    if args.only in (None, "t3"):
        lines = scene_lines(ROLLUP_VIDEO)
        if lines:
            prompt = PROMPT_VIDEO_ROLLUP + "\n\nDANH SÁCH CẢNH:\n" + "\n".join(lines)
            gate.wait()
            result = client.chat_completion(
                [{"role": "user", "content": prompt}],
                model=settings.fpt_fast_llm_model or settings.fpt_llm_model,
                temperature=0.0, max_tokens=2500,
            )
            results.append(report(f"T3 {ROLLUP_VIDEO}",
                                  f"rollup text-only từ {len(lines)} cảnh có caption",
                                  result.text, FIELDS_ROLLUP,
                                  ("TOMTAT_VIDEO", "CHUOI_SUKIEN"), result.usage))

    path = OUT / "pilot_results.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    total_in = sum(r["tokens_in"] for r in results)
    total_out = sum(r["tokens_out"] for r in results)
    print(f"\n{'=' * 78}\n{len(results)} lời gọi · token vào {total_in} / ra {total_out}")
    print(f"ghi: {path}")


if __name__ == "__main__":
    main()
