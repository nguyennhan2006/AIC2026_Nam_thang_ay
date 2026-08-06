"""Phiên âm audio và chiếu ASR vào scene của một export.

Lấp bất đối xứng cuối cùng của bộ dữ liệu đa video: L21_V002/V003 không có
audio nên `bm25_asr` là nhánh CHỈ trả về L21_V001, và mọi phép đo đa video phải
tắt nó đi. Có ASR thì nhánh này mới cạnh tranh được và bàn cân mới đối xứng
hoàn toàn.

Hai đường vào, dùng cái nào cũng được:

``--audio <file>``
    Phiên âm qua FPT (`AIC_FPT_ASR_MODEL`, mặc định `whisper-large-v3-turbo`).
    Nhận cả file video — tự tách audio bằng ffmpeg.

``--segments <file.jsonl>``
    Đã có bản phiên âm sẵn (cùng shape với `input/asr_segments.jsonl` của
    L21_V001: `start_sec`, `end_sec`, `text`). Bỏ qua bước gọi model.

**Chia nhỏ theo thời lượng, không theo dung lượng.** Video ~20 phút cho file
audio hàng chục MB; nhiều endpoint từ chối upload lớn và một lần hỏng là mất
trắng cả lượt. Chia thành lát vài phút rồi cộng bù mốc thời gian: hỏng một lát
chỉ mất một lát, và có cache thì chạy lại chỉ làm phần thiếu.

Chiếu đoạn lời nói vào scene theo **giao nhau về thời gian**, không theo tâm
đoạn: một câu nói kéo dài qua ranh giới scene thì thuộc về CẢ HAI, vì người
tìm kiếm gõ cụm từ đó có thể đang nhớ tới bất kỳ cảnh nào trong hai.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from online.adapters.fpt_client import FptClient
from online.config import Settings

FFMPEG = Path("D:/Shotcut/ffmpeg.exe")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def find_ffmpeg(explicit: Path | None) -> Path:
    for candidate in (explicit, FFMPEG, Path("ffmpeg")):
        if candidate is None:
            continue
        if candidate.exists() or candidate.name == "ffmpeg":
            return candidate
    raise SystemExit("không tìm thấy ffmpeg — truyền --ffmpeg <đường dẫn>")


def extract_audio(source: Path, out_dir: Path, ffmpeg: Path) -> Path:
    """Tách audio 16kHz mono — định dạng whisper mong đợi, và nhỏ hơn nhiều."""

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}.wav"
    if target.exists():
        return target
    subprocess.run(
        [str(ffmpeg), "-y", "-i", str(source), "-vn",
         "-ac", "1", "-ar", "16000", "-f", "wav", str(target)],
        check=True, capture_output=True,
    )
    return target


def split_audio(audio: Path, out_dir: Path, ffmpeg: Path, chunk_sec: int) -> list[tuple[Path, float]]:
    """Cắt thành lát `chunk_sec` giây. Trả `[(file, offset_giây)]`."""

    out_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [str(ffmpeg), "-i", str(audio)], capture_output=True, text=True
    ).stderr
    duration = 0.0
    for token in probe.split():
        if token.startswith("Duration:"):
            continue
    for line in probe.splitlines():
        if "Duration:" in line:
            raw = line.split("Duration:")[1].split(",")[0].strip()
            hours, minutes, seconds = raw.split(":")
            duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            break
    if duration <= chunk_sec:
        return [(audio, 0.0)]

    chunks: list[tuple[Path, float]] = []
    offset = 0.0
    index = 0
    while offset < duration:
        target = out_dir / f"{audio.stem}_c{index:03d}.wav"
        if not target.exists():
            subprocess.run(
                [str(ffmpeg), "-y", "-ss", str(offset), "-t", str(chunk_sec),
                 "-i", str(audio), "-ac", "1", "-ar", "16000", str(target)],
                check=True, capture_output=True,
            )
        chunks.append((target, offset))
        offset += chunk_sec
        index += 1
    return chunks


def transcribe(
    client: FptClient, chunks: list[tuple[Path, float]], model: str, cache_dir: Path
) -> list[dict]:
    """Phiên âm từng lát, cộng bù mốc thời gian về trục của cả video."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    segments: list[dict] = []
    for path, offset in chunks:
        cached = cache_dir / f"{path.stem}.json"
        if cached.exists():
            payload = json.loads(cached.read_text(encoding="utf-8"))
        else:
            print(f"  phiên âm {path.name} (offset {offset:.0f}s)", flush=True)
            payload = client.transcribe(path, model=model)
            cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        for item in payload.get("segments") or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            segments.append({
                "start_sec": float(item.get("start", 0.0)) + offset,
                "end_sec": float(item.get("end", 0.0)) + offset,
                "text": text,
            })
    segments.sort(key=lambda item: item["start_sec"])
    return segments


def _source_id(item: dict, video_id: str, fallback_index: int) -> str:
    """Id nguồn đúng pattern schema `^L\d{2}_V\d{3}_ASR\d{6}$`.

    File đầu vào của whisper dùng `L21_V002_A000000` (một chữ A), còn export
    dùng `..._ASR000000` — `offline/assemble.py` chuyển đổi khi dựng V001.
    Chép nguyên id đầu vào sẽ hỏng validation lúc NẠP, tức sau khi đã ghi file.
    """

    raw = str(item.get("_source_id") or "")
    if "_ASR" in raw:
        return raw
    if "_A" in raw:
        head, _, number = raw.rpartition("_A")
        if number.isdigit():
            return f"{head}_ASR{int(number):06d}"
    return f"{video_id}_ASR{fallback_index:06d}"


def project(scenes: list[dict], video_id: str, segments: list[dict], model: str) -> int:
    """Gán đoạn lời nói vào scene theo giao nhau, CẮT mốc theo biên scene.

    Một câu nói vắt qua ranh giới scene thuộc về CẢ HAI — người tìm kiếm gõ cụm
    từ đó có thể đang nhớ tới bất kỳ cảnh nào trong hai. Nhưng schema
    (`datasection/schemas/scene.py`) bắt buộc mốc ASR nằm TRONG khoảng của
    scene, nên không thể gán nguyên mốc gốc.

    Cách L21_V001 làm, và đây theo đúng: giữ NGUYÊN text, CẮT mốc về biên
    scene. Ví dụ thật trong V001: cùng câu "Đồng bằng sông Cửu Long..." xuất
    hiện ở S0002 [10.29,11.43], S0003 [11.43,13.70] và S0004 [13.70,15.27].

    Hệ quả cần biết: `start_sec`/`end_sec` của đoạn ASR sau khi cắt KHÔNG còn
    là mốc phát âm thật, mà là phần giao với scene. Muốn mốc gốc thì tra
    `source_segment_id` ngược về file đầu vào.
    """

    provenance = {
        "created_at": _now(),
        "device": "unknown",
        "model_name": f"{model}:asr",
        "model_revision": "ingest-asr-v1",
        "parameters": {},
        "pipeline_version": "aic-v1.0.0",
        "prompt_version": None,
    }
    touched = 0
    for scene in scenes:
        if scene["video_id"] != video_id:
            continue
        start, end = float(scene["start_sec"]), float(scene["end_sec"])
        matched = [
            item for item in segments
            if item["end_sec"] > start and item["start_sec"] < end
        ]
        if not matched:
            continue
        scene["asr_segments"] = [
            {
                "confidence": None,
                "end_sec": min(end, item["end_sec"]),
                "language": "vi",
                "normalized_text": None,
                "provenance": provenance,
                "segment_id": f"{scene['scene_id']}_A{index:04d}",
                "source_segment_id": _source_id(item, video_id, index),
                "speaker_id": None,
                "start_sec": max(start, item["start_sec"]),
                "text": item["text"],
            }
            for index, item in enumerate(matched)
        ]
        touched += 1
    return touched


def main() -> None:
    parser = argparse.ArgumentParser(description="Phiên âm + chiếu ASR vào export")
    parser.add_argument("--export", type=Path, default=Path("storage/exports_multivideo"))
    parser.add_argument("--video", required=True, help="vd L21_V002")
    parser.add_argument("--audio", type=Path, default=None, help="File audio HOẶC video")
    parser.add_argument("--segments", type=Path, default=None,
                        help="JSONL đã phiên âm sẵn (start_sec/end_sec/text)")
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--chunk-sec", type=int, default=300)
    parser.add_argument("--work-dir", type=Path, default=Path("storage/cache/asr_work"))
    parser.add_argument("--cache-dir", type=Path, default=Path("storage/cache/asr"))
    args = parser.parse_args()

    if not args.audio and not args.segments:
        raise SystemExit("cần --audio hoặc --segments")

    if args.segments:
        segments = [
            json.loads(line) for line in args.segments.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        segments = [
            {"start_sec": float(s["start_sec"]), "end_sec": float(s["end_sec"]),
             "text": str(s["text"]).strip(),
             "_source_id": s.get("asr_segment_id")}
            for s in segments
            if str(s.get("text") or "").strip()
            and (s.get("video_id") in (None, args.video))
        ]
        model = "provided"
    else:
        settings = Settings.from_env()
        if not (settings.fpt_enabled and settings.fpt_api_key):
            raise SystemExit("cần AIC_FPT_ENABLED=true và AIC_FPT_API_KEY")
        import os

        model = os.getenv("AIC_FPT_ASR_MODEL", "whisper-large-v3-turbo")
        ffmpeg = find_ffmpeg(args.ffmpeg)
        audio = args.audio
        if audio.suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac"):
            print(f"tách audio từ {audio.name}")
            audio = extract_audio(audio, args.work_dir, ffmpeg)
        chunks = split_audio(audio, args.work_dir, ffmpeg, args.chunk_sec)
        print(f"{len(chunks)} lát, mỗi lát {args.chunk_sec}s")
        segments = transcribe(
            FptClient.from_settings(Settings.from_env()), chunks, model, args.cache_dir
        )

    print(f"{len(segments)} đoạn lời nói cho {args.video}")
    if not segments:
        raise SystemExit("không có đoạn nào — dừng, không ghi đè export")

    scenes_path = args.export / "scenes.jsonl"
    scenes = [
        json.loads(line) for line in scenes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    touched = project(scenes, args.video, segments, model)
    scenes_path.write_text(
        "".join(json.dumps(scene, ensure_ascii=False) + "\n" for scene in scenes),
        encoding="utf-8",
    )
    total = sum(1 for scene in scenes if scene["video_id"] == args.video)
    print(f"đã gán ASR cho {touched}/{total} scene của {args.video}")


if __name__ == "__main__":
    main()
