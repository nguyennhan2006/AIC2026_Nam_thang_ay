"""FFmpeg/ffprobe adapter kept outside domain code."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess


@dataclass(frozen=True, slots=True)
class MediaInfo:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int
    codec: str
    audio_present: bool


def _fraction(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


class FFmpegMedia:
    def probe(self, path: Path) -> MediaInfo:
        command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        video = next(item for item in payload["streams"] if item["codec_type"] == "video")
        fps = _fraction(video.get("avg_frame_rate") or video["r_frame_rate"])
        duration = float(video.get("duration") or payload["format"]["duration"])
        frame_count = int(video.get("nb_frames") or round(duration * fps))
        return MediaInfo(
            fps=fps,
            frame_count=frame_count,
            duration_sec=frame_count / fps,
            width=int(video["width"]),
            height=int(video["height"]),
            codec=video.get("codec_name", "unknown"),
            audio_present=any(item["codec_type"] == "audio" for item in payload["streams"]),
        )

    def extract_frame(self, source: Path, frame_idx: int, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
        command = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"select=eq(n\\,{frame_idx})", "-frames:v", "1", str(temporary),
        ]
        subprocess.run(command, check=True)
        temporary.replace(destination)
