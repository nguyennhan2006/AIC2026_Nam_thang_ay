from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _positive(name: str, default: str, cast):
    value = cast(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class OfflineSettings:
    data_root: Path
    input_dir: Path
    export_dir: Path
    state_dir: Path
    gpu_url: str | None
    gpu_api_key: str | None
    timeout_sec: float
    retries: int
    scene_seconds: float
    keyframes_per_scene: int
    pipeline_version: str
    provider: str
    # Clip pooling (Search Mixing Console W1) — sliding window bên trong một scene.
    # Có default để không phá các chỗ dựng OfflineSettings trực tiếp (không qua
    # from_env()) từ trước khi có clip pooling, vd tests/test_offline.py.
    clip_duration_sec: float = 4.0
    clip_stride_sec: float = 2.0
    clip_min_tail_sec: float = 1.0
    clip_max_sampled_frames: int = 4
    clip_empty_window_policy: str = "nearest_scene_keyframe"
    clip_fallback_max_distance_sec: float = 6.0
    clip_config_id: str = "clip-v1"
    # Event grouping (Search Mixing Console W1) — greedy partition of consecutive scenes.
    event_max_gap_sec: float = 2.0
    event_max_duration_sec: float = 60.0
    event_min_text_overlap: float = 0.0
    event_config_id: str = "event-v1"

    @classmethod
    def from_env(cls) -> "OfflineSettings":
        root = Path(os.getenv("AIC_DATA_ROOT", "storage")).resolve()
        provider = os.getenv("AIC_OFFLINE_PROVIDER", "mock").lower()
        if provider not in {"mock", "remote"}:
            raise ValueError("AIC_OFFLINE_PROVIDER must be mock or remote")
        gpu_url = os.getenv("AIC_GPU_URL")
        if provider == "remote" and not gpu_url:
            raise ValueError("AIC_GPU_URL is required for remote provider")
        empty_window_policy = os.getenv("AIC_CLIP_EMPTY_WINDOW_POLICY", "nearest_scene_keyframe")
        if empty_window_policy not in {"nearest_scene_keyframe", "skip"}:
            raise ValueError("AIC_CLIP_EMPTY_WINDOW_POLICY must be nearest_scene_keyframe or skip")
        event_min_text_overlap = float(os.getenv("AIC_EVENT_MIN_TEXT_OVERLAP", "0.0"))
        if not 0.0 <= event_min_text_overlap <= 1.0:
            raise ValueError("AIC_EVENT_MIN_TEXT_OVERLAP must be between 0.0 and 1.0")
        return cls(
            data_root=root,
            input_dir=Path(os.getenv("AIC_INPUT_DIR", str(root / "raw/videos"))),
            export_dir=Path(os.getenv("AIC_EXPORT_DIR", str(root / "exports"))),
            state_dir=Path(os.getenv("AIC_STATE_DIR", str(root / "state"))),
            gpu_url=gpu_url.rstrip("/") if gpu_url else None,
            gpu_api_key=os.getenv("AIC_GPU_API_KEY"),
            timeout_sec=_positive("AIC_GPU_TIMEOUT_SEC", "120", float),
            retries=_positive("AIC_GPU_RETRIES", "3", int),
            scene_seconds=_positive("AIC_SCENE_SECONDS", "8", float),
            keyframes_per_scene=_positive("AIC_KEYFRAMES_PER_SCENE", "1", int),
            pipeline_version=os.getenv("AIC_PIPELINE_VERSION", "aic-v1.0.0"),
            provider=provider,
            clip_duration_sec=_positive("AIC_CLIP_DURATION_SEC", "4", float),
            clip_stride_sec=_positive("AIC_CLIP_STRIDE_SEC", "2", float),
            clip_min_tail_sec=_positive("AIC_CLIP_MIN_TAIL_SEC", "1", float),
            clip_max_sampled_frames=_positive("AIC_CLIP_MAX_SAMPLED_FRAMES", "4", int),
            clip_empty_window_policy=empty_window_policy,
            clip_fallback_max_distance_sec=_positive("AIC_CLIP_FALLBACK_MAX_DISTANCE_SEC", "6", float),
            clip_config_id=os.getenv("AIC_CLIP_CONFIG_ID", "clip-v1"),
            event_max_gap_sec=_positive("AIC_EVENT_MAX_GAP_SEC", "2", float),
            event_max_duration_sec=_positive("AIC_EVENT_MAX_DURATION_SEC", "60", float),
            event_min_text_overlap=event_min_text_overlap,
            event_config_id=os.getenv("AIC_EVENT_CONFIG_ID", "event-v1"),
        )
