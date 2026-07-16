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

    @classmethod
    def from_env(cls) -> "OfflineSettings":
        root = Path(os.getenv("AIC_DATA_ROOT", "storage")).resolve()
        provider = os.getenv("AIC_OFFLINE_PROVIDER", "mock").lower()
        if provider not in {"mock", "remote"}:
            raise ValueError("AIC_OFFLINE_PROVIDER must be mock or remote")
        gpu_url = os.getenv("AIC_GPU_URL")
        if provider == "remote" and not gpu_url:
            raise ValueError("AIC_GPU_URL is required for remote provider")
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
        )
