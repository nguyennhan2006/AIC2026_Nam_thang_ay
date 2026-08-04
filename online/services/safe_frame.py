"""Chọn frame an toàn nhất trong một scene (PR-07).

Bài toán: tìm đúng scene mới là một nửa. Nộp `frame_idx` rơi vào frame mờ,
frame dính transition, hoặc frame nằm ngay biên cut thì hoặc mất điểm, hoặc
người chấm nhìn vào không thấy nội dung mà query mô tả.

Điểm safe-frame::

    semantic          độ khớp text của frame với query
  + quality           độ nét, độ sáng hợp lý, không phải frame đen
  + interval_centrality  càng gần giữa scene càng ổn định
  - boundary_penalty  sát biên scene = nguy cơ dính cut
  - blur_penalty      ảnh mờ

Mọi thành phần đều nằm trong [0, 1] và trọng số ở `SafeFrameConfig` cùng
thang, nên chỉnh một trọng số không phá cân bằng các phần còn lại. Tất cả là
điểm khởi đầu — chỉ giữ nếu tăng metric trên `scripts/eval_tasks.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from online.domain.candidate import FrameEvidence
from online.domain.models import SceneDocument

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)

# Ngưỡng độ nét (variance of Laplacian) mà dưới đó ảnh coi như mờ. Giá trị
# phổ biến cho ảnh 720p; notebook keyframe ghi cùng thang đo.
SHARPNESS_FLOOR = 40.0
SHARPNESS_CEILING = 300.0


@dataclass(frozen=True, slots=True)
class SafeFrameConfig:
    semantic_weight: float = 0.45
    quality_weight: float = 0.20
    centrality_weight: float = 0.15
    boundary_penalty: float = 0.15
    blur_penalty: float = 0.20
    # Frame cách biên ít hơn ngần này bị coi là "sát cut".
    boundary_margin_frames: int = 12


@dataclass(frozen=True, slots=True)
class SafeFrameScore:
    frame: FrameEvidence
    total: float
    semantic: float
    quality: float
    centrality: float
    boundary_penalty: float
    blur_penalty: float

    def as_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 4),
            "semantic": round(self.semantic, 4),
            "quality": round(self.quality, 4),
            "centrality": round(self.centrality, 4),
            "boundary_penalty": round(self.boundary_penalty, 4),
            "blur_penalty": round(self.blur_penalty, 4),
        }


def _semantic(frame: FrameEvidence, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    tokens = set(TOKEN_RE.findall(frame.search_text.casefold()))
    return len(query_tokens & tokens) / len(query_tokens)


def _quality(frame: FrameEvidence) -> float:
    """Điểm chất lượng [0, 1]; thiếu tín hiệu thì trả 0.5 (trung tính).

    Trả 0.0 khi thiếu dữ liệu sẽ phạt oan toàn bộ dataset chưa chạy stage
    keyframe quality, biến safe-frame thành thuần semantic mà không ai biết.
    """

    signals: list[float] = []
    quality = frame.quality
    if quality.sharpness is not None:
        span = SHARPNESS_CEILING - SHARPNESS_FLOOR
        signals.append(min(max((quality.sharpness - SHARPNESS_FLOOR) / span, 0.0), 1.0))
    if quality.brightness is not None:
        # Sáng quá hay tối quá đều xấu; 0.5 là lý tưởng.
        signals.append(1.0 - min(abs(quality.brightness - 0.5) * 2.0, 1.0))
    if quality.black_frame_ratio is not None:
        signals.append(1.0 - quality.black_frame_ratio)
    if frame.selection_score is not None:
        signals.append(frame.selection_score)
    return sum(signals) / len(signals) if signals else 0.5


def _blur_penalty(frame: FrameEvidence) -> float:
    sharpness = frame.quality.sharpness
    if sharpness is None or sharpness >= SHARPNESS_FLOOR:
        return 0.0
    return 1.0 - (sharpness / SHARPNESS_FLOOR)


def _centrality(frame: FrameEvidence, scene: SceneDocument) -> float:
    span = scene.end_frame_exclusive - scene.start_frame
    if span <= 1:
        return 1.0
    midpoint = (scene.start_frame + scene.end_frame_exclusive - 1) / 2
    return 1.0 - min(abs(frame.frame_idx - midpoint) / (span / 2), 1.0)


def _boundary_penalty(frame: FrameEvidence, scene: SceneDocument, margin: int) -> float:
    if margin <= 0:
        return 0.0
    distance = frame.boundary_distance_frames
    if distance is None:
        distance = min(
            frame.frame_idx - scene.start_frame,
            scene.end_frame_exclusive - 1 - frame.frame_idx,
        )
    if distance >= margin:
        return 0.0
    return 1.0 - (max(distance, 0) / margin)


def score_frames(
    scene: SceneDocument, query: str, config: SafeFrameConfig | None = None
) -> list[SafeFrameScore]:
    """Chấm mọi keyframe của scene, sắp giảm dần theo tổng điểm."""

    config = config or SafeFrameConfig()
    query_tokens = set(TOKEN_RE.findall(query.casefold()))
    scored: list[SafeFrameScore] = []
    for frame in scene.keyframes:
        semantic = _semantic(frame, query_tokens)
        quality = _quality(frame)
        centrality = _centrality(frame, scene)
        boundary = _boundary_penalty(frame, scene, config.boundary_margin_frames)
        blur = _blur_penalty(frame)
        total = (
            config.semantic_weight * semantic
            + config.quality_weight * quality
            + config.centrality_weight * centrality
            - config.boundary_penalty * boundary
            - config.blur_penalty * blur
        )
        scored.append(
            SafeFrameScore(
                frame=frame,
                total=total,
                semantic=semantic,
                quality=quality,
                centrality=centrality,
                boundary_penalty=boundary,
                blur_penalty=blur,
            )
        )
    # Tie-break theo frame_idx để kết quả tái lập được giữa các lần chạy.
    scored.sort(key=lambda item: (-item.total, item.frame.frame_idx))
    return scored


def select_safe_frame(
    scene: SceneDocument,
    query: str,
    config: SafeFrameConfig | None = None,
    *,
    prefer_frame_idx: int | None = None,
) -> SafeFrameScore | None:
    """Chọn frame tốt nhất; `prefer_frame_idx` là gợi ý từ retrieval mức frame.

    Gợi ý chỉ được tôn trọng khi frame đó không bị phạt nặng: một frame do
    dense index chỉ ra nhưng mờ hoặc dính biên vẫn là frame xấu để nộp.
    """

    scored = score_frames(scene, query, config)
    if not scored:
        return None
    if prefer_frame_idx is not None:
        preferred = next(
            (item for item in scored if item.frame.frame_idx == prefer_frame_idx), None
        )
        if preferred is not None and preferred.blur_penalty == 0.0 and preferred.boundary_penalty < 0.5:
            return preferred
    return scored[0]


__all__ = [
    "SafeFrameConfig",
    "SafeFrameScore",
    "score_frames",
    "select_safe_frame",
]
