"""Clip-level embedding pooling — Search Mixing Console W1.

Baseline V1 does not extract any new frames. It slides a time window across
each scene's *existing* keyframes (already produced by scene enrichment) and
pools their embeddings by normalized mean. With sparse keyframe density
(``AIC_KEYFRAMES_PER_SCENE=1`` default) many clips degenerate to a single
pooled frame — that is valid baseline behavior, not an error. See
docs/15_RESEARCH_AGENDA.md "Clip embedding density" for the planned ablation
that would justify denser per-clip sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from datasection.schemas import ClipSegment, EmbeddingReference, Keyframe, ModelProvenance, Scene, VectorLocation
from offline.embedding_reader import EmbeddingReader


@dataclass(frozen=True, slots=True)
class ClipWindow:
    start_sec: float
    end_sec: float
    start_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class SelectedFrames:
    keyframes: list[Keyframe]
    sampling_method: str
    sampling_degraded: bool
    fallback_distance_sec: float | None


def build_clip_windows(
    scene: Scene, fps: float, duration_sec: float, stride_sec: float, min_tail_sec: float,
) -> list[ClipWindow]:
    """Slide a `duration_sec` window across the scene every `stride_sec`.

    A final window shorter than `min_tail_sec` is merged into the previous
    window instead of kept as a tiny trailing clip. The first/last window
    always anchor to the scene's exact frame boundary (not an fps-rounded
    value) so a clip can never be rejected for exceeding its scene's range.
    """

    scene_start, scene_end = scene.start_sec, scene.end_sec
    if scene_end <= scene_start:
        return []
    raw: list[list[float]] = []
    cursor = scene_start
    while True:
        window_end = min(cursor + duration_sec, scene_end)
        raw.append([cursor, window_end])
        if window_end >= scene_end:
            break
        cursor += stride_sec
    if len(raw) > 1 and (raw[-1][1] - raw[-1][0]) < min_tail_sec:
        raw[-2][1] = scene_end
        raw.pop()
    windows: list[ClipWindow] = []
    last_index = len(raw) - 1
    for index, (start_sec, end_sec) in enumerate(raw):
        start_frame = scene.start_frame if index == 0 else round(start_sec * fps)
        end_frame = scene.end_frame_exclusive if index == last_index else round(end_sec * fps)
        start_frame = max(scene.start_frame, min(start_frame, scene.end_frame_exclusive - 1))
        end_frame = max(start_frame + 1, min(end_frame, scene.end_frame_exclusive))
        windows.append(ClipWindow(start_sec=start_sec, end_sec=end_sec, start_frame=start_frame, end_frame=end_frame))
    return windows


def select_clip_frames(
    scene: Scene, window: ClipWindow, max_sampled_frames: int, fallback_max_distance_sec: float, empty_window_policy: str,
) -> SelectedFrames | None:
    """Pick the keyframes a clip pools. Falls back to the nearest scene
    keyframe (bounded by `fallback_max_distance_sec`) when the window
    contains none; returns None when even that fallback is unavailable, so
    the caller can skip the clip with a visible warning."""

    in_window = [frame for frame in scene.keyframes if window.start_frame <= frame.frame_idx < window.end_frame]
    if in_window:
        sampled = in_window[:max_sampled_frames]
        return SelectedFrames(keyframes=sampled, sampling_method="in_window", sampling_degraded=False, fallback_distance_sec=None)
    if empty_window_policy == "skip":
        return None
    window_center = (window.start_sec + window.end_sec) / 2
    nearest = min(scene.keyframes, key=lambda frame: abs(frame.timestamp_sec - window_center))
    distance = abs(nearest.timestamp_sec - window_center)
    if distance > fallback_max_distance_sec:
        return None
    return SelectedFrames(keyframes=[nearest], sampling_method="nearest_scene_keyframe", sampling_degraded=True, fallback_distance_sec=distance)


def pool_embeddings(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot pool an empty list of vectors")
    dimension = len(vectors[0])
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("all pooled vectors must share the same dimension")
    mean = [sum(values) / len(vectors) for values in zip(*vectors, strict=True)]
    norm = math.sqrt(sum(value * value for value in mean))
    if norm == 0.0:
        raise ValueError("pooled clip embedding has zero norm")
    return [value / norm for value in mean]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def choose_representative_frame(
    keyframes: list[Keyframe], vectors: list[list[float]], pooled: list[float], window_center_sec: float,
) -> Keyframe:
    """Closest keyframe to the pooled vector; ties broken by selection_score
    (higher first), distance to the window's time center, then frame_idx."""

    def sort_key(pair: tuple[Keyframe, list[float]]) -> tuple[float, float, float, int]:
        keyframe, vector = pair
        similarity = _cosine_similarity(vector, pooled)
        quality = keyframe.selection_score if keyframe.selection_score is not None else float("-inf")
        distance = abs(keyframe.timestamp_sec - window_center_sec)
        return (-similarity, -quality, distance, keyframe.frame_idx)

    best_keyframe, _ = min(zip(keyframes, vectors, strict=True), key=sort_key)
    return best_keyframe


def _write_vector_atomic(path: Path, vector: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(vector, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


async def build_clip_segment(
    scene: Scene,
    window: ClipWindow,
    selected: SelectedFrames,
    embedding_reader: EmbeddingReader,
    data_root: Path,
    clip_config_id: str,
    provenance: ModelProvenance,
    model_name: str,
    model_revision: str | None,
) -> ClipSegment:
    """Pool `selected`'s embeddings and write the resulting ClipSegment.

    The pooled vector is written to disk atomically (temp file + replace)
    before the EmbeddingReference is constructed, so a clip never references
    a vector file that doesn't fully exist on disk.
    """

    vectors = [await embedding_reader.read(frame) for frame in selected.keyframes]
    pooled = pool_embeddings(vectors)
    window_center = (window.start_sec + window.end_sec) / 2
    representative = choose_representative_frame(selected.keyframes, vectors, pooled, window_center)
    clip_id = f"{scene.scene_id}_C{window.start_frame:08d}_{window.end_frame:08d}"
    action_tags = sorted({tag for frame in selected.keyframes for tag in frame.action_tags})
    relative_vector_path = Path("processed/clip_embeddings") / scene.video_id / f"{clip_id}.json"
    _write_vector_atomic(data_root / relative_vector_path, pooled)
    embedding_ref = EmbeddingReference(
        embedding_name="visual_pool_v1",
        modality="image",
        model_name=model_name,
        model_revision=model_revision,
        dimension=len(pooled),
        normalized=True,
        storage_locations=[VectorLocation(
            backend="file", vector_id=clip_id, index_name="clip_pool_v1", vector_uri=relative_vector_path.as_posix(),
        )],
    )
    return ClipSegment(
        clip_id=clip_id,
        video_id=scene.video_id,
        scene_id=scene.scene_id,
        start_sec=window.start_sec,
        end_sec=window.end_sec,
        duration_sec=window.end_sec - window.start_sec,
        start_frame=window.start_frame,
        end_frame=window.end_frame,
        sampled_frame_ids=[frame.keyframe_id for frame in selected.keyframes],
        representative_frame_id=representative.keyframe_id,
        sampling_method=selected.sampling_method,
        sampling_degraded=selected.sampling_degraded,
        fallback_distance_sec=selected.fallback_distance_sec,
        embedding_refs=[embedding_ref],
        action_tags=action_tags,
        clip_config_id=clip_config_id,
        provenance=provenance,
    )


__all__ = [
    "ClipWindow",
    "SelectedFrames",
    "build_clip_windows",
    "select_clip_frames",
    "pool_embeddings",
    "choose_representative_frame",
    "build_clip_segment",
]
