"""Event grouping + aggregation — Search Mixing Console W1.

Baseline V1 partitions a video's *consecutive* scenes into events using only
temporal gap and a total-duration cap — no visual similarity model, since no
scene-level visual embedding is persisted yet (see
docs/14_TECHNICAL_PREPARATION.md tech debt "persistent frame-embedding
cache"). A keyword/action-tag Jaccard-overlap signal is available as an
extra merge gate for future tuning but is off by default (`min_text_overlap
= 0.0`): with today's sparse keyframe density, per-scene keyword/action-tag
vocabulary is usually too small for overlap to mean anything, so treating it
as a real "content similarity" signal today would be pretending a weak proxy
does more than it does. See docs/15_RESEARCH_AGENDA.md before raising it.

Every scene ends up in exactly one event — a partition, not a subset. A
scene that doesn't merge with any neighbor becomes a singleton event.
"""

from __future__ import annotations

from datasection.schemas import Event, ModelProvenance, Scene


def _text_overlap(a: Scene, b: Scene) -> float:
    def vocab(scene: Scene) -> set[str]:
        words = {item.normalized_text for item in scene.keywords}
        words |= set(scene.action_tags)
        return words

    va, vb = vocab(a), vocab(b)
    if not va or not vb:
        return 0.0
    return len(va & vb) / len(va | vb)


def group_scenes_into_events(
    scenes: list[Scene], max_gap_sec: float, max_event_duration_sec: float, min_text_overlap: float = 0.0,
) -> list[list[Scene]]:
    """Greedily partition `scenes` (already ordered by scene_idx) into events."""

    if not scenes:
        return []
    groups: list[list[Scene]] = [[scenes[0]]]
    for scene in scenes[1:]:
        current = groups[-1]
        last = current[-1]
        gap = scene.start_sec - last.end_sec
        duration_if_merged = scene.end_sec - current[0].start_sec
        can_merge = (
            gap <= max_gap_sec
            and duration_if_merged <= max_event_duration_sec
            and (min_text_overlap <= 0.0 or _text_overlap(last, scene) >= min_text_overlap)
        )
        if can_merge:
            current.append(scene)
        else:
            groups.append([scene])
    return groups


def build_event(
    video_id: str, event_idx: int, scene_group: list[Scene], event_config_id: str, provenance: ModelProvenance,
) -> Event:
    """Aggregate `scene_group`'s own fields into one Event — no new model output."""

    first, last = scene_group[0], scene_group[-1]
    captions = [record.text for scene in scene_group for record in scene.captions]
    event_caption = " ".join(dict.fromkeys(captions)) or None
    return Event(
        event_id=f"{video_id}_E{event_idx:04d}",
        video_id=video_id,
        event_idx=event_idx,
        scene_ids=[scene.scene_id for scene in scene_group],
        start_frame=first.start_frame,
        end_frame_exclusive=last.end_frame_exclusive,
        start_sec=first.start_sec,
        end_sec=last.end_sec,
        event_caption=event_caption,
        representative_frame_ids=[scene.keyframes[0].keyframe_id for scene in scene_group],
        keywords=sorted({item.normalized_text for scene in scene_group for item in scene.keywords}),
        action_tags=sorted({tag for scene in scene_group for tag in scene.action_tags}),
        event_config_id=event_config_id,
        provenance=provenance,
    )


def link_event_neighbors(events: list[Event]) -> list[Event]:
    """Fill in previous_event_id/next_event_id for a video's ordered events."""

    linked: list[Event] = []
    for index, event in enumerate(events):
        previous_id = events[index - 1].event_id if index > 0 else None
        next_id = events[index + 1].event_id if index < len(events) - 1 else None
        linked.append(event.model_copy(update={"previous_event_id": previous_id, "next_event_id": next_id}))
    return linked


__all__ = ["group_scenes_into_events", "build_event", "link_event_neighbors"]
