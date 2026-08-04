"""Temporal linking of independently retrieved query events."""

from __future__ import annotations

from online.domain.models import SearchHit, SequenceHit


def link_event_hits(
    event_hits: list[list[SearchHit]],
    *,
    limit: int = 20,
    beam_size: int = 100,
    gap_penalty: float = 0.002,
) -> list[SequenceHit]:
    """Beam-link scenes in the same video with strictly increasing scene_idx."""

    if len(event_hits) < 2 or any(not hits for hits in event_hits):
        return []
    beams: list[tuple[list[SearchHit], float]] = [
        ([hit], hit.score) for hit in event_hits[0]
    ]
    for hits in event_hits[1:]:
        expanded: list[tuple[list[SearchHit], float]] = []
        for sequence, score in beams:
            previous = sequence[-1]
            for hit in hits:
                if hit.video_id != previous.video_id or hit.scene_idx <= previous.scene_idx:
                    continue
                gap = max(0.0, hit.start_sec - previous.end_sec)
                expanded.append((sequence + [hit], score + hit.score - gap_penalty * gap))
        expanded.sort(
            key=lambda item: (-item[1], [scene.scene_id for scene in item[0]])
        )
        beams = expanded[:beam_size]
        if not beams:
            return []
    return [
        SequenceHit(video_id=scenes[0].video_id, score=score, scenes=scenes)
        for scenes, score in beams[:limit]
    ]

