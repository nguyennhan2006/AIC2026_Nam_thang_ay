"""Weighted Reciprocal Rank Fusion across incomparable retrieval scores."""

from __future__ import annotations

from collections import defaultdict

from online.domain.models import Candidate, Modality


def weighted_rrf(
    ranked_lists: list[list[Candidate]],
    modality_weights: dict[Modality, float],
    *,
    rrf_k: int = 60,
    limit: int = 100,
) -> list[Candidate]:
    """Fuse ranks, retain component scores, and return deterministic ordering."""

    totals: defaultdict[str, float] = defaultdict(float)
    representatives: dict[str, Candidate] = {}
    components: defaultdict[str, dict[str, float]] = defaultdict(dict)
    modalities: defaultdict[str, set[str]] = defaultdict(set)
    evidence: defaultdict[str, list[dict]] = defaultdict(list)

    for candidates in ranked_lists:
        for fallback_rank, candidate in enumerate(candidates, start=1):
            rank = candidate.rank or fallback_rank
            weight = modality_weights.get(candidate.modality, 0.0)
            contribution = weight / (rrf_k + rank)
            totals[candidate.scene_id] += contribution
            representatives.setdefault(candidate.scene_id, candidate)
            components[candidate.scene_id][candidate.source] = candidate.score
            modalities[candidate.scene_id].add(candidate.modality.value)
            matched_text = candidate.payload.get("matched_text")
            if matched_text:
                evidence[candidate.scene_id].append(
                    {
                        "modality": candidate.modality.value,
                        "text": matched_text,
                        "score": candidate.score,
                    }
                )

    ordered = sorted(totals, key=lambda item: (-totals[item], item))[:limit]
    output: list[Candidate] = []
    for rank, scene_id in enumerate(ordered, start=1):
        base = representatives[scene_id]
        payload = dict(base.payload)
        payload.update(
            {
                "component_scores": components[scene_id],
                "matched_modalities": sorted(modalities[scene_id]),
                "evidence": evidence[scene_id],
            }
        )
        output.append(
            Candidate(
                entity_id=scene_id,
                scene_id=scene_id,
                video_id=base.video_id,
                source="weighted_rrf",
                modality=base.modality,
                score=totals[scene_id],
                rank=rank,
                payload=payload,
            )
        )
    return output

