"""Dựng EvidencePack cho top candidate (PR-06).

Lazy theo chủ đích: `_hydrate` chạy trên toàn bộ top-K để hiển thị, còn pack
đầy đủ (neighbor context, provenance) chỉ dựng cho số ít candidate thật sự
đi vào rerank/verify hoặc được người dùng mở ra xem.
"""

from __future__ import annotations

from online.domain.candidate import Candidate
from online.domain.evidence import EvidencePack, NeighborContext, RuleAdjustment
from online.domain.models import SceneDocument
from online.domain.scores import BranchScore
from online.ports.interfaces import SceneRepository


def _neighbor(document: SceneDocument | None) -> NeighborContext | None:
    if document is None:
        return None
    return NeighborContext(
        scene_id=document.scene_id,
        start_frame=document.start_frame,
        end_frame_exclusive=document.end_frame_exclusive,
        start_sec=document.start_sec,
        end_sec=document.end_sec,
        caption=" ".join(document.captions)[:500] or None,
        ocr_text=" ".join(document.ocr_texts)[:500] or None,
    )


class EvidenceBuilder:
    """Ghép candidate + metadata scene thành một EvidencePack tự chứa."""

    def __init__(
        self,
        repository: SceneRepository,
        *,
        dataset_version: str | None = None,
        model_versions: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        self.dataset_version = dataset_version
        self.model_versions = model_versions or {}

    async def _neighbors(
        self, document: SceneDocument
    ) -> tuple[NeighborContext | None, NeighborContext | None]:
        # Scene liền kề suy ra từ scene_idx: rẻ hơn nhiều so với quét theo thời
        # gian, và scene_idx đã được assemble đánh liên tục theo trình tự.
        video = document.video_id
        previous_id = f"{video}_S{document.scene_idx - 1:04d}" if document.scene_idx else None
        next_id = f"{video}_S{document.scene_idx + 1:04d}"
        previous = await self.repository.get(previous_id) if previous_id else None
        following = await self.repository.get(next_id)
        return _neighbor(previous), _neighbor(following)

    async def build(
        self, candidate: Candidate, *, best_frame_idx: int | None = None, with_neighbors: bool = True
    ) -> EvidencePack | None:
        if not candidate.scene_id:
            return None
        document = await self.repository.get(candidate.scene_id)
        if document is None:
            return None
        payload = candidate.payload
        previous, following = (
            await self._neighbors(document) if with_neighbors else (None, None)
        )
        branch_scores = {
            source: BranchScore(raw_score=float(score), score_kind="fusion")
            for source, score in (payload.get("component_scores") or {}).items()
        }
        adjustments = [
            RuleAdjustment(rule=name, delta=float(delta))
            for name, delta in (payload.get("rule_adjustments") or {}).items()
        ]
        return EvidencePack(
            candidate_id=candidate.candidate_id,
            video_id=candidate.video_id,
            scene_id=document.scene_id,
            event_id=candidate.event_id or document.event_id,
            start_frame=document.start_frame,
            end_frame_exclusive=document.end_frame_exclusive,
            start_sec=document.start_sec,
            end_sec=document.end_sec,
            keyframes=document.keyframes,
            best_frame_idx=best_frame_idx if best_frame_idx is not None else candidate.frame_idx,
            asr_window=" ".join(document.asr_texts)[:2000] or None,
            caption_text=" ".join(document.captions)[:2000] or None,
            ocr_text=" ".join(document.ocr_texts)[:2000] or None,
            previous_context=previous,
            next_context=following,
            branch_scores=branch_scores,
            branch_contributions=dict(payload.get("branch_contributions") or {}),
            rule_adjustments=adjustments,
            model_versions=self.model_versions,
            index_versions={
                item: item for item in sorted(payload.get("matched_branches") or [])
            },
            dataset_version=self.dataset_version or document.artifact_version,
        )

    async def build_many(
        self, candidates: list[Candidate], *, with_neighbors: bool = True
    ) -> list[EvidencePack]:
        packs: list[EvidencePack] = []
        for candidate in candidates:
            pack = await self.build(candidate, with_neighbors=with_neighbors)
            if pack is not None:
                packs.append(pack)
        return packs


__all__ = ["EvidenceBuilder"]
