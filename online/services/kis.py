"""Textual KIS processor: known-item signature + safe-frame (PR-07).

KIS là bài toán *known item*: có đúng một khoảnh khắc thỏa mãn, và query mô
tả nó bằng vài dấu hiệu rất cụ thể. Ranking thuần theo điểm fusion bỏ qua
điều này — nó không phân biệt được "khớp 3/3 dấu hiệu bắt buộc" với "khớp
mạnh 1 dấu hiệu và thiếu 2 cái còn lại".

`KisSignature` tách query thành:

    must_match    dấu hiệu bắt buộc (cụm trong ngoặc kép, danh từ riêng, số)
    rare_cues     dấu hiệu hiếm — khớp được thì gần như chắc chắn đúng
    nice_to_have  phần còn lại, chỉ cộng nhẹ
    negative      thứ KHÔNG được xuất hiện

Trước PR-07 `rules.py` suy ra "must-match" chỉ từ dấu ngoặc kép, nên query
không có ngoặc kép hoàn toàn không được hưởng cơ chế này.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.normalizers import ScoreNormalizers

from dataclasses import dataclass, field
import re

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.evidence import EvidencePack
from online.domain.models import SceneDocument, SearchHit
from online.domain.task_results import KisResultItem
from online.services.negative_constraints import extract_negative_constraints
from online.services.query_planner import QUOTED_RE
from online.services.safe_frame import SafeFrameConfig, select_safe_frame

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)
NUMBER_RE = re.compile(r"\b\d[\d.,]*\b")
# Danh từ riêng: chữ hoa giữa câu (kể cả acronym viết hoa toàn bộ như
# "UNESCO"). Tiếng Việt viết hoa tên riêng nên đây là tín hiệu rẻ và khá
# chắc cho "dấu hiệu hiếm".
PROPER_NOUN_RE = re.compile(
    r"(?<!^)(?<![.!?]\s)\b([A-ZĐÀ-Ỹ][a-zà-ỹA-ZĐÀ-Ỹ]{1,}(?:\s+[A-ZĐÀ-Ỹ][a-zà-ỹA-ZĐÀ-Ỹ]{1,})*)"
)

# Từ chức năng tiếng Việt/Anh — không mang thông tin phân biệt.
STOPWORDS = frozenset(
    """
    va co cua mot nguoi nhung trong tren duoi khi la cac nhung nhieu tai voi
    den tu ra vao thi ma nay do dang se da cho ve theo cung nhu hay hoac
    the a an and are as at be by for from has have in is it its of on or that
    to was were will with this these those there their
    tim canh dong hinh anh video khoanh khac
    """.split()
)


@dataclass(frozen=True, slots=True)
class KisSignature:
    """Chữ ký known-item của một query."""

    must_match: tuple[str, ...] = ()
    rare_cues: tuple[str, ...] = ()
    nice_to_have: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()

    def coverage(self, text: str) -> tuple[float, float, bool]:
        """Trả `(must_coverage, nice_coverage, contradicted)` trên một đoạn text."""

        normalized = normalize_vi(text)
        words = set(normalized.split())

        def hit(term: str) -> bool:
            term_norm = normalize_vi(term)
            if not term_norm:
                return False
            if " " in term_norm:
                return term_norm in normalized
            return term_norm in words

        must = [hit(item) for item in self.must_match]
        nice = [hit(item) for item in self.nice_to_have]
        contradicted = any(hit(item) for item in self.negative)
        return (
            sum(must) / len(must) if must else 1.0,
            sum(nice) / len(nice) if nice else 0.0,
            contradicted,
        )

    def rare_hits(self, text: str) -> int:
        normalized = normalize_vi(text)
        return sum(1 for item in self.rare_cues if normalize_vi(item) in normalized)


def build_signature(query: str) -> KisSignature:
    """Tách chữ ký từ query bằng rule thuần (không LLM, deterministic)."""

    quoted = [item.strip() for item in QUOTED_RE.findall(query) if item.strip()]
    numbers = NUMBER_RE.findall(query)
    proper = [item.strip() for item in PROPER_NOUN_RE.findall(query)]
    negatives = extract_negative_constraints(query)

    # Cụm trong ngoặc kép, số và danh từ riêng đều là dấu hiệu hiếm: chúng gần
    # như không xuất hiện ngẫu nhiên ở scene khác.
    rare = list(dict.fromkeys([*quoted, *numbers, *proper]))

    without_quotes = QUOTED_RE.sub(" ", query)
    content: list[str] = []
    for token in TOKEN_RE.findall(without_quotes):
        normalized = normalize_vi(token)
        if len(normalized) < 3 or normalized in STOPWORDS:
            continue
        if normalized not in content:
            content.append(token)

    negative_words = {word for item in negatives for word in normalize_vi(item).split()}
    content = [item for item in content if normalize_vi(item) not in negative_words]

    # must = dấu hiệu hiếm + vài từ nội dung đầu tiên (thường là chủ thể/hành
    # động chính). Lấy quá nhiều sẽ biến must-match thành "phải khớp cả câu".
    must = list(dict.fromkeys([*quoted, *content[:3]]))
    nice = [item for item in content[3:] if item not in must]
    return KisSignature(
        must_match=tuple(must),
        rare_cues=tuple(rare),
        nice_to_have=tuple(nice),
        negative=tuple(negatives),
    )


@dataclass(frozen=True, slots=True)
class KisConfig:
    retrieval_weight: float = 1.0
    must_match_weight: float = 0.6
    rare_cue_weight: float = 0.25
    nice_weight: float = 0.1
    cross_branch_weight: float = 0.15
    safe_frame_weight: float = 0.3
    contradiction_penalty: float = 0.8
    safe_frame: SafeFrameConfig = field(default_factory=SafeFrameConfig)


class KisProcessor:
    """Xếp hạng lại theo chữ ký known-item và chọn frame an toàn để nộp."""

    def __init__(self, config: KisConfig | None = None) -> None:
        self.config = config or KisConfig()

    def rank(
        self,
        query: str,
        hits: list[SearchHit],
        documents: dict[str, SceneDocument],
        *,
        packs: dict[str, EvidencePack] | None = None,
        limit: int = 100,
        normalizers: "ScoreNormalizers | None" = None,
    ) -> list[KisResultItem]:
        signature = build_signature(query)
        config = self.config
        # Điểm fusion không cùng thang giữa các query; chuẩn hóa để trộn được
        # với các thành phần signature nằm trong [0, 1].
        #
        # Mẫu số PHẢI đến từ pool trước dedup (`normalizers`), không phải từ
        # `hits`: `hits` đã bị `fusion.max_results_per_video` cắt, nên lấy max
        # trên nó khiến nới cap là đổi cả thứ hạng đã có (EVAL-01). Fallback
        # về max-trong-tập chỉ để caller cũ/test gọi trực tiếp vẫn chạy.
        if normalizers is not None:
            best_score = normalizers.best_retrieval_score
            branch_ceiling = normalizers.branch_ceiling
        else:
            best_score = max((hit.score for hit in hits), default=0.0) or 1.0
            branch_ceiling = max((len(hit.matched_branches) for hit in hits), default=1) or 1

        scored: list[tuple[float, KisResultItem]] = []
        for hit in hits:
            document = documents.get(hit.scene_id)
            if document is None:
                continue
            text = " ".join(
                [*document.captions, *document.ocr_texts, *document.asr_texts,
                 *document.object_labels, *document.action_tags]
            )
            must_coverage, nice_coverage, contradicted = signature.coverage(text)
            rare = signature.rare_hits(text)
            rare_score = min(rare / len(signature.rare_cues), 1.0) if signature.rare_cues else 0.0
            agreement = len(hit.matched_branches) / branch_ceiling

            safe = select_safe_frame(
                document, query, config.safe_frame, prefer_frame_idx=hit.best_frame_idx
            )
            safe_score = safe.total if safe else 0.0
            frame_idx = safe.frame.frame_idx if safe else hit.best_frame_idx

            total = (
                config.retrieval_weight * (hit.score / best_score)
                + config.must_match_weight * must_coverage
                + config.rare_cue_weight * rare_score
                + config.nice_weight * nice_coverage
                + config.cross_branch_weight * agreement
                + config.safe_frame_weight * safe_score
            )
            if contradicted:
                total -= config.contradiction_penalty

            scored.append((
                total,
                KisResultItem(
                    rank=1,
                    video_id=hit.video_id,
                    frame_idx=frame_idx,
                    scene_id=hit.scene_id,
                    event_id=hit.event_id,
                    score=total,
                    safe_frame_score=safe_score if safe else None,
                    must_match_coverage=must_coverage,
                    evidence=(packs or {}).get(hit.candidate_id),
                ),
            ))

        scored.sort(key=lambda item: (-item[0], item[1].video_id, item[1].frame_idx))
        return [
            item.model_copy(update={"rank": rank})
            for rank, (_total, item) in enumerate(scored[:limit], start=1)
        ]


__all__ = ["KisConfig", "KisProcessor", "KisSignature", "build_signature"]
