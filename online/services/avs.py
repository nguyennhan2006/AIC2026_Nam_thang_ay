"""AVS processor: inclusion/exclusion -> grade 0–3 -> cluster -> MMR (PR-07).

AVS chấm bằng mAP/nDCG nên **độ đa dạng quan trọng ngang độ liên quan**: 20
segment gần như giống nhau của cùng một sự kiện ăn hết top-20 mà chỉ đóng góp
bằng một segment.

`_diversify_avs` trước PR-07 chỉ giới hạn N kết quả mỗi video — không phân
biệt "3 segment của 3 sự kiện khác nhau" với "3 segment của cùng một sự kiện
trong một video dài".

MMR (Maximal Marginal Relevance)::

    MMR(c) = λ · relevance(c) − (1 − λ) · max_similarity(c, đã chọn)

Similarity ở đây là độ chồng lấn từ vựng giữa hai segment — không cần
embedding, và đủ để tách hai sự kiện khác nhau.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.evidence import EvidencePack
from online.domain.task_results import AvsResultItem
from online.services.negative_constraints import extract_negative_constraints

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)
_OR_SPLIT = re.compile(r"\s+(?:hoặc|hay|or)\s+", flags=re.IGNORECASE)
_AND_SPLIT = re.compile(r"[,;]|\s+(?:và|and)\s+", flags=re.IGNORECASE)

STOPWORDS = frozenset(
    """
    tim cac doan canh nhung co the mot voi trong tren duoi khi la va cua
    find segments showing scenes with the and are that this those
    """.split()
)


@dataclass(frozen=True, slots=True)
class AvsCriteria:
    """Điều kiện của một truy vấn AVS.

    `inclusion` là danh sách nhóm AND-of-OR: mỗi nhóm phải khớp ít nhất một
    biến thể. Diễn đạt "người lớn và trẻ em trong vườn, đang dạy hoặc tưới
    cây" thành 3 nhóm, trong đó nhóm cuối có 2 lựa chọn.
    """

    inclusion: tuple[tuple[str, ...], ...] = ()
    exclusion: tuple[str, ...] = ()

    def grade(self, text: str) -> int:
        """Chấm 0–3 theo tỉ lệ nhóm inclusion được thỏa mãn."""

        normalized = normalize_vi(text)
        if not normalized:
            return 0
        words = set(normalized.split())
        if any(_matches(item, normalized, words) for item in self.exclusion):
            return 0
        if not self.inclusion:
            return 0
        satisfied = sum(
            1
            for group in self.inclusion
            if any(_matches(item, normalized, words) for item in group)
        )
        ratio = satisfied / len(self.inclusion)
        if ratio >= 0.999:
            return 3
        if ratio >= 0.6:
            return 2
        if ratio > 0.0:
            return 1
        return 0


def _matches(term: str, normalized_text: str, words: set[str]) -> bool:
    term_norm = normalize_vi(term)
    if not term_norm:
        return False
    if " " in term_norm:
        return term_norm in normalized_text
    return term_norm in words


def extract_criteria(query: str) -> AvsCriteria:
    """Tách inclusion/exclusion bằng rule; không LLM, deterministic."""

    exclusions = tuple(extract_negative_constraints(query))
    excluded_words = {word for item in exclusions for word in normalize_vi(item).split()}

    # Bỏ mệnh đề phủ định khỏi phần inclusion, nếu không "không có ô tô" lại
    # biến "ô tô" thành điều kiện phải có.
    positive = query
    for item in exclusions:
        positive = re.sub(
            r"\b(?:không có|không|chẳng có|chẳng|without|not)\s+[^,.;:!?\n]+",
            " ", positive, flags=re.IGNORECASE,
        )

    groups: list[tuple[str, ...]] = []
    for chunk in _AND_SPLIT.split(positive):
        options: list[str] = []
        for option in _OR_SPLIT.split(chunk):
            terms = [
                token
                for token in TOKEN_RE.findall(option)
                if len(normalize_vi(token)) >= 3
                and normalize_vi(token) not in STOPWORDS
                and normalize_vi(token) not in excluded_words
            ]
            if terms:
                # Giữ nguyên cụm: "xanh dương" khác "xanh" + "dương".
                options.append(" ".join(terms))
        if options:
            groups.append(tuple(options))
    return AvsCriteria(inclusion=tuple(groups), exclusion=exclusions)


def _tokens(pack: EvidencePack) -> set[str]:
    return {
        token
        for token in normalize_vi(pack.rerank_text(max_chars=2000)).split()
        if token not in STOPWORDS and len(token) >= 3
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


@dataclass(frozen=True, slots=True)
class AvsConfig:
    mmr_lambda: float = 0.7
    min_grade: int = 1
    max_per_video: int = 3
    # Hai segment giống nhau tới mức này coi như cùng một sự kiện.
    cluster_threshold: float = 0.6


class AvsProcessor:
    """Chấm relevance, gom cụm sự kiện và chọn theo MMR."""

    def __init__(self, config: AvsConfig | None = None) -> None:
        self.config = config or AvsConfig()

    def rank(
        self,
        query: str,
        packs: list[EvidencePack],
        *,
        retrieval_scores: dict[str, float] | None = None,
        limit: int = 100,
    ) -> list[AvsResultItem]:
        config = self.config
        criteria = extract_criteria(query)
        scores = retrieval_scores or {}
        best_score = max(scores.values(), default=1.0) or 1.0

        graded: list[tuple[EvidencePack, int, float, set[str]]] = []
        for pack in packs:
            text = pack.rerank_text(max_chars=4000)
            grade = criteria.grade(text)
            if grade < config.min_grade:
                continue
            relevance = (grade / 3.0) * 0.7 + (scores.get(pack.candidate_id, 0.0) / best_score) * 0.3
            graded.append((pack, grade, relevance, _tokens(pack)))

        graded.sort(key=lambda item: (-item[2], item[0].candidate_id))
        clusters = self._cluster(graded, config.cluster_threshold)

        selected: list[tuple[EvidencePack, int, float, set[str]]] = []
        selected_tokens: list[set[str]] = []
        per_video: dict[str, int] = {}
        remaining = list(graded)
        while remaining and len(selected) < limit:
            best_index = 0
            best_value = float("-inf")
            for index, (pack, _grade, relevance, tokens) in enumerate(remaining):
                if per_video.get(pack.video_id, 0) >= config.max_per_video:
                    continue
                redundancy = max(
                    (jaccard(tokens, chosen) for chosen in selected_tokens), default=0.0
                )
                value = config.mmr_lambda * relevance - (1 - config.mmr_lambda) * redundancy
                if value > best_value:
                    best_value, best_index = value, index
            if best_value == float("-inf"):
                break
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            selected_tokens.append(chosen[3])
            per_video[chosen[0].video_id] = per_video.get(chosen[0].video_id, 0) + 1

        return [
            AvsResultItem(
                rank=rank,
                video_id=pack.video_id,
                segment_id=pack.scene_id or pack.candidate_id,
                start_frame=pack.start_frame,
                end_frame=pack.end_frame_exclusive - 1,
                relevance_grade=grade,
                score=relevance,
                cluster_id=clusters.get(pack.candidate_id),
                best_frame_idx=pack.best_frame_idx,
            )
            for rank, (pack, grade, relevance, _tokens) in enumerate(selected, start=1)
        ]

    @staticmethod
    def _cluster(
        graded: list[tuple[EvidencePack, int, float, set[str]]], threshold: float
    ) -> dict[str, str]:
        """Gom cụm tham lam theo độ chồng lấn từ vựng; trả candidate_id -> cluster_id."""

        assignments: dict[str, str] = {}
        centroids: list[tuple[str, set[str]]] = []
        for pack, _grade, _relevance, tokens in graded:
            match = next(
                (name for name, centroid in centroids if jaccard(tokens, centroid) >= threshold),
                None,
            )
            if match is None:
                match = f"event_cluster_{len(centroids):02d}"
                centroids.append((match, tokens))
            assignments[pack.candidate_id] = match
        return assignments


__all__ = ["AvsConfig", "AvsCriteria", "AvsProcessor", "extract_criteria", "jaccard"]
