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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from online.services.normalizers import ScoreNormalizers

from dataclasses import dataclass
import re

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.evidence import EvidencePack
from online.domain.task_results import AvsResultItem
from online.services.keyword_extraction import CorpusIdf
from online.services.negative_constraints import iter_negative_constraints

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
    #: IDF của chính corpus đang tìm. Không có thì độ phủ tính không trọng số —
    #: kém hơn nhưng vẫn tốt hơn hẳn khớp chuỗi con.
    idf: "CorpusIdf | None" = None

    def grade(self, text: str) -> int:
        """Chấm 0–3 theo ĐỘ PHỦ TOKEN của các nhóm inclusion.

        AVS-CRITERIA-01. Bản cũ hỏi "cụm này có xuất hiện nguyên văn không".
        Nhưng `extract_criteria` dựng cụm bằng cách lọc token rồi NỐI phần còn
        sót lại, nên nó tạo ra những chuỗi chưa từng tồn tại trong caption nào
        (`'phóng hoạt động bảo môi trường'`). Đo trên 765 scene: chỉ 16/67 tiêu
        chí từng khớp một scene gold của chính truy vấn nó, và cổng grade loại
        trung bình 3.75 candidate ĐÚNG mỗi truy vấn.

        Không sửa được ở phía trích: mọi phép xoá từ đều phá tính liền mạch, và
        thử giữ âm tiết 2 ký tự hay đổi bộ stopword đều còn 71–77% tiêu chí
        chết. Nên hợp đồng phải đổi — option là một TÚI token, chấm bằng độ phủ.

        Trọng số IDF là cần thiết chứ không phải trang trí: tiêu chí đầy những
        từ như `người`, `hoạt động`, `đang`, `cảnh`, và đếm token thô cho chúng
        ngang với `thợ lặn` hay `rùa biển`.

        Hai phép gộp KHÁC nhau, không được lẫn:
          - giữa các option trong một nhóm (tách bởi "hoặc"/"hay"): `max`,
            vì chúng là các cách nói của cùng một ý;
          - giữa các nhóm: TRUNG BÌNH, vì truy vấn đòi cả "người cứu hộ" lẫn
            "đưa nạn nhân lên xe" thì khớp một nửa không phải khớp.

        Đo bằng `scripts/replay_avs_grading.py` trên cùng 2098 evidence pack:
        nDCG@100 0.545 → 0.598, event_coverage 0.752 → 0.841, candidate đúng bị
        cổng loại 3.75 → 0.42, và tăng trên CẢ BA video. Thưởng cụm nguyên văn
        và thưởng khoảng cách gần (biến thể D) làm tụt xuống 0.589 nên không lấy.
        """

        normalized = normalize_vi(text)
        if not normalized:
            return 0
        words = set(normalized.split())
        if any(_matches(item, normalized, words) for item in self.exclusion):
            return 0
        if not self.inclusion:
            return 0
        ratio = sum(
            max((self._coverage(item, words) for item in group), default=0.0)
            for group in self.inclusion
        ) / len(self.inclusion)
        if ratio >= 0.999:
            return 3
        if ratio >= 0.6:
            return 2
        if ratio > 0.0:
            return 1
        return 0

    def _coverage(self, option: str, words: set[str]) -> float:
        tokens = [token for token in normalize_vi(option).split() if token]
        if not tokens:
            return 0.0
        if self.idf is None:
            return sum(1 for token in tokens if token in words) / len(tokens)
        total = sum(self.idf.idf(token) for token in tokens)
        hit = sum(self.idf.idf(token) for token in tokens if token in words)
        return hit / total if total else 0.0


def _matches(term: str, normalized_text: str, words: set[str]) -> bool:
    term_norm = normalize_vi(term)
    if not term_norm:
        return False
    if " " in term_norm:
        return term_norm in normalized_text
    return term_norm in words


def extract_criteria(query: str, *, idf: "CorpusIdf | None" = None) -> AvsCriteria:
    """Tách inclusion/exclusion bằng rule; không LLM, deterministic."""

    matches = iter_negative_constraints(query)
    exclusions = tuple(phrase for phrase, _span in matches)
    excluded_words = {word for item in exclusions for word in normalize_vi(item).split()}

    # Bỏ mệnh đề phủ định khỏi phần inclusion, nếu không "không có ô tô" lại
    # biến "ô tô" thành điều kiện phải có. Cắt ĐÚNG span đã sinh ra constraint
    # (từ cuối lên đầu để chỉ số không lệch) — chạy lại regex thô ở đây sẽ cắt
    # cả những cụm mà guard danh-từ-ghép đã cố tình giữ lại.
    positive = query
    for _phrase, (start, end) in sorted(matches, key=lambda item: -item[1][0]):
        positive = positive[:start] + " " + positive[end:]

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
    return AvsCriteria(inclusion=tuple(groups), exclusion=exclusions, idf=idf)


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
    # AVS-GRADE-01. `grade()` chấm bằng KHỚP TOKEN NGUYÊN VĂN trên các nhóm
    # inclusion, và `min_grade=1` loại thẳng mọi candidate không khớp chữ nào —
    # tức một cổng từ vựng đứng chắn sau toàn bộ retrieval ngữ nghĩa. Đo được:
    # 3/8 truy vấn trả về ĐÚNG 0 kết quả dù top-100 được phép.
    #
    #   hard_gate           giữ nếu grade >= min_grade (hành vi cũ)
    #   no_gate             không loại ai — để XÁC NHẬN cổng là thủ phạm
    #   soft                không loại; grade thành một số hạng của điểm
    #   semantic_or_lexical giữ nếu điểm ngữ nghĩa >= tau HOẶC grade >= min_grade
    grade_mode: str = "hard_gate"
    # Trọng số của grade khi nó là đặc trưng mềm (mode `soft`).
    soft_lambda: float = 0.3
    # Ngưỡng điểm ngữ nghĩa đủ để tự cứu một candidate (mode `semantic_or_lexical`).
    semantic_tau: float = 0.35


class AvsProcessor:
    """Chấm relevance, gom cụm sự kiện và chọn theo MMR."""

    def __init__(
        self, config: AvsConfig | None = None, *, idf: "CorpusIdf | None" = None
    ) -> None:
        self.config = config or AvsConfig()
        # IDF của corpus, dựng một lần lúc khởi động. `None` thì `grade()` rơi
        # về độ phủ không trọng số — vẫn đúng, chỉ kém phân biệt hơn.
        self.idf = idf

    @staticmethod
    def _keeps(grade: int, semantic: float, config: "AvsConfig") -> tuple[bool, str]:
        """Có giữ candidate không, và nếu không thì vì sao.

        Trả kèm lý do để `diagnostics` phân biệt được "bị cổng từ vựng loại"
        với "vốn không có trong pool" — hai nguyên nhân cần hai cách sửa khác
        hẳn nhau, mà `zero_result_rate` gộp chung không nói được.
        """

        mode = config.grade_mode
        if mode == "no_gate" or mode == "soft":
            return True, ""
        if mode == "semantic_or_lexical":
            if semantic >= config.semantic_tau or grade >= config.min_grade:
                return True, ""
            return False, "below_semantic_tau_and_min_grade"
        # hard_gate
        if grade >= config.min_grade:
            return True, ""
        return False, "min_grade"

    @staticmethod
    def _relevance(grade: int, semantic: float, config: "AvsConfig") -> float:
        """Điểm xếp hạng.

        Bản gốc là `0.7*(grade/3) + 0.3*semantic` — tức grade CHIẾM ƯU THẾ dù
        nó chỉ là khớp token. Mode `soft` đảo lại: ngữ nghĩa là chính, grade chỉ
        là điểm cộng.
        """

        if config.grade_mode == "soft":
            return semantic + config.soft_lambda * (grade / 3.0)
        return (grade / 3.0) * 0.7 + semantic * 0.3

    def rank(
        self,
        query: str,
        packs: list[EvidencePack],
        *,
        retrieval_scores: dict[str, float] | None = None,
        limit: int = 100,
        normalizers: "ScoreNormalizers | None" = None,
        diagnostics: dict | None = None,
    ) -> list[AvsResultItem]:
        config = self.config
        criteria = extract_criteria(query, idf=self.idf)
        scores = retrieval_scores or {}
        # Mẫu số từ pool trước dedup (EVAL-01) — max của lát cắt hiện tại làm
        # thứ hạng phụ thuộc `fusion.max_results_per_video`.
        best_score = (
            normalizers.best_retrieval_score if normalizers is not None
            else (max(scores.values(), default=1.0) or 1.0)
        )

        graded: list[tuple[EvidencePack, int, float, set[str]]] = []
        dropped: list[dict] = []
        for pack in packs:
            text = pack.rerank_text(max_chars=4000)
            grade = criteria.grade(text)
            semantic = scores.get(pack.candidate_id, 0.0) / best_score
            keep, reason = self._keeps(grade, semantic, config)
            if not keep:
                dropped.append({
                    "candidate_id": pack.candidate_id,
                    "video_id": pack.video_id,
                    "best_frame_idx": pack.best_frame_idx,
                    "semantic_score": round(semantic, 4),
                    "lexical_grade": grade,
                    "dropped": True,
                    "drop_reason": reason,
                })
                continue
            relevance = self._relevance(grade, semantic, config)
            graded.append((pack, grade, relevance, _tokens(pack)))

        if diagnostics is not None:
            diagnostics.update({
                "grade_mode": config.grade_mode,
                "min_grade": config.min_grade,
                "max_per_video": config.max_per_video,
                "pre_grade_candidate_count": len(packs),
                "post_grade_candidate_count": len(graded),
                "dropped": dropped,
            })

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
