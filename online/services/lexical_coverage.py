"""BM25-01: chấm mức độ *bao phủ* query, không chỉ đếm token khớp.

Bằng chứng dẫn tới module này (docs/20_EXPERIMENT_LOG.md § ROUTE-01): một
scene NGẪU NHIÊN đã trùng sẵn trung bình 2.25 token OCR và 6.75 token ASR với
query. Nghĩa là nền nhiễu của token-overlap rất cao — cộng thêm điểm cho mỗi
token khớp sẽ thưởng cho cả candidate chỉ tình cờ dùng chung vài từ phổ biến.

Giả thuyết ở đây KHÔNG phải "token đơn lẻ luôn gây nhiễu", mà là:

    Candidate đúng thường khớp NHIỀU phần khác nhau của query và khớp các
    token có tính phân biệt cao; candidate sai thường chỉ khớp một mảnh, hoặc
    khớp các token phổ biến.

Ví dụ đã tái hiện, query "cột nước phun lên từ lòng đất":

    gold           khớp  nước + phun + cao      -> 3/3 nhóm
    lở đất         khớp  đất                    -> 1/3 nhóm, toàn token phổ biến
    bản tin lũ     khớp  nước, lên              -> 2/3 nhóm nhưng IDF thấp

KHÔNG lọc cứng: OCR/ASR có thể thiếu từ dù scene đúng, query dài không nhất
thiết mọi từ đều xuất hiện trong metadata, và benchmark nhỏ nên lọc cứng rất
dễ làm tụt candidate recall. Coverage chỉ là bonus/penalty cộng vào điểm
lexical, rồi mới đưa vào Weighted RRF sẵn có.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

# Hư từ tiếng Việt + English function words. Chúng không mang nội dung nên
# không được tính vào mẫu số của coverage, và chính chúng là chỗ tách nhóm
# khái niệm ("cột nước | phun lên | TỪ | lòng đất").
STOPWORDS = frozenset("""
va la co cua mot nhung trong tren duoi voi den tu cac nguoi khi da bi cho ra vao o
mot hai ba nay do kia thi ma nhu de neu con rat qua cung tai boi vi nen
the nao sao gi day day ay o phia ben canh giua sau truoc
a an the this that of in on at to for with from by is are was were be been
and or but not no any some it its his her their
""".split())

# Từ nối/giới từ dùng làm ranh giới nhóm khái niệm.
_GROUP_SEPARATORS = frozenset("""
tu trong tren duoi voi den ra vao o ben canh giua sau truoc qua boi vi
of in on at to for with from by into onto over under
""".split())

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn").replace("đ", "d")


def content_tokens(text: str) -> list[str]:
    """Token mang nội dung, giữ nguyên thứ tự và cho phép lặp."""

    return [
        token
        for token in _TOKEN_RE.findall(text.casefold())
        if len(token) > 1 and strip_accents(token) not in STOPWORDS
    ]


def concept_groups(query: str) -> list[list[str]]:
    """Tách query thành các nhóm khái niệm bằng ranh giới hư từ.

    Không có từ điển đồng nghĩa nên đây là tập con trung thực của ý tưởng
    concept group: mỗi cụm liền mạch giữa hai hư từ là MỘT khái niệm, khớp
    được nếu bất kỳ token nào trong cụm khớp. Với "cột nước phun lên từ lòng
    đất" ta được [[cột, nước], [phun, lên], [lòng, đất]] — đúng ba nhóm mà
    phân tích thủ công đưa ra.
    """

    groups: list[list[str]] = []
    current: list[str] = []
    for token in _TOKEN_RE.findall(query.casefold()):
        bare = strip_accents(token)
        if bare in _GROUP_SEPARATORS:
            if current:
                groups.append(current)
                current = []
            continue
        if len(token) > 1 and bare not in STOPWORDS:
            current.append(token)
    if current:
        groups.append(current)
    return groups


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """Hệ số của các biến thể ablation BM25-01.

    A = tất cả bằng 0 (BM25 nguyên bản).
    """

    unique_weight: float = 0.0      # B — tỉ lệ token nội dung khớp
    idf_weight: float = 0.0         # C — tỉ lệ IDF khớp
    group_weight: float = 0.0       # D — tỉ lệ nhóm khái niệm khớp
    phrase_weight: float = 0.0      # thưởng khi cả cụm xuất hiện liền
    partial_penalty: float = 0.0    # phạt khi chỉ khớp <= 1/3 số nhóm

    @property
    def is_noop(self) -> bool:
        return not any(
            (self.unique_weight, self.idf_weight, self.group_weight,
             self.phrase_weight, self.partial_penalty)
        )


@dataclass(frozen=True, slots=True)
class CoverageResult:
    unique: float = 0.0
    idf_weighted: float = 0.0
    group: float = 0.0
    phrase: float = 0.0
    matched_terms: tuple[str, ...] = ()

    def adjustment(self, config: CoverageConfig) -> float:
        """Cộng/trừ vào điểm BM25. Dương = candidate bao phủ query tốt."""

        bonus = (
            config.unique_weight * self.unique
            + config.idf_weight * self.idf_weighted
            + config.group_weight * self.group
            + config.phrase_weight * self.phrase
        )
        # Chỉ chạm được một mảnh nhỏ của query -> gần như chắc chắn là khớp
        # nhầm token phổ biến (ca "lở đất" khớp mỗi "đất").
        if self.group <= 1 / 3:
            bonus -= config.partial_penalty * (1.0 - self.group)
        return bonus


def compute_coverage(
    query: str,
    document_text: str,
    idf: dict[str, float] | None = None,
) -> CoverageResult:
    """Đo query được bao phủ tới đâu bởi MỘT field text của candidate."""

    q_tokens = content_tokens(query)
    if not q_tokens:
        return CoverageResult()
    doc_tokens = set(_TOKEN_RE.findall(document_text.casefold()))
    matched = {token for token in q_tokens if token in doc_tokens}

    unique_terms = set(q_tokens)
    unique = len(matched) / len(unique_terms)

    if idf:
        total_idf = sum(idf.get(token, 0.0) for token in unique_terms)
        matched_idf = sum(idf.get(token, 0.0) for token in matched)
        idf_weighted = matched_idf / total_idf if total_idf > 0 else unique
    else:
        idf_weighted = unique

    groups = concept_groups(query)
    if groups:
        hit_groups = sum(1 for group in groups if any(token in doc_tokens for token in group))
        group = hit_groups / len(groups)
    else:
        group = unique

    # Cụm liền mạch xuất hiện nguyên vẹn là tín hiệu mạnh hơn hẳn token rời.
    lowered = document_text.casefold()
    phrases = [" ".join(g) for g in groups if len(g) > 1]
    phrase = (
        sum(1 for p in phrases if p in lowered) / len(phrases) if phrases else 0.0
    )

    return CoverageResult(
        unique=unique, idf_weighted=idf_weighted, group=group, phrase=phrase,
        matched_terms=tuple(sorted(matched)),
    )


__all__ = [
    "CoverageConfig",
    "CoverageResult",
    "compute_coverage",
    "concept_groups",
    "content_tokens",
    "STOPWORDS",
]
