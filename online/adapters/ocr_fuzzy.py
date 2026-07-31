"""OCR fuzzy retriever with Vietnamese normalization (Phương án C).

Nhánh OCR chuyên biệt cho KIS: query có biển hiệu / slogan / tên riêng phải
match được cả khi OCR nhận dạng sai vài ký tự hoặc lệch dấu tiếng Việt.
BM25 trên field ``ocr`` (bm25_ocr) chỉ match token *chính xác và có dấu*,
nên miss các trường hợp như:

    Query : "Gừng cay muối mặn xin đừng quên nhau"
    OCR   : "Mường Cay Muối Mặn" / "Xin Đừng Quên Nhau"

Retriever này xử lý bằng 3 lớp:

1. Chuẩn hóa tiếng Việt: lowercase + bỏ dấu (NFD strip combining marks,
   đ→d) + bỏ ký tự nhiễu.
2. Prefilter bằng character-trigram overlap (rẻ, precompute lúc build) để
   không chạy fuzzy đắt trên toàn bộ corpus.
3. Chấm điểm chi tiết = 0.55 * token-containment (mỗi token query được tính
   là match nếu khớp exact hoặc difflib ratio >= 0.8 với một token OCR)
   + 0.45 * partial-phrase ratio (SequenceMatcher trên cửa sổ trượt theo từ,
   bắt cụm liên tục kiểu slogan).

Chỉ dùng stdlib, thuần deterministic — cùng phong cách với bm25.py.

Cách sử dụng
------------
Là một ``Retriever`` chuẩn (xem ``online/ports/interfaces.py``), thêm vào
danh sách retrievers của ``SearchService``:

    from online.adapters.ocr_fuzzy import OcrFuzzyRetriever

    ocr_fuzzy = await OcrFuzzyRetriever.build(repository)
    retrievers = [dense, *lexical, ocr_fuzzy]
    service = SearchService(repository, retrievers)

Lưu ý khi fuse: retriever này báo modality = OCR, nên nếu chạy *song song*
với bm25_ocr thì nhánh OCR được cộng 2 lần trong weighted RRF (2 danh sách
cùng modality). Đó có thể là điều bạn muốn (OCR quan trọng với KIS), nhưng
hãy đo Recall@K/MRR cả hai cấu hình:
  (a) bm25_ocr + ocr_fuzzy   (OCR weight thực tế ~x2)
  (b) chỉ ocr_fuzzy          (thay thế bm25_ocr)
trước khi chốt. Có thể chạy riêng nhánh này bằng
``python -m scripts.eval_kis --mode ocr_only``.

Tham số chính cần tune trên dev set: ``min_score`` (ngưỡng cắt nhiễu,
mặc định 0.35) và ``fuzzy_token_ratio`` (mặc định 0.8).
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from online.domain.models import Candidate, Modality, QueryPlan, SceneDocument
from online.ports.interfaces import SceneRepository
from online.services.branch_options import effective_limit, effective_weight


_NON_WORD_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt: 'Gừng' -> 'Gung', 'đường' -> 'duong'."""

    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.replace("đ", "d").replace("Đ", "D")


def normalize_vi(text: str) -> str:
    """lowercase + bỏ dấu + bỏ ký tự nhiễu + gộp khoảng trắng."""

    text = strip_diacritics(text).casefold()
    text = _NON_WORD_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def char_trigrams(text: str) -> set[str]:
    """Tập character-trigram dùng cho prefilter rẻ trước fuzzy đắt."""

    padded = f"  {text} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _token_containment(query_tokens: list[str], ocr_tokens: list[str], ratio: float) -> float:
    """Tỉ lệ token query xuất hiện trong OCR (exact hoặc fuzzy >= ratio)."""

    if not query_tokens:
        return 0.0
    ocr_set = set(ocr_tokens)
    matched = 0
    for token in query_tokens:
        if token in ocr_set:
            matched += 1
            continue
        # Fuzzy từng token: bắt lỗi OCR kiểu 'gung' vs 'muong', 'quen' vs 'qven'.
        matcher = SequenceMatcher(a=token)
        for candidate in ocr_tokens:
            if abs(len(candidate) - len(token)) > max(2, len(token) // 3):
                continue
            matcher.set_seq2(candidate)
            if matcher.ratio() >= ratio:
                matched += 1
                break
    return matched / len(query_tokens)


def _partial_phrase_ratio(query: str, ocr_text: str) -> float:
    """Best SequenceMatcher ratio giữa query và cửa sổ trượt theo từ của OCR.

    Bắt trường hợp slogan là một cụm liên tục nằm giữa chuỗi OCR dài.
    Cửa sổ trượt theo *từ* (không theo ký tự) để chi phí bị chặn bởi số từ.
    """

    query_words = query.split()
    ocr_words = ocr_text.split()
    if not query_words or not ocr_words:
        return 0.0
    window = len(query_words)
    if len(ocr_words) <= window:
        return SequenceMatcher(a=query, b=ocr_text).ratio()
    best = 0.0
    matcher = SequenceMatcher(a=query)
    for start in range(len(ocr_words) - window + 1):
        segment = " ".join(ocr_words[start : start + window])
        matcher.set_seq2(segment)
        # quick_ratio là cận trên rẻ; bỏ qua cửa sổ chắc chắn thấp hơn best.
        if matcher.quick_ratio() <= best:
            continue
        best = max(best, matcher.ratio())
    return best


def fuzzy_score(
    normalized_query: str,
    normalized_ocr: str,
    *,
    fuzzy_token_ratio: float = 0.8,
) -> float:
    """Điểm fuzzy trong [0, 1] giữa query đã chuẩn hóa và OCR đã chuẩn hóa."""

    containment = _token_containment(
        normalized_query.split(), normalized_ocr.split(), fuzzy_token_ratio
    )
    partial = _partial_phrase_ratio(normalized_query, normalized_ocr)
    return 0.55 * containment + 0.45 * partial


class OcrFuzzyRetriever:
    """Retriever OCR fuzzy, precompute chuẩn hóa + trigram lúc build."""

    name = "ocr_fuzzy"
    modality = Modality.OCR

    def __init__(
        self,
        documents: list[SceneDocument],
        *,
        min_score: float = 0.35,
        fuzzy_token_ratio: float = 0.8,
        prefilter_multiplier: int = 10,
    ) -> None:
        # Chỉ giữ scene có OCR: nhánh này im lặng với scene không chữ.
        self.entries = [
            (doc, normalized, char_trigrams(normalized))
            for doc in documents
            if (normalized := normalize_vi(" ".join(doc.ocr_texts)))
        ]
        self.min_score = min_score
        self.fuzzy_token_ratio = fuzzy_token_ratio
        self.prefilter_multiplier = prefilter_multiplier

    @classmethod
    async def build(
        cls,
        repository: SceneRepository,
        *,
        min_score: float = 0.35,
        fuzzy_token_ratio: float = 0.8,
    ) -> "OcrFuzzyRetriever":
        return cls(
            await repository.all(),
            min_score=min_score,
            fuzzy_token_ratio=fuzzy_token_ratio,
        )

    def _queries(self, plan: QueryPlan) -> list[str]:
        """Ưu tiên cụm trong ngoặc kép (exact_phrases) — đó chính là ocr_query.

        Nếu planner không tách được cụm nào thì rơi về full query, để nhánh
        OCR vẫn hoạt động với query không có ngoặc kép.
        """

        phrases = [
            phrase
            for event in plan.events
            for phrase in event.exact_phrases
        ]
        raw = phrases or [plan.normalized_query]
        return [q for q in (normalize_vi(item) for item in raw) if q]

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.name, Modality.OCR) <= 0:
            return []
        limit = effective_limit(plan, self.name, limit)
        queries = self._queries(plan)
        if not queries:
            return []
        query_trigrams = [char_trigrams(q) for q in queries]

        # Bước 1: prefilter bằng trigram-overlap, giữ top (limit * multiplier).
        prefilter: list[tuple[float, int]] = []
        for index, (doc, _normalized, trigrams) in enumerate(self.entries):
            overlap = max(len(qt & trigrams) / max(len(qt), 1) for qt in query_trigrams)
            if overlap > 0:
                prefilter.append((overlap, index))
        prefilter.sort(key=lambda item: (-item[0], item[1]))
        shortlist = prefilter[: max(limit * self.prefilter_multiplier, limit)]

        # Bước 2: fuzzy chi tiết + filter, chỉ trên shortlist.
        scored: list[tuple[float, str, SceneDocument, str]] = []
        for _overlap, index in shortlist:
            doc, normalized, _trigrams = self.entries[index]
            if plan.filters.video_ids and doc.video_id not in plan.filters.video_ids:
                continue
            if plan.filters.scene_ids and doc.scene_id not in plan.filters.scene_ids:
                continue
            if plan.filters.has_ocr is False:
                continue
            if (
                plan.filters.start_sec_gte is not None
                and doc.start_sec < plan.filters.start_sec_gte
            ):
                continue
            if (
                plan.filters.end_sec_lte is not None
                and doc.end_sec > plan.filters.end_sec_lte
            ):
                continue
            best_score = 0.0
            best_query = queries[0]
            for query in queries:
                score = fuzzy_score(
                    query, normalized, fuzzy_token_ratio=self.fuzzy_token_ratio
                )
                if score > best_score:
                    best_score, best_query = score, query
            if best_score >= self.min_score:
                scored.append((best_score, best_query, doc, normalized))

        scored.sort(key=lambda item: (-item[0], item[2].scene_id))
        return [
            Candidate(
                entity_id=doc.scene_id,
                scene_id=doc.scene_id,
                video_id=doc.video_id,
                source=self.name,
                modality=self.modality,
                score=score,
                rank=rank,
                payload={
                    # matched_text giữ OCR *gốc còn dấu* để evidence dễ đọc.
                    "matched_text": " ".join(doc.ocr_texts)[:1000],
                    "fuzzy_score": round(score, 4),
                    "matched_query": query,
                },
            )
            for rank, (score, query, doc, _normalized) in enumerate(
                scored[:limit], start=1
            )
        ]
