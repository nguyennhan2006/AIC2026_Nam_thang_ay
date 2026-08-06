"""Small in-memory BM25 baseline used for local demos and regression tests."""

from __future__ import annotations

from collections import Counter
import math
import re

from online.domain.models import Candidate, Modality, QueryPlan, SceneDocument
from online.ports.interfaces import SceneRepository
from online.services.branch_options import effective_limit, effective_weight
from online.services.lexical_coverage import CoverageConfig, compute_coverage


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _document_id(document) -> str:
    """Stable tie-break id. BM25Index is generic over any document exposing
    `field_text()` — SceneDocument (`scene_id`) and EventDocument (`event_id`,
    see online/adapters/event_search.py) are both used with it today."""

    return getattr(document, "scene_id", None) or document.event_id


_CLOCK = re.compile(r"^\s*\d{1,2}[:.]\d{2}([:.]\d{2})?\s*$")
_DATE = re.compile(r"^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")


# Ranh giới suy từ phân bố bbox, không phải chọn tròn số.
_BANNER_TOP = 0.86
_CORNER_Y = 0.26
_CORNER_X = 0.74


def _boxes(item) -> list[dict | None]:
    """bbox của từng chuỗi OCR, cùng thứ tự với `ocr_texts`.

    `None` cho chuỗi có bbox khung-toàn-ảnh: đó là phần bù bằng prompt
    text-only, không biết vị trí nên không lọc theo vị trí được. Chúng được xử
    lý ở tầng dữ liệu (`scripts/ocr_backfill.py --crop-overlay`) thay vì ở đây.
    """

    out: list[dict | None] = []
    for keyframe in getattr(item, "keyframes", []) or []:
        for instance in getattr(keyframe, "ocr_instances", []) or []:
            box = getattr(instance, "bbox", None)
            if box is None:
                out.append(None)
                continue
            full = (
                box.x1 <= 0.001 and box.y1 <= 0.001
                and box.x2 >= 0.999 and box.y2 >= 0.999
            )
            out.append(None if full else {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2})
    return out


def _in_overlay_band(box: dict) -> bool:
    center_y = (box["y1"] + box["y2"]) / 2
    center_x = (box["x1"] + box["x2"]) / 2
    if center_y > _BANNER_TOP:
        return True
    return center_y < _CORNER_Y and center_x > _CORNER_X


def _strip_overlay(documents, field: str, df_threshold: float, max_words: int = 0) -> list[str]:
    """Bỏ chữ LỚP PHỦ khỏi văn bản OCR trước khi index.

    Bù OCR lên 100% độ phủ lại làm KIS TỆ ĐI (`R@1` 0.444 -> 0.361,
    `top1_pairwise` 0.667 -> 0.481). Nguyên nhân đo được: phần bù gần như toàn
    lớp phủ của đài — `"HTV9 HD"` xuất hiện ở 179/181 keyframe, cộng đồng hồ
    mỗi frame một giá trị, cộng tiêu đề CHỮ CHẠY nói về tin KHÁC với hình đang
    chiếu.

    Chữ chạy là thứ hại nhất: nó bơm nội dung của tin B vào khung hình của tin
    A, tạo khớp sai ở `bm25_ocr` và `ocr_fuzzy`.

    Hai bộ lọc, cùng một nguyên tắc "xuất hiện khắp nơi thì không phân biệt
    được gì":

    - chuỗi có mặt ở hơn `df_threshold` tỉ lệ scene CỦA CÙNG VIDEO -> lớp phủ;
    - chuỗi chỉ là đồng hồ hoặc ngày tháng -> bỏ theo mẫu;
    - `max_words > 0`: bỏ luôn chuỗi dài hơn ngần ấy từ (khung tiêu đề/chữ chạy).

    Về `max_words`: đây là ĐÁNH ĐỔI, không phải cải tiến hiển nhiên. Khung chữ
    dài chứa cả tiêu đề của tin ĐANG chiếu (hữu ích — gold `KIS_E03` hỏi đúng
    nội dung một tiêu đề như thế) lẫn chữ chạy về tin KHÁC (có hại). Không tách
    được hai loại bằng luật, nên phải đo rồi mới chọn. Mặc định 0 = không bỏ.

    Tính DF theo TỪNG VIDEO, không phải toàn corpus: logo đài khác nhau giữa
    các kênh, và một chuỗi phổ biến ở video này có thể là nội dung riêng của
    video kia.
    """

    if field != "ocr":
        return [item.field_text(field) for item in documents]

    # Lọc theo VỊ TRÍ trước — chính xác hơn hẳn mọi heuristic về nội dung.
    # Thống kê bbox thật của 2028 chuỗi cho phân bố lưỡng cực rõ rệt:
    #   y~0.1, x>0.75  762 chuỗi, TB 1.6 từ   -> logo kênh + đồng hồ
    #   y>0.85         963 chuỗi, TB 9.3 từ   -> thanh chữ chạy
    #   phần giữa      323 chuỗi, TB 4.8 từ   -> NỘI DUNG thật
    # Tức 84% OCR là lớp phủ, và chúng nằm đúng hai chỗ cố định trên khung hình.
    per_video: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    for item in documents:
        video = item.video_id
        totals[video] += 1
        bucket = per_video.setdefault(video, Counter())
        bucket.update({" ".join(text.split()) for text in item.ocr_texts})

    out: list[str] = []
    for item in documents:
        counts = per_video.get(item.video_id, Counter())
        total = max(totals.get(item.video_id, 1), 1)
        kept = []
        for text, box in zip(item.ocr_texts, _boxes(item), strict=False):
            cleaned = " ".join(text.split())
            if box is not None and _in_overlay_band(box):
                continue
            if not cleaned or _CLOCK.match(cleaned) or _DATE.match(cleaned):
                continue
            if counts.get(cleaned, 0) / total > df_threshold:
                continue
            if max_words and len(cleaned.split()) > max_words:
                continue
            kept.append(cleaned)
        out.append(" ".join(kept))
    return out


class BM25Index:
    def __init__(
        self,
        documents: list[SceneDocument],
        field: str,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        coverage: CoverageConfig | None = None,
        drop_overlay_df: float | None = None,
        overlay_max_words: int = 0,
    ) -> None:
        self.documents = documents
        self.field = field
        # BM25-01. Mặc định None = BM25 nguyên bản (nhánh A của ablation), nên
        # thêm module coverage KHÔNG tự động đổi hành vi của ai.
        self.coverage = coverage
        self.k1 = k1
        self.b = b
        texts = [item.field_text(field) for item in documents]
        if drop_overlay_df is not None:
            texts = _strip_overlay(documents, field, drop_overlay_df, overlay_max_words)
        self.tokens = [tokenize(text) for text in texts]
        self.lengths = [len(item) for item in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        frequencies: Counter[str] = Counter()
        for tokens in self.tokens:
            frequencies.update(set(tokens))
        count = len(documents)
        self.idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in frequencies.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[SceneDocument, float]]:
        query_tokens = tokenize(query)
        scored: list[tuple[SceneDocument, float]] = []
        for document, tokens, length in zip(
            self.documents, self.tokens, self.lengths, strict=True
        ):
            tf = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = tf[token]
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_length, 1e-9)
                )
                score += self.idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                if self.coverage is not None and not self.coverage.is_noop:
                    # Coverage nhân theo điểm BM25 chứ không cộng hằng số: hai
                    # đại lượng này khác thang đo, cộng thẳng sẽ để coverage
                    # lấn át ở query mà BM25 vốn cho điểm thấp.
                    result = compute_coverage(query, document.field_text(self.field), self.idf)
                    score *= max(0.0, 1.0 + result.adjustment(self.coverage))
                scored.append((document, score))
        scored.sort(key=lambda item: (-item[1], _document_id(item[0])))
        return scored[:limit]


class LexicalRetriever:
    # `branch_id` là danh tính ổn định của adapter; `execution_id` = branch +
    # biến thể query (PR-03). Candidate mang execution_id ở `source` để
    # /capabilities và cấu hình per-branch nói về cùng một chuỗi.
    backend_kind = "lexical"
    supported_controls = ("enabled", "weight", "top_k", "timeout_ms")

    def __init__(self, field: str, index: BM25Index, query_transform=None) -> None:
        self.field = field
        self.index = index
        # Hook biến đổi truy vấn trước khi vào BM25. Chỉ nhánh `keyword` dùng:
        # phía tài liệu là nhãn object 2-3 từ, phía truy vấn là cả câu kể
        # chuyện — hai hạt khác hẳn nhau (xem online/services/keyword_extraction.py).
        self.query_transform = query_transform
        self.branch_id = f"bm25_{field}"
        self.execution_id = f"{self.branch_id}.raw"
        self.name = self.branch_id
        self.modality = Modality(field)

    @classmethod
    async def build(
        cls,
        field: str,
        repository: SceneRepository,
        *,
        query_transform=None,
        drop_overlay_df: float | None = None,
        overlay_max_words: int = 0,
    ) -> "LexicalRetriever":
        index = BM25Index(
            await repository.all(), field,
            drop_overlay_df=drop_overlay_df, overlay_max_words=overlay_max_words,
        )
        return cls(field, index, query_transform)

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        if effective_weight(plan, self.execution_id, self.modality, self.branch_id) <= 0:
            return []
        limit = effective_limit(plan, self.execution_id, limit, self.branch_id)
        query = plan.events[0].text if len(plan.events) == 1 else plan.normalized_query
        if self.query_transform is not None:
            query = self.query_transform(query)
        results = self.index.search(query, limit * 2)
        filtered = []
        for scene, score in results:
            if plan.filters.video_ids and scene.video_id not in plan.filters.video_ids:
                continue
            if plan.filters.scene_ids and scene.scene_id not in plan.filters.scene_ids:
                continue
            if plan.filters.has_ocr is not None and bool(scene.ocr_texts) != plan.filters.has_ocr:
                continue
            if plan.filters.has_asr is not None and bool(scene.asr_texts) != plan.filters.has_asr:
                continue
            if (
                plan.filters.start_sec_gte is not None
                and scene.start_sec < plan.filters.start_sec_gte
            ):
                continue
            if (
                plan.filters.end_sec_lte is not None
                and scene.end_sec > plan.filters.end_sec_lte
            ):
                continue
            text = scene.field_text(self.field)
            filtered.append(
                Candidate(
                    candidate_id=scene.scene_id,
                    entity_type="scene",
                    scene_id=scene.scene_id,
                    video_id=scene.video_id,
                    start_frame=scene.start_frame,
                    end_frame=scene.end_frame_exclusive - 1,
                    source=self.execution_id,
                    modality=self.modality,
                    raw_score=score,
                    score_kind="bm25",
                    rank=len(filtered) + 1,
                    index_id=f"bm25_{self.field}_inmemory",
                    payload={"matched_text": text[:1000]},
                )
            )
            if len(filtered) >= limit:
                break
        return filtered
