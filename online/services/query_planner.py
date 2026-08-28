"""Deterministic Vietnamese/English query normalization and event planning."""

from __future__ import annotations

import re
import unicodedata

from online.domain.models import (
    Modality,
    QueryEvent,
    QueryPlan,
    SearchRequest,
    TaskType,
)
from online.domain.search_config import BranchRuntimeOptions, SearchOptions
from online.errors import InvalidQueryError


QUOTED_RE = re.compile(r'["“”]([^"“”]{2,})["“”]')
SPACE_RE = re.compile(r"\s+")
# Marker MẠNH: luôn là chuyển cảnh, tách ở đâu cũng đúng.
TEMPORAL_RE = re.compile(
    r"\b(?:sau đó|tiếp theo|kế tiếp|rồi thì|then|next)\b",
    flags=re.IGNORECASE,
)

# Marker YẾU: "cuối cùng"/"finally" chỉ là chuyển cảnh khi ĐỨNG ĐẦU MỆNH ĐỀ.
#
#     "Cuối cùng, người đàn ông đi vào nhà."   -> marker thời gian
#     "Con số cuối cùng trên cân là bao nhiêu?" -> ĐỊNH NGỮ của "con số"
#
# Trước đây chúng nằm chung TEMPORAL_RE nên câu hỏi bị cắt thành một "event"
# không tồn tại. Đo được trên truy vấn cá/cân: TRAKE tách 3 event, event thứ 3
# là "trên cân là bao nhiêu?" — không phải cảnh nào cả. Chuỗi đòi khớp đủ 3
# theo thứ tự nên không bao giờ dựng được, và TRAKE miss dù hai cảnh thật đều
# có trong video gold.
#
# Cùng một lỗi, cùng một cách sửa như `online/services/query/normalize.py`;
# đường này bị bỏ sót vì TRAKE dùng planner riêng, không đi qua QueryRouter.
WEAK_TEMPORAL_RE = re.compile(
    r"(?<=[,;:.])\s*(?:cuối cùng|finally)\s*,?\s*",
    flags=re.IGNORECASE,
)
# Gold TRAKE query dùng format liệt kê đánh số "(1) ...; (2) ...; (3) ..."
# (xem examples/AIC2026_L21_V001_queries_4tasks.jsonl) — KHÔNG dùng từ nối
# tiếp diễn kiểu "sau đó"/"cuối cùng" mà TEMPORAL_RE bắt. Không có nhánh này,
# mọi query TRAKE thật rơi về đúng 1 event, khiến `len(plan.events) >= 2` ở
# search.py luôn False và TrakeProcessor không bao giờ chạy.
NUMBERED_STEP_RE = re.compile(r"\(\d+\)\s*")
# ROUTE-01: cue quyết định một modality có ĐƯỢC CHẠY hay không, nên phải phủ
# đủ cách hỏi thường gặp. Thiếu cue => branch không chạy (không phải chạy rồi
# nhân 0), nên bỏ sót cue ở đây là mất recall thật.
SPEECH_HINTS = {
    "nói", "phát biểu", "trình bày", "nghe thấy", "lời thoại", "giọng",
    "hội thoại", "trả lời", "phỏng vấn", "âm thanh", "tiếng",
    "says", "speaks", "speech", "heard", "said", "interview", "audio", "voice",
}
TEXT_HINTS = {
    "chữ", "dòng chữ", "biển", "bảng", "khẩu hiệu", "biển hiệu", "biển báo",
    "nhãn", "logo", "tiêu đề", "phụ đề", "màn hình", "ghi", "viết", "đọc được",
    "text", "sign", "caption", "subtitle", "label", "banner", "written", "screen",
}


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).strip()
    normalized = SPACE_RE.sub(" ", normalized)
    if not normalized:
        raise InvalidQueryError("query is empty after normalization")
    return normalized


def has_text_cue(text: str, exact_phrases: list[str]) -> bool:
    """Query có yêu cầu đọc chữ trong hình không (OCR)."""

    return bool(exact_phrases) or any(hint in text.casefold() for hint in TEXT_HINTS)


def has_speech_cue(text: str) -> bool:
    """Query có yêu cầu nội dung lời nói không (ASR)."""

    return any(hint in text.casefold() for hint in SPEECH_HINTS)


def compute_modality_weights(
    text: str, exact_phrases: list[str], *, allow_zero: bool = True
) -> dict[Modality, float]:
    """Suy modality weight từ MỘT đoạn text — dùng cho cả full query lẫn
    từng event riêng của TRAKE (PR-14A: trước đây mọi step TRAKE dùng chung
    weight của cả câu, nên step không có OCR/ASR vẫn bị đẩy nhánh sai).

    `allow_zero=True` (mặc định, ROUTE-01): query không có cue chữ/lời nói thì
    OCR/ASR nhận đúng 0 và branch KHÔNG chạy — `effective_weight() <= 0` đã
    được adapter kiểm ngay đầu `search()`. Trước đây hai modality này có sàn
    0.35/0.25 nên luôn góp candidate, tạo false positive kiểu query "cột nước
    phun lên từ lòng đất" khớp một bản tin cháy rừng qua OCR "Sông Hồng ... lũ
    lớn". `allow_zero=False` giữ nguyên sàn cũ, chỉ để chạy nhánh A của
    ablation ROUTE-01.
    """

    lowered = text.casefold()
    weights = {
        Modality.VISUAL: 1.0,
        Modality.CAPTION: 1.0,
        Modality.OCR: 0.0 if allow_zero else 0.35,
        Modality.ASR: 0.0 if allow_zero else 0.25,
        Modality.KEYWORD: 0.65,
        # Buckets mới (W3) chỉ có hiệu lực khi container thực sự đăng ký
        # retriever tương ứng (mặc định tắt, xem AIC_ENABLE_* ở online/config.py)
        # — giá trị ở đây chỉ là default hợp lý cho lúc retriever được bật.
        Modality.OBJECT: 0.5,
        Modality.ACTION: 0.5,
        Modality.COLOR: 0.4,
        Modality.EVENT: 0.3,
    }
    if has_text_cue(lowered, exact_phrases):
        weights[Modality.OCR] = 2.0
    if has_speech_cue(lowered):
        weights[Modality.ASR] = 1.7
    return weights


def _join_constraints(*groups: list[str]) -> str:
    return " ".join(
        item.strip()
        for group in groups
        for item in group
        if item.strip()
    ).strip()


def _structured_kis_queries(request: SearchRequest) -> tuple[dict[str, str], dict[Modality, str]]:
    constraints = request.kis_constraints
    if (request.task or TaskType.TEXTUAL_KIS) != TaskType.TEXTUAL_KIS or constraints is None:
        return {}, {}

    visual = constraints.visual
    ocr = constraints.ocr
    asr = constraints.asr
    must = constraints.must
    visual_must = _join_constraints(visual, must)
    caption = _join_constraints(visual, must, asr)
    ocr_query = _join_constraints(ocr)
    asr_query = _join_constraints(asr)
    event = _join_constraints(visual, must, asr)

    branch_queries = {
        "dense_visual": visual_must,
        "lexical_hash_fallback": visual_must,
        "caption_dense": caption,
        "bm25_caption": caption,
        "bm25_keyword": visual_must,
        "bm25_object": visual_must,
        "bm25_action": visual_must,
        "color_search": visual_must,
        "bm25_ocr": ocr_query,
        "ocr_fuzzy": ocr_query,
        "bm25_asr": asr_query,
        "event_search": event,
    }
    modality_queries = {
        Modality.VISUAL: visual_must,
        Modality.CAPTION: caption,
        Modality.KEYWORD: visual_must,
        Modality.OBJECT: visual_must,
        Modality.ACTION: visual_must,
        Modality.COLOR: visual_must,
        Modality.OCR: ocr_query,
        Modality.ASR: asr_query,
        Modality.EVENT: event,
    }
    return branch_queries, modality_queries


class RuleBasedQueryPlanner:
    """Safe V1 planner; an LLM planner can replace it through the same output model."""

    def __init__(self, *, allow_zero_modality: bool = False) -> None:
        """`allow_zero_modality=True` bật zero-gating của ROUTE-01.

        MẶC ĐỊNH TẮT vì đã đo và thua trên corpus hiện tại (xem
        docs/20_EXPERIMENT_LOG.md § ROUTE-01). Lý do: OCR ở bộ dữ liệu này là
        lower-third bản tin MÔ TẢ CHÍNH CẢNH ĐÓ — 11/12 gold KIS có OCR trùng
        từ khoá query — nên nó là tín hiệu ngữ nghĩa, không phải chữ ngẫu
        nhiên. Tắt nó đi mất nhiều hơn được.

        Cơ chế vẫn giữ nguyên vì tiền đề "query không nhắc chữ ⇒ OCR không
        liên quan" đúng với nhiều corpus khác (phim, video đời thường); bật
        cờ này là đủ, không phải viết lại.
        """

        self.allow_zero_modality = allow_zero_modality

    # Nhánh khớp GẦN-NGUYÊN-CHUỖI: kết quả của chúng chỉ xuất hiện khi có một
    # chuỗi trùng khớp thật, nên chúng KHÔNG thể tạo ra false positive kiểu
    # "trùng vài token phổ biến" mà ROUTE-01 nhắm tới. Tắt chúng theo modality
    # OCR chỉ làm mất recall của truy vấn gõ thẳng chữ nhìn thấy trên màn hình
    # (vd "hen ngay gap lai") — loại truy vấn không hề chứa cue "chữ"/"biển".
    EXACT_MATCH_BRANCHES = {"ocr_fuzzy": 0.6}

    def _exempt_exact_match_branches(
        self, options: "SearchOptions", weights: dict[Modality, float]
    ) -> "SearchOptions":
        """Giữ nhánh khớp nguyên chuỗi sống khi modality của nó bị route về 0.

        KHÔNG ghi đè cấu hình người dùng đã đặt tay: chỉ thêm override khi
        branch đó chưa có mục nào trong `search_options.branches`.
        """

        if weights.get(Modality.OCR, 0.0) > 0.0:
            return options
        missing = {
            branch: weight
            for branch, weight in self.EXACT_MATCH_BRANCHES.items()
            if branch not in options.branches
        }
        if not missing:
            return options
        branches = dict(options.branches)
        for branch, weight in missing.items():
            branches[branch] = BranchRuntimeOptions(enabled=True, weight=weight)
        return options.model_copy(update={"branches": branches})

    async def plan(self, request: SearchRequest) -> QueryPlan:
        task = request.task or TaskType.TEXTUAL_KIS
        normalized = normalize_query(request.query)
        exact_phrases = [item.strip() for item in QUOTED_RE.findall(normalized)]
        parts = [normalized]
        if task == TaskType.TRAKE:
            # Ưu tiên format đánh số "(1)...(2)..." — đúng format gold thật.
            # `[1:]` bỏ phần dẫn trước "(1)" (vd "...căn chỉnh bốn khoảnh khắc:").
            numbered = [
                item.strip(" ,.;:") for item in NUMBERED_STEP_RE.split(normalized)[1:]
            ]
            numbered = [item for item in numbered if item]
            if len(numbered) >= 2:
                parts = numbered
            else:
                # Marker yếu tách TRƯỚC (chỉ khớp khi đứng đầu mệnh đề), rồi
                # marker mạnh tách tiếp từng phần. Làm ngược lại thì "cuối
                # cùng" ở giữa câu hỏi vẫn bị cắt.
                pieces = WEAK_TEMPORAL_RE.split(normalized)
                temporal = [
                    part.strip(" ,.;:")
                    for piece in pieces
                    for part in TEMPORAL_RE.split(piece)
                ]
                temporal = [item for item in temporal if item]
                if len(temporal) >= 2:
                    parts = temporal
        events = [
            QueryEvent(
                event_idx=index,
                text=text,
                exact_phrases=[phrase for phrase in exact_phrases if phrase in text],
            )
            for index, text in enumerate(parts)
        ]
        weights = compute_modality_weights(
            normalized, exact_phrases, allow_zero=self.allow_zero_modality
        )
        search_options = request.search_options or SearchOptions()
        search_options = self._exempt_exact_match_branches(search_options, weights)
        branch_queries, modality_queries = _structured_kis_queries(request)
        return QueryPlan(
            task=task,
            original_query=request.query,
            normalized_query=normalized,
            events=events,
            modality_weights=weights,
            filters=request.filters,
            search_options=search_options,
            branch_queries=branch_queries,
            modality_queries=modality_queries,
        )
