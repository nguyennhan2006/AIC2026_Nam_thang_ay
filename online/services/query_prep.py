"""Rule-based query preparation: target / ocr / context split (Phương án F).

Query KIS thường dài và kể chuyện theo trình tự; đưa nguyên câu vào một
encoder duy nhất sẽ bị "loãng". Module này tách query thành 3 phần bằng
rule đơn giản (không LLM, deterministic, không query drift):

    {
      "target_query":  phần mô tả moment cần tìm (mặc định: sau "cuối cùng"
                       hoặc marker temporal cuối cùng; nếu không có thì cả câu),
      "ocr_query":     các cụm trong dấu ngoặc kép "..." — chữ trên màn hình,
      "context_query": các mệnh đề đứng trước target (bối cảnh, dùng để
                       giữ recall ở nhánh lexical),
    }

Ví dụ:
    'Người cào muối, sau đó đoàn người vẫy tay, cuối cùng trước căn nhà
     có chữ "Gừng cay muối mặn xin đừng quên nhau"'
    -> target_query  = 'trước căn nhà có chữ'
       ocr_query     = 'Gừng cay muối mặn xin đừng quên nhau'
       context_query = 'Người cào muối, sau đó đoàn người vẫy tay'

Cách sử dụng
------------
1. Dùng trực tiếp hàm ``prepare_query`` khi cần tách phần:

       from online.services.query_prep import prepare_query
       parts = prepare_query('... có chữ "xin đừng quên nhau"')
       parts.target_query, parts.ocr_query, parts.context_query

2. Dùng ``PreparedQueryPlanner`` như một drop-in planner cho SearchService —
   giữ nguyên ``RuleBasedQueryPlanner`` bên trong rồi định tuyến lại:

       from online.services.query_prep import PreparedQueryPlanner
       service = SearchService(repository, retrievers,
                               planner=PreparedQueryPlanner())

   Routing sau khi wrap (chỉ áp dụng cho KIS/AVS, không đụng SEQUENCE):
   - ``plan.normalized_query``  <- target_query (+ ocr_query)
     -> nhánh dense (DenseRetriever đọc normalized_query) nhận query gọn,
        đúng moment đích thay vì cả đoạn văn.
   - ``plan.events[0].text``    <- giữ nguyên full query
     -> nhánh BM25 (LexicalRetriever đọc events[0].text khi có 1 event)
        vẫn thấy toàn bộ từ khóa của context, không mất recall lexical.
   - ``plan.events[0].exact_phrases`` <- cụm trong ngoặc kép
     -> nhánh ``OcrFuzzyRetriever`` (online/adapters/ocr_fuzzy.py) đọc
        exact_phrases làm ocr_query.

Đây là bản rule đơn giản theo đúng tinh thần Phương án F: chưa cần LLM
parser (Phương án J); khi nào rule không đủ, thay planner khác qua cùng
interface mà không sửa SearchService.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from online.domain.models import QueryPlan, SearchRequest, TaskType
from online.services.query_planner import (
    QUOTED_RE,
    RuleBasedQueryPlanner,
    normalize_query,
)


# Marker temporal: phần đứng SAU marker cuối cùng được coi là target.
_TEMPORAL_MARKER_RE = re.compile(
    r"\b(?:sau đó|tiếp theo|kế tiếp|cuối cùng|rồi(?: thì)?|then|next|finally)\b",
    flags=re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class QueryParts:
    """Kết quả tách query; mọi field có thể rỗng trừ target_query."""

    target_query: str
    ocr_query: str = ""
    context_query: str = ""
    exact_phrases: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip(" ,.;:")


def prepare_query(query: str) -> QueryParts:
    """Tách query thành target/ocr/context bằng rule; không gọi mô hình nào."""

    normalized = normalize_query(query)

    # 1. Cụm trong ngoặc kép -> ocr_query, và loại khỏi phần visual để
    #    target_query không bị chuỗi OCR dài làm loãng.
    exact_phrases = [item.strip() for item in QUOTED_RE.findall(normalized)]
    without_quotes = _clean(QUOTED_RE.sub(" ", normalized))

    # 2. Phần sau marker temporal cuối cùng -> target; phần trước -> context.
    pieces = [_clean(item) for item in _TEMPORAL_MARKER_RE.split(without_quotes)]
    pieces = [item for item in pieces if item]
    if len(pieces) >= 2:
        target = pieces[-1]
        context = " ".join(pieces[:-1])
    else:
        target = pieces[0] if pieces else _clean(without_quotes)
        context = ""

    # 3. Query chỉ toàn ngoặc kép (không còn phần visual): target rơi về
    #    chính nội dung ngoặc kép để nhánh dense vẫn có gì đó để encode.
    if not target:
        target = " ".join(exact_phrases) or normalized

    return QueryParts(
        target_query=target,
        ocr_query=" ".join(exact_phrases),
        context_query=context,
        exact_phrases=exact_phrases,
    )


class PreparedQueryPlanner:
    """Drop-in planner: chạy planner gốc rồi định tuyến lại theo QueryParts.

    Giữ nguyên hành vi cho TRAKE nhiều bước (đã có tách event riêng trong
    planner gốc); chỉ can thiệp TEXTUAL_KIS/AVS/QA một-event.
    """

    def __init__(
        self,
        inner: RuleBasedQueryPlanner | None = None,
        *,
        include_ocr_in_dense: bool = True,
    ) -> None:
        self.inner = inner or RuleBasedQueryPlanner()
        # include_ocr_in_dense: nối ocr_query vào query dense — chữ trên ảnh
        # đôi khi cũng là tín hiệu visual (biển hiệu lớn). Tắt đi nếu ablation
        # cho thấy nhiễu.
        self.include_ocr_in_dense = include_ocr_in_dense

    async def plan(self, request: SearchRequest) -> QueryPlan:
        plan = await self.inner.plan(request)
        if plan.task == TaskType.TRAKE and len(plan.events) >= 2:
            return plan

        # QA queries bị phá nghiêm trọng bởi temporal splitting:
        # "Con số hiển thị cuối cùng trên cân là bao nhiêu?" -> "trên cân là bao nhiêu?"
        # Visual embedding nhận gần như empty query và recall sụp đổ.
        # Giữ nguyên full query cho QA để retrieval có đủ ngữ cảnh thị giác.
        if plan.task == TaskType.QA:
            return plan

        parts = prepare_query(plan.normalized_query)
        dense_query = parts.target_query
        if self.include_ocr_in_dense and parts.ocr_query:
            dense_query = f"{parts.target_query} {parts.ocr_query}".strip()
        if dense_query == plan.normalized_query:
            return plan

        # events giữ full query cho BM25; chỉ dense đọc normalized_query mới.
        events = [
            event.model_copy(
                update={
                    "exact_phrases": sorted(
                        set(event.exact_phrases) | set(parts.exact_phrases)
                    )
                }
            )
            for event in plan.events
        ]
        return plan.model_copy(
            update={"normalized_query": dense_query, "events": events}
        )
