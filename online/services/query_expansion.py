"""VI→EN sparse keyword expansion cho nhánh lexical (Phương án K / G-lite).

Metadata caption/object trong pipeline này phần lớn là tiếng Anh (ví dụ
scene caption "Workers rake salt into piles in a salt field") trong khi
query thi đấu là tiếng Việt — BM25 sẽ miss hoàn toàn nếu không có cầu nối
từ vựng. Module này mở rộng query bằng một lexicon VI→EN *được kiểm soát*
(curated, match cụm dài trước, có trần số term thêm vào) để tránh query
drift — đúng tinh thần "expansion có kiểm soát" của Phương án K.

Đây KHÔNG phải dịch máy (Phương án G đầy đủ). Nếu sau này muốn thử G, cắm
một translator ngoài vào tham số ``translate`` của ``expand_query`` rồi đo
ablation vector_vi vs vector_en vs vector_vi+en bằng scripts/eval_kis.py.

Cách sử dụng
------------
1. Mở rộng chuỗi query trực tiếp:

       from online.services.query_expansion import expand_query
       expanded = expand_query("đoàn người trước căn nhà")
       # -> "đoàn người trước căn nhà group of people crowd house building"

2. Wrap một lexical retriever sẵn có (khuyến nghị — chỉ nhánh BM25 thấy
   query mở rộng, nhánh dense/OCR giữ nguyên):

       from online.adapters.bm25 import LexicalRetriever
       from online.services.query_expansion import QueryExpansionRetriever

       caption = await LexicalRetriever.build("caption", repository)
       retrievers = [dense, QueryExpansionRetriever(caption), ocr, ...]

   Wrapper giữ nguyên ``modality`` của retriever gốc nên weighted RRF không
   cần đổi gì.

3. Eval nhanh: ``python -m scripts.eval_kis --use-expansion``.

Mở rộng lexicon: thêm entry vào ``DEFAULT_LEXICON`` (key là cụm tiếng Việt
KHÔNG dấu — match chạy trên văn bản đã bỏ dấu nên "căn nhà" và "can nha"
đều trúng key "can nha"). Giữ mỗi key <= 3 term tiếng Anh sát nghĩa; đừng
nhét synonym xa nghĩa — false positive của BM25 đến từ đây.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from online.adapters.ocr_fuzzy import normalize_vi
from online.domain.models import Candidate, QueryPlan
from online.ports.interfaces import Retriever


# Lexicon hạt giống cho domain tin tức/đời sống của AIC — mở rộng dần khi
# phân tích lỗi trên dev set. Key: tiếng Việt đã bỏ dấu, lowercase.
#
# CẢNH BÁO khi thêm key: match chạy trên văn bản ĐÃ BỎ DẤU nên từ đơn một
# âm tiết rất dễ nhập nhằng ("chợ"/"chó"→"cho", "có"/"cờ"/"cỏ"→"co",
# "mưa"/"mua"/"múa"→"mua") và sẽ mở rộng sai cho gần như mọi query.
# Quy tắc: chỉ dùng key >= 2 âm tiết, hoặc từ đơn không có đồng âm sau khi
# bỏ dấu (vd. "thuyen"). Với con vật/danh từ ngắn, dùng dạng "con X".
DEFAULT_LEXICON: dict[str, list[str]] = {
    "doan nguoi": ["group of people", "crowd"],
    "dam dong": ["crowd", "large group of people"],
    "nguoi dan": ["people", "residents"],
    "can nha": ["house", "building"],
    "ngoi nha": ["house", "building"],
    "toa nha": ["building", "tower"],
    "duong pho": ["street", "road"],
    "xe may": ["motorbike", "scooter"],
    "o to": ["car", "automobile"],
    "xe tai": ["truck"],
    "xe dap": ["bicycle"],
    "thuyen": ["boat"],
    "tau thuy": ["ship", "boat"],
    "cho phien": ["market"],
    "khu cho": ["market"],
    "bien hieu": ["sign", "signboard"],
    "bang chu": ["sign", "text banner"],
    "khau hieu": ["slogan", "banner"],
    "la co": ["flag"],
    "canh dong": ["field"],
    "canh dong muoi": ["salt field", "salt flat"],
    "cao muoi": ["raking salt", "salt harvesting"],
    "bai bien": ["beach", "seashore"],
    "dong song": ["river"],
    "ngon nui": ["mountain"],
    "khu rung": ["forest"],
    "tre em": ["children", "kids"],
    "hoc sinh": ["students", "pupils"],
    "phu nu": ["woman", "women"],
    "dan ong": ["man", "men"],
    "nguoi gia": ["elderly person", "old man", "old woman"],
    "ao dai": ["ao dai", "traditional dress"],
    "non la": ["conical hat"],
    "vay tay": ["waving", "wave hands"],
    "nhay mua": ["dancing"],
    "ca hat": ["singing"],
    "phat bieu": ["speech", "speaking", "presenter"],
    "phong van": ["interview"],
    "trinh bay": ["presenting", "presentation"],
    "truong quay": ["studio", "television studio"],
    "dan chuong trinh": ["presenter", "host", "anchor"],
    "ban tin": ["news", "newscast"],
    "le hoi": ["festival"],
    "dam chay": ["fire", "burning"],
    "lu lut": ["flood", "flooding"],
    "troi mua": ["rain", "raining"],
    "ban dem": ["night", "nighttime"],
    "phao hoa": ["fireworks"],
    "san van dong": ["stadium"],
    "benh vien": ["hospital"],
    "truong hoc": ["school"],
    "ngoi chua": ["pagoda", "temple"],
    "nha tho": ["church"],
    "cay cau": ["bridge"],
    "may bay": ["airplane", "aircraft"],
    "bong da": ["football", "soccer"],
    "nau an": ["cooking"],
    "con cho": ["dog"],
    "con meo": ["cat"],
    "con trau": ["buffalo"],
    "con bo": ["cow"],
    "con ga": ["chicken"],
}


def expand_terms(
    query: str,
    lexicon: dict[str, list[str]] | None = None,
    *,
    max_terms: int = 6,
) -> list[str]:
    """Trả về danh sách term tiếng Anh cần nối thêm vào query.

    Match cụm dài trước (greedy theo số từ của key giảm dần) và đánh dấu
    vùng đã dùng để "cánh đồng muối" không kích hoạt thêm key "cánh đồng".
    ``max_terms`` chặn drift: quá nhiều term mới sẽ pha loãng BM25.
    """

    lexicon = lexicon if lexicon is not None else DEFAULT_LEXICON
    words = normalize_vi(query).split()
    if not words:
        return []
    consumed = [False] * len(words)
    terms: list[str] = []
    for key in sorted(lexicon, key=lambda item: -len(item.split())):
        key_words = key.split()
        span = len(key_words)
        for start in range(len(words) - span + 1):
            if any(consumed[start : start + span]):
                continue
            if words[start : start + span] == key_words:
                for offset in range(span):
                    consumed[start + offset] = True
                for term in lexicon[key]:
                    if term not in terms:
                        terms.append(term)
                break
        if len(terms) >= max_terms:
            break
    return terms[:max_terms]


def expand_query(
    query: str,
    lexicon: dict[str, list[str]] | None = None,
    *,
    max_terms: int = 6,
    translate: Callable[[str], str] | None = None,
) -> str:
    """Query gốc + các term mở rộng (query gốc luôn giữ nguyên phía trước).

    ``translate``: hook tùy chọn cho Phương án G — truyền một hàm dịch
    (vd. gọi API dịch) thì bản dịch cũng được nối vào sau expansion terms.
    Mặc định None = không phụ thuộc dịch vụ ngoài.
    """

    parts = [query]
    parts.extend(expand_terms(query, lexicon, max_terms=max_terms))
    if translate is not None:
        translated = translate(query).strip()
        if translated and normalize_vi(translated) != normalize_vi(query):
            parts.append(translated)
    return " ".join(parts)


class QueryExpansionRetriever:
    """Wrapper Retriever: mở rộng query trước khi ủy quyền cho retriever gốc.

    Chỉ nên wrap các retriever lexical (BM25 caption/keyword). Không wrap
    nhánh dense (encoder đa ngữ tự xử lý) và không wrap OCR (chuỗi OCR cần
    giữ nguyên văn).
    """

    def __init__(
        self,
        inner: Retriever,
        lexicon: dict[str, list[str]] | None = None,
        *,
        max_terms: int = 6,
        expander=None,
    ) -> None:
        self.inner = inner
        self.lexicon = lexicon
        self.max_terms = max_terms
        # `expander` (tùy chọn): object có `.expand(str) -> str` trả thêm term.
        # Có nó thì lexicon VI→EN cứng ở dưới gần như vô dụng — caption trong
        # export hiện là TIẾNG VIỆT nên term tiếng Anh khớp 0 token. Xem
        # online/adapters/fpt_query.py.
        self.expander = expander
        # Wrapper KHÔNG đổi branch (vẫn cùng adapter/index), chỉ đổi biến thể
        # query. Trước PR-03 nó đặt name="bm25_caption_expanded" trong khi
        # candidate bên trong vẫn mang source="bm25_caption", nên cấu hình cho
        # id mà /capabilities công bố hoàn toàn không có tác dụng.
        self.branch_id = getattr(inner, "branch_id", None) or inner.name
        self.execution_id = f"{self.branch_id}.expanded"
        self.name = self.branch_id
        self.query_variant = "expanded"
        # Giữ modality của retriever gốc để weighted RRF tính đúng trọng số.
        self.modality = getattr(inner, "modality", None)
        self.backend_kind = getattr(inner, "backend_kind", "lexical")
        self.supported_controls = getattr(inner, "supported_controls", ())

    async def _extra_terms(self, texts: list[str]) -> dict[str, str]:
        """Term thêm từ LLM cho từng đoạn văn bản, gọi NGOÀI event loop.

        Gom các đoạn trùng nhau trước khi gọi: `normalized_query` và
        `events[0].text` thường giống hệt nhau ở truy vấn một sự kiện, gọi hai
        lần là trả tiền hai lần cho cùng một câu.
        """

        if self.expander is None:
            return {}
        unique = list(dict.fromkeys(text for text in texts if text))
        if not unique:
            return {}
        results = await asyncio.gather(
            *(asyncio.to_thread(self.expander.expand, text) for text in unique)
        )
        return dict(zip(unique, results, strict=True))

    async def search(self, plan: QueryPlan, *, limit: int) -> list[Candidate]:
        extra = await self._extra_terms(
            [plan.normalized_query, *(event.text for event in plan.events)]
        )

        def expand(text: str) -> str:
            base = expand_query(text, self.lexicon, max_terms=self.max_terms)
            terms = extra.get(text, "")
            return f"{base} {terms}".strip() if terms else base

        expanded_events = [
            event.model_copy(update={"text": expand(event.text)}) for event in plan.events
        ]
        expanded_plan = plan.model_copy(
            update={
                "normalized_query": expand(plan.normalized_query),
                "events": expanded_events,
            }
        )
        candidates = await self.inner.search(expanded_plan, limit=limit)
        # Đóng dấu lại `source` bằng execution_id của wrapper: đây là kết quả
        # của biến thể expanded, và fusion phải tính nó theo đúng cấu hình đó.
        return [item.model_copy(update={"source": self.execution_id}) for item in candidates]
