"""Tầng hiểu truy vấn bằng LLM FPT — lấp hai chỗ trước đây thuần rule.

Bối cảnh: sau khi enrichment bằng FPT VLM, caption/keyword trong export đã là
**tiếng Việt** (`"language": "vi"`, keyword kiểu "người dẫn chương trình").
Điều đó làm đảo ngược giả định của `online/services/query_expansion.py`, vốn
được viết khi caption còn tiếng Anh: nó dịch VI→EN rồi nối vào query BM25, nên
bây giờ các term tiếng Anh khớp 0 token. Đo được: bật/tắt `AIC_ENABLE_EXPANSION`
cho kết quả y hệt ở 3/4 truy vấn thử.

Nên chỗ rỗng thật sự là HAI chỗ khác nhau, dễ nhầm thành một:

``FptQueryTranslator`` — VI→EN cho **CLIP text tower**
    Vector ảnh sinh bằng `openai/clip-vit-large-patch14`, mà text tower của
    CLIP chỉ được huấn luyện trên tiếng Anh. Đưa thẳng truy vấn tiếng Việt vào
    đó là so một câu model chưa từng học với vector ảnh — vẫn ra số, vẫn xếp
    hạng, nhưng gần như vô nghĩa. Đây là chỗ dịch THẬT SỰ cần.

``FptQueryExpander`` — mở rộng **trong tiếng Việt** cho BM25
    Caption đã là tiếng Việt nên cầu nối cần thiết là đồng nghĩa/cách diễn đạt
    khác *cùng ngôn ngữ* ("cột nước" ~ "vòi nước", "tia nước"), không phải dịch.

Cả hai đều cache theo truy vấn trong vòng đời tiến trình: một buổi eval chạy
lại cùng bộ truy vấn nhiều lần, không cache thì vừa tốn tiền vừa làm kết quả
dao động giữa các lần chạy vì LLM không hoàn toàn tất định.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.errors import DependencyUnavailableError
from online.prompts import EXPAND_QUERY, TRANSLATE_QUERY


class FptQueryTranslator:
    """Dịch truy vấn VI→EN trước khi đưa vào text tower của CLIP.

    Cache HAI TẦNG:

    - trong tiến trình: một truy vấn chỉ dịch một lần cho mọi nhánh/sự kiện;
    - trên đĩa (`cache_dir`): còn giữ qua các lần chạy khác nhau.

    Tầng đĩa quan trọng vì hai lý do đo được. Thứ nhất, mỗi lần chạy eval
    trước đây dịch lại toàn bộ 40 truy vấn — chi phí và thời gian trả đi trả
    lại cho cùng một kết quả tất định. Thứ hai, lệnh gọi dịch hỏng khoảng
    **1/40 lượt**, và mỗi lần hỏng là nhánh dense của truy vấn đó biến mất;
    có cache thì lần chạy sau dùng lại bản dịch đã có thay vì tung xúc xắc lại.

    Bản dịch ở nhiệt độ 0 nên cache không làm mất tính đúng đắn — nó chỉ biến
    một kết quả đáng lẽ tất định thành thật sự tất định.
    """

    spec = TRANSLATE_QUERY

    def __init__(
        self,
        client: FptClient,
        *,
        model_id: str,
        cache_dir: Path | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self._cache: dict[str, str] = {}
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max(1, max_attempts)

    def _disk_path(self, query: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            f"{self.spec.stamp}|{self.model_id}|{query}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.txt"

    def translate(self, query: str) -> str:
        cached = self._cache.get(query)
        if cached is not None:
            return cached
        path = self._disk_path(query)
        if path is not None and path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                self._cache[query] = text
                return text

        messages = [{"role": "user", "content": self.spec.render(query=query)}]
        # Thử lại ngay trong adapter: `FptClient` chỉ retry lỗi transient của
        # HTTP, còn "trả về chuỗi rỗng" thì nó coi là thành công. Mất bản dịch
        # đồng nghĩa mất cả nhánh dense của truy vấn đó, nên đáng thử thêm.
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                text = self.client.chat_completion(
                    messages,
                    model=self.model_id,
                    temperature=self.spec.temperature,
                    max_tokens=self.spec.max_tokens,
                ).text.strip().strip('"')
            except ProviderError as exc:
                last_error = exc
                continue
            if text:
                self._cache[query] = text
                if path is not None:
                    path.write_text(text, encoding="utf-8")
                return text
        raise DependencyUnavailableError(
            f"dịch truy vấn hỏng sau {self.max_attempts} lần: {last_error or 'LLM trả chuỗi rỗng'}"
        )


class TranslatingTextEncoder:
    """Bọc một `TextEncoder`, dịch truy vấn trước khi encode.

    Chỉ bọc encoder của nhánh dense **visual**. Không bọc nhánh nào khác: BM25
    chạy trên caption tiếng Việt nên nhận bản dịch tiếng Anh sẽ khớp 0 token —
    đúng lỗi mà module này sinh ra để sửa, chỉ là theo chiều ngược lại.

    Dịch hỏng thì để lỗi nổi lên chứ KHÔNG lặng lẽ encode bản tiếng Việt:
    query tiếng Việt qua text tower tiếng Anh chính là trạng thái hỏng đang
    cần loại bỏ, nên "fallback" ở đây chỉ là tái lập lỗi cũ mà không ai thấy.
    Nhánh dense sẽ báo `failed` kèm warning, các nhánh khác vẫn chạy.
    """

    def __init__(self, inner, translator: FptQueryTranslator) -> None:
        self.inner = inner
        self.translator = translator

    def warmup(self) -> None:
        # Giữ nguyên hợp đồng warmup của encoder gốc (online/api/container.py
        # dò bằng hasattr) — nếu mất, lỗi cold-start ở FIX-DETERMINISM-01 quay lại.
        warmup = getattr(self.inner, "warmup", None)
        if callable(warmup):
            warmup()

    async def encode(self, text: str) -> list[float]:
        try:
            translated = await asyncio.to_thread(self.translator.translate, text)
        except ProviderError as exc:
            raise DependencyUnavailableError(f"dịch truy vấn VI→EN hỏng: {exc}") from exc
        return await self.inner.encode(translated)


class FptQueryExpander:
    """Sinh term đồng nghĩa TIẾNG VIỆT để nối vào query BM25.

    Trả chuỗi rỗng khi hỏng, KHÔNG ném: khác với dịch cho CLIP, BM25 không có
    bản dịch vẫn hoạt động đúng hoàn toàn (đó là hành vi mặc định lâu nay).
    Giết cả nhánh BM25 vì một lần gọi LLM lỗi là đánh đổi tệ.
    """

    spec = EXPAND_QUERY

    def __init__(self, client: FptClient, *, model_id: str, max_terms: int = 3) -> None:
        self.client = client
        self.model_id = model_id
        # v2 hạ trần từ 6 xuống 3: đo được v1 gây query drift, làm giảm KIS
        # MRR và AVS nDCG (xem notes của EXPAND_QUERY trong registry).
        self.max_terms = max_terms
        self._cache: dict[str, str] = {}

    def expand(self, query: str) -> str:
        cached = self._cache.get(query)
        if cached is not None:
            return cached
        messages = [
            {"role": "user", "content": self.spec.render(query=query, max_terms=self.max_terms)}
        ]
        try:
            text = self.client.chat_completion(
                messages,
                model=self.model_id,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
            ).text
            terms = _parse_term_list(text)[: self.max_terms]
        except (ProviderError, ValueError):
            terms = []
        expanded = " ".join(terms)
        self._cache[query] = expanded
        return expanded


def _parse_term_list(text: str) -> list[str]:
    """Bóc mảng JSON các chuỗi khỏi câu trả lời LLM.

    Model hay bọc trong ```json ... ``` hoặc thêm câu dẫn, nên cắt theo cặp
    ngoặc vuông ngoài cùng thay vì `json.loads` thẳng.
    """

    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item.strip() for item in payload if isinstance(item, str) and item.strip()]


__all__ = [
    "FptQueryExpander",
    "FptQueryTranslator",
    "TranslatingTextEncoder",
]
