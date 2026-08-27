"""Soạn `SearchQueryBundle` bằng LLM — dữ liệu riêng cho từng search engine.

Mỗi engine mạnh ở một loại dữ liệu khác nhau, nên đưa cùng một chuỗi cho cả
bốn engine là bỏ phí thế mạnh của từng cái:

    CLIP        so ẢNH với CÂU  -> cần MỘT CÂU MÔ TẢ khung hình trông ra sao
    BM25 caption khớp TỪ         -> cần danh từ/động từ cụ thể + cách gọi khác
    OCR         đọc CHỮ trên hình-> cần chuỗi ký tự thật sự hiện trên màn hình
    ASR         tìm trong LỜI NÓI-> cần câu như người ta sẽ NÓI RA

Tầng rule (`online/services/query/`) làm được phép TRỪ: cắt phần hỏi trừu
tượng khỏi câu. Nó không làm được phép VIẾT LẠI, và đo được đó mới là chỗ mất
điểm lớn nhất (gold L21_V023 frame 25995, cùng một frame đích):

    "Bàn tay đeo đồng hồ đổ chất lỏng vào bát trắng đặt trên cân điện tử"
        -> rank 1
    "một con cá được đặt lên cân, sau đó ... Con số hiển thị cuối cùng trên cân"
        -> rank 35

Cả hai mô tả đúng một khung hình. Cái đầu là MÔ TẢ ẢNH, cái sau là KỂ CHUYỆN
kèm câu hỏi. Viết lại câu sau thành câu đầu là việc chỉ LLM làm được.

Nguyên tắc an toàn: **LLM chỉ ĐỀ XUẤT, rule là nền**. Trường nào LLM bỏ trống,
trả rác, hoặc gọi lỗi/timeout thì giữ nguyên giá trị rule đã tính. Truy vấn
không bao giờ mất vì một lệnh gọi mạng — cùng triết lý với `FptQueryExpander`,
ngược với `FptQueryTranslator` (chỗ đó không có bản dịch thì nhánh dense vô
nghĩa nên phải ném).
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.prompts import PREPARE_QUERY_BUNDLE
from online.services.query.models import AnswerType, SearchQueryBundle

logger = logging.getLogger(__name__)

# Trường văn bản: LLM chỉ được ghi đè khi nó trả về nội dung DÀI HƠN ngưỡng
# này. Một hai chữ gần như luôn là dấu hiệu model hiểu sai đề, và ghi đè bằng
# nó sẽ tái lập đúng lỗi "visual query còn hai chữ" mà cả module này sinh ra
# để tránh.
_MIN_TEXT_CHARS = 8


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip().strip('"')


def _clean_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [_clean_text(item) for item in value]
    return [item for item in dict.fromkeys(items) if item][:limit]


def _extract_json_object(text: str) -> dict:
    """Lấy object JSON đầu tiên trong câu trả lời.

    Model `fast` hay bọc JSON trong ```json fence hoặc thêm một câu dẫn, nên
    parse thẳng `json.loads` sẽ hỏng ở những lượt hoàn toàn dùng được.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("không tìm thấy object JSON trong câu trả lời")
    return json.loads(text[start : end + 1])


class FptQueryBundlePreparer:
    """Nhờ LLM soạn lại bundle; giữ giá trị rule ở mọi trường LLM không cải thiện."""

    spec = PREPARE_QUERY_BUNDLE

    def __init__(
        self,
        client: FptClient,
        *,
        model_id: str,
        cache_dir: Path | None = None,
        max_events: int = 6,
        max_ocr_terms: int = 8,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.max_events = max_events
        self.max_ocr_terms = max_ocr_terms
        self._cache: dict[str, dict] = {}
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache ------------------------------------------------------------
    def _disk_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            f"{self.spec.stamp}|{self.model_id}|{key}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load(self, key: str) -> dict | None:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._disk_path(key)
        if path is not None and path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
            self._cache[key] = payload
            return payload
        return None

    def _store(self, key: str, payload: dict) -> None:
        self._cache[key] = payload
        path = self._disk_path(key)
        if path is not None:
            try:
                path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            except OSError:
                logger.debug("không ghi được cache bundle cho %s", key[:60])

    # -- gọi model --------------------------------------------------------
    def _ask(self, query: str, task: str) -> dict | None:
        key = f"{task}|{query}"
        cached = self._load(key)
        if cached is not None:
            return cached

        messages = [
            {"role": "user", "content": self.spec.render(query=query, task=task)}
        ]
        try:
            text = self.client.chat_completion(
                messages,
                model=self.model_id,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
            ).text
            payload = _extract_json_object(text)
        except (ProviderError, ValueError, json.JSONDecodeError) as exc:
            # Hỏng thì mất phần cải thiện, KHÔNG mất truy vấn: caller giữ bundle rule.
            logger.warning("prepare_bundle hỏng, dùng bundle rule: %s", exc)
            return None

        if not isinstance(payload, dict):
            return None
        self._store(key, payload)
        return payload

    # -- API --------------------------------------------------------------
    def refine(self, bundle: SearchQueryBundle, *, task: str) -> SearchQueryBundle:
        """Trả bundle đã được LLM cải thiện; giữ nguyên bundle vào nếu không cải thiện được."""

        payload = self._ask(bundle.raw_query or bundle.normalized_query, task)
        if payload is None:
            return bundle

        updates: dict[str, object] = {}
        applied: list[str] = []

        for field, key in (
            ("visual_query", "visual_vi"),
            ("visual_query_en", "visual_en"),
            ("caption_query", "caption_vi"),
            ("asr_query", "asr_vi"),
        ):
            value = _clean_text(payload.get(key))
            if len(value) >= _MIN_TEXT_CHARS:
                updates[field] = value
                applied.append(key)

        ocr_terms = _clean_list(payload.get("ocr_terms"), limit=self.max_ocr_terms)
        if ocr_terms:
            # Cụm trong ngoặc kép do rule trích ra là bằng chứng CHẮC CHẮN
            # (người ra đề gõ nguyên văn chữ trên màn hình), nên luôn giữ và
            # cho đứng trước phần LLM suy đoán.
            merged = list(dict.fromkeys([*bundle.exact_phrases, *ocr_terms]))
            updates["ocr_query"] = " ".join(merged)
            applied.append("ocr_terms")

        events = _clean_list(payload.get("events"), limit=self.max_events)
        if len(events) >= 2:
            updates["events"] = events
            applied.append("events")

        answer_type = _clean_text(payload.get("answer_type")).lower()
        if answer_type:
            try:
                updates["answer_type"] = AnswerType(answer_type)
                applied.append("answer_type")
            except ValueError:
                pass

        if not updates:
            return bundle

        debug = dict(bundle.debug_info)
        debug["llm_bundle"] = {"prompt": self.spec.stamp, "fields": applied}
        updates["debug_info"] = debug
        return replace(bundle, **updates)


__all__ = ["FptQueryBundlePreparer"]
