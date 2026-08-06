"""Hai việc LLM làm ở tầng "cố vấn": đề xuất trọng số, và lọc bằng chứng.

Khác với `fpt_query.py` (biến đổi truy vấn trước khi tìm), hai adapter ở đây
không đụng vào retrieval. Chúng đọc kết quả rồi nói lại cho người dùng — nên
hỏng thì mất lời khuyên, không mất kết quả.

``FptWeightRecommender``
    Đề xuất trọng số cho từng nhánh theo truy vấn. **Chỉ đề xuất, không tự
    áp.** Trọng số tự đổi ngầm giữa hai lần tìm là kiểu thay đổi khiến không
    ai tái lập được kết quả, và đúng loại "hạ cấp âm thầm" mà hệ này đã mất
    nhiều thời gian để loại bỏ.

``FptEvidenceSelector``
    Lọc bằng chứng thô xuống còn phần thật sự liên quan tới truy vấn.
    `EvidencePack.rerank_text()` gộp máy móc caption + OCR + ASR + object +
    action, nên lớp phủ đồ hoạ của đài truyền hình lọt vào ngang hàng với nội
    dung cảnh. Quan sát thật: bằng chứng trả về là `HTV9 HD` và `06:33:29`
    trong khi truy vấn không hỏi kênh nào cũng chẳng hỏi mấy giờ. Những chuỗi
    đó xuất hiện ở MỌI khung hình nên không chứng minh được gì.
"""

from __future__ import annotations

import asyncio
import json

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.domain.evidence import EvidencePack
from online.prompts import RECOMMEND_WEIGHTS, SELECT_EVIDENCE

# Mô tả nhánh cho LLM. Không lấy từ docstring adapter: LLM cần biết nhánh đó
# tìm trên LOẠI DỮ LIỆU nào để quyết có đáng bật không, chứ không cần biết nó
# cài đặt bằng gì.
BRANCH_DESCRIPTIONS: dict[str, str] = {
    "dense_visual": "so truy vấn với NỘI DUNG HÌNH ẢNH của khung hình (CLIP)",
    "bm25_caption": "tìm từ khoá trong câu mô tả cảnh do model sinh",
    "bm25_ocr": "tìm từ khoá trong CHỮ HIỆN TRÊN MÀN HÌNH",
    "bm25_asr": "tìm từ khoá trong LỜI NÓI đã gỡ băng",
    "bm25_keyword": "tìm trong danh sách từ khoá rút gọn của cảnh",
    "bm25_object": "tìm theo tên VẬT THỂ được phát hiện trong khung hình",
    "bm25_action": "tìm theo HÀNH ĐỘNG được gán cho cảnh",
    "ocr_fuzzy": "khớp gần đúng chuỗi chữ trên màn hình (chịu được lỗi nhận dạng)",
    "color_search": "tìm theo MÀU SẮC chủ đạo của khung hình",
    "event_search": "tìm theo sự kiện đã gom nhóm theo thời gian",
    "lexical_hash_fallback": "nhánh dự phòng khi export chưa có embedding thật",
}


class FptWeightRecommender:
    """Đề xuất trọng số từng nhánh cho một truy vấn cụ thể."""

    spec = RECOMMEND_WEIGHTS

    def __init__(self, client: FptClient, *, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def _describe(self, branch_ids: list[str]) -> str:
        return "\n".join(
            f"- {branch_id}: {BRANCH_DESCRIPTIONS.get(branch_id, 'không rõ')}"
            for branch_id in branch_ids
        )

    async def recommend(self, query: str, *, task: str, branch_ids: list[str]) -> dict | None:
        """`{"weights": {...}, "reason": str, "disabled": [...]}` hoặc None.

        Trả None khi hỏng thay vì ném: đây là lời khuyên kèm theo kết quả tìm
        kiếm, mất nó không được phép làm hỏng chính kết quả.
        """

        if not branch_ids:
            return None
        messages = [
            {
                "role": "user",
                "content": self.spec.render(
                    query=query, task=task, branches=self._describe(branch_ids)
                ),
            }
        ]

        def call() -> str:
            return self.client.chat_completion(
                messages,
                model=self.model_id,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
                response_format={"type": "json_object"} if self.spec.json_output else None,
            ).text

        try:
            payload = _parse_json_object(await asyncio.to_thread(call))
        except (ProviderError, ValueError):
            return None
        if payload is None:
            return None

        # Chỉ nhận nhánh CÓ THẬT và số hợp lệ: LLM bịa thêm branch_id thì phần
        # bịa đó phải bị bỏ, không được đẩy xuống tầng cấu hình.
        allowed = set(branch_ids)
        weights: dict[str, float] = {}
        for key, value in (payload.get("weights") or {}).items():
            if key not in allowed:
                continue
            try:
                weights[key] = min(3.0, max(0.0, float(value)))
            except (TypeError, ValueError):
                continue
        if not weights:
            return None
        return {
            "weights": weights,
            "reason": str(payload.get("reason") or "").strip(),
            "disabled": [
                str(item) for item in (payload.get("disabled") or []) if isinstance(item, str)
            ],
            "prompt_version": self.spec.stamp,
            "model_id": self.model_id,
        }


class FptEvidenceSelector:
    """Lọc bằng chứng thô của một pack xuống phần thật sự liên quan."""

    spec = SELECT_EVIDENCE

    def __init__(
        self, client: FptClient, *, model_id: str, max_items: int = 4, max_chars: int = 1800
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.max_items = max_items
        self.max_chars = max_chars

    async def select(self, query: str, pack: EvidencePack) -> dict | None:
        raw = pack.rerank_text(max_chars=self.max_chars).strip()
        if not raw:
            return None
        messages = [
            {
                "role": "user",
                "content": self.spec.render(
                    query=query, evidence=raw, max_items=self.max_items
                ),
            }
        ]

        def call() -> str:
            return self.client.chat_completion(
                messages,
                model=self.model_id,
                temperature=self.spec.temperature,
                max_tokens=self.spec.max_tokens,
                response_format={"type": "json_object"} if self.spec.json_output else None,
            ).text

        try:
            payload = _parse_json_object(await asyncio.to_thread(call))
        except (ProviderError, ValueError):
            return None
        if payload is None:
            return None
        return {
            "candidate_id": pack.candidate_id,
            "supports": bool(payload.get("supports")),
            "evidence": [
                str(item).strip()
                for item in (payload.get("evidence") or [])
                if isinstance(item, str) and str(item).strip()
            ][: self.max_items],
            "reason": str(payload.get("reason") or "").strip(),
            "dropped_as_overlay": [
                str(item).strip()
                for item in (payload.get("dropped_as_overlay") or [])
                if isinstance(item, str) and str(item).strip()
            ],
            "prompt_version": self.spec.stamp,
        }

    async def select_many(
        self, query: str, packs: list[EvidencePack], *, max_concurrency: int = 4
    ) -> list[dict | None]:
        """Giữ ĐÚNG thứ tự packs — caller ghép theo vị trí."""

        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def one(pack: EvidencePack) -> dict | None:
            async with semaphore:
                return await self.select(query, pack)

        return list(await asyncio.gather(*(one(pack) for pack in packs)))


def _parse_json_object(text: str) -> dict | None:
    """Bóc object JSON ngoài cùng. Model hay thêm câu dẫn hoặc bọc ```json."""

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


__all__ = ["BRANCH_DESCRIPTIONS", "FptEvidenceSelector", "FptWeightRecommender"]
