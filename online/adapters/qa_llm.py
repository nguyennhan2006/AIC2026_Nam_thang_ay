"""FPT LLM adapter cho QA answer generation (PR-15+).

Rule-based `ANSWER_TOOLS` (`online/services/qa.py`) là regex/counter trên
metadata offline — mạnh cho OCR số/màu nhưng không hiểu câu hỏi tự do dạng
suy luận ("người đó đang làm gì và vì sao"). `FptQaAnswerer` gọi LLM thật
trên evidence text (không phải ảnh — VLM rerank đã dùng ảnh ở `rerank.py`),
strict JSON, và **không được bịa**: prompt yêu cầu trả rỗng nếu evidence
không đủ. Verifier độc lập (`verify_answer`) vẫn chạy lại trên kết quả này
giống mọi tool khác — không tin tưởng mù quáng câu trả lời LLM.
"""

from __future__ import annotations

import asyncio
import json

from online.adapters.fpt_client import FptClient
from online.adapters.provider_errors import ProviderError
from online.domain.evidence import EvidencePack
from online.domain.task_results import AnswerCandidate
from online.errors import DependencyUnavailableError

_ANSWER_TYPES = ("count", "color", "ocr_text", "asr_text", "entity", "yes_no", "temporal", "other")

_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi (QA) cho một bản tin thời sự tiếng Việt. "
    "Chỉ được dùng thông tin trong phần BẰNG CHỨNG bên dưới — không suy đoán, "
    "không dùng kiến thức ngoài bằng chứng. Nếu bằng chứng không đủ để trả lời "
    "chắc chắn, trả về answer rỗng (\"\") kèm confidence thấp thay vì đoán.\n"
    "Trả lời NGẮN GỌN, chỉ đúng phần được hỏi — không lặp lại câu hỏi, không "
    "giải thích, không kèm đơn vị/tiền tố thừa trừ khi chính con số/chữ đó "
    "cần đơn vị để không mơ hồ (vd mốc thời gian).\n"
    "Định dạng answer theo đúng answer_type đã chọn — hệ thống chấm điểm so "
    "khớp CHUỖI CON sau khi đã bỏ dấu/hoa-thường, sai định dạng dù đúng nội "
    "dung vẫn có thể bị chấm sai:\n"
    "- count: chỉ một con số Ả Rập (vd \"5\"), không viết chữ, không kèm danh từ.\n"
    "- color: đúng MỘT từ màu tiếng Việt (vd \"đỏ\", \"xanh\"), không mô tả thêm.\n"
    "- yes_no: chỉ \"có\" hoặc \"không\", không thêm gì khác.\n"
    "- temporal: mốc giây dạng \"<số>s\" nếu biết chính xác, hoặc mô tả ngắn "
    "thời điểm nếu evidence chỉ nói tương đối (vd \"trước khi xe tải tới\").\n"
    "- ocr_text: chép lại NGUYÊN VĂN chuỗi chữ xuất hiện trong evidence, không "
    "diễn giải lại.\n"
    "- asr_text/entity/other: cụm từ ngắn nhất đủ trả lời đúng câu hỏi.\n"
    "Luôn trả về ĐÚNG MỘT object JSON, không kèm văn bản nào khác, đúng schema: "
    '{"answer": string, "answer_type": một trong '
    f"{list(_ANSWER_TYPES)}, "
    '"confidence": số từ 0 đến 1}.'
)


def _build_messages(question: str, evidence: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"BẰNG CHỨNG:\n{evidence}\n\nCÂU HỎI: {question}"},
    ]


def _parse_response(text: str) -> dict | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


class FptQaAnswerer:
    """Sinh MỘT answer candidate cho một (câu hỏi, evidence pack) qua FPT LLM.

    Không gọi cho toàn bộ candidate_limit (có thể lên tới hàng trăm) — caller
    (`QaProcessor`) chỉ nên gọi trên vài evidence pack đứng đầu sau khi đã
    xếp hạng bằng rule-based, để chi phí/độ trễ có giới hạn dự đoán được.
    """

    def __init__(
        self,
        client: FptClient,
        *,
        model_id: str,
        max_evidence_chars: int = 3000,
        max_tokens: int = 3000,
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.max_evidence_chars = max_evidence_chars
        # 200 token là quá ít cho model reasoning: nó tiêu hết vào
        # `reasoning_content` rồi trả `content=None`, nên MỌI câu QA đều
        # hỏng và rơi về rule-based mà không ai để ý (đã xảy ra thật).
        self.max_tokens = max_tokens

    async def answer(self, question: str, pack: EvidencePack) -> AnswerCandidate | None:
        evidence = pack.rerank_text(max_chars=self.max_evidence_chars).strip()
        if not evidence:
            return None
        messages = _build_messages(question, evidence)

        def call():
            return self.client.chat_completion(
                messages,
                model=self.model_id,
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

        try:
            result = await asyncio.to_thread(call)
        except ProviderError as exc:
            raise DependencyUnavailableError(f"FPT QA LLM unavailable: {exc}") from exc

        data = _parse_response(result.text)
        if data is None:
            return None
        answer = str(data.get("answer", "")).strip()
        if not answer:
            return None
        answer_type = data.get("answer_type")
        if answer_type not in _ANSWER_TYPES:
            answer_type = "other"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return AnswerCandidate(
            canonical=answer, surface=answer, confidence=confidence,
            answer_type=answer_type, source="fpt_llm",
        )


__all__ = ["FptQaAnswerer"]
