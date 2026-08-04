"""Q&A processor: parse -> route -> tool -> verify -> joint rank (PR-07).

`EvidenceOnlyAnswerGenerator` cũ nối caption/OCR/ASR thành một chuỗi rồi trả
về — đó là *bằng chứng*, không phải *câu trả lời*, nên mọi item QA đều 0 điểm
theo luật (sai một trong ba video/frame/answer là mất trắng).

Kiến trúc ở đây bám đúng flow đã chốt:

    câu hỏi
      -> tách mô tả sự kiện / mục tiêu câu hỏi   (QuestionParser)
      -> định tuyến theo kiểu câu trả lời         (ANSWER_TOOLS)
      -> trích answer candidate từ evidence       (tool)
      -> chuẩn hóa                                (normalize_answer)
      -> verify độc lập trên evidence             (verify_answer)
      -> xếp hạng chung frame + answer            (joint_rank)

Tất cả tool đều là rule/regex trên metadata đã trích xuất offline. Chúng
không đoán: không tìm được bằng chứng thì trả về rỗng và verifier báo
`INSUFFICIENT` — thà không trả lời còn hơn trả lời sai kèm evidence sai.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
import unicodedata

from online.domain.evidence import EvidencePack
from online.domain.task_results import AnswerCandidate, AnswerType, QaResultItem, VerifierStatus
from online.errors import DependencyUnavailableError

if TYPE_CHECKING:
    from online.services.normalizers import ScoreNormalizers

_NUMBER_WORDS_VI = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10,
}
_COLOR_WORDS = {
    "đỏ": "red", "cam": "orange", "vàng": "yellow", "xanh lá": "green",
    "xanh dương": "blue", "xanh": "blue", "tím": "purple", "hồng": "pink",
    "đen": "black", "trắng": "white", "xám": "gray",
}

NUMBER_RE = re.compile(r"\b\d[\d.,]*\b")
TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)


def normalize_answer(text: str) -> str:
    """Chuẩn hóa để so khớp: bỏ dấu, bỏ ký tự thừa, gộp khoảng trắng."""

    decomposed = unicodedata.normalize("NFD", str(text).casefold())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", stripped.replace("đ", "d"))).strip()


def answer_matches(predicted: str, accepted: tuple[str, ...] | list[str]) -> bool:
    """So khớp answer đã nộp với danh sách answer được chấp nhận của gold.

    Dùng bởi cả eval harness (`scripts/eval_tasks.py`) và local scorer
    (`online/competition/scorer.py`) — một định nghĩa duy nhất cho "đúng" để
    hai nơi không lệch nhau.
    """

    if not predicted or not accepted:
        return False
    normalized = normalize_answer(predicted)
    return any(normalize_answer(item) in normalized for item in accepted)


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParsedQuestion:
    """Câu hỏi đã tách thành phần đi tìm evidence và phần cần trả lời."""

    event_query: str
    question_target: str
    answer_type: AnswerType
    raw: str

    @property
    def retrieval_query(self) -> str:
        """Query dùng để tìm evidence.

        Ghép cả hai phần: mô tả sự kiện định vị đúng đoạn video, còn mục tiêu
        câu hỏi thường chứa danh từ then chốt (vd "biển báo", "xe máy").
        """

        return f"{self.event_query} {self.question_target}".strip()


_ANSWER_TYPE_RULES: tuple[tuple[AnswerType, tuple[str, ...]], ...] = (
    ("count", ("bao nhiêu", "mấy", "số lượng", "how many")),
    ("color", ("màu gì", "màu nào", "what color", "màu sắc")),
    ("yes_no", ("có phải", "phải không", "đúng không", "có ... không", "is there", "does")),
    ("temporal", ("khi nào", "trước hay sau", "lúc nào", "when", "sau đó điều gì")),
    ("ocr_text", ("ghi gì", "chữ gì", "dòng chữ", "biển ghi", "tiêu đề", "written")),
    ("asr_text", ("nói gì", "phát biểu gì", "said", "câu nói")),
    ("entity", ("là gì", "là ai", "cái gì", "loại gì", "ai đang", "what", "who")),
)

_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")


class QuestionParser:
    """Tách câu hỏi thành mô tả sự kiện + mục tiêu, không dùng LLM."""

    def parse(self, question: str, event_description: str | None = None) -> ParsedQuestion:
        text = question.strip()
        lowered = text.casefold()
        answer_type: AnswerType = "other"
        for candidate_type, markers in _ANSWER_TYPE_RULES:
            if any(marker in lowered for marker in markers):
                answer_type = candidate_type
                break

        if event_description:
            return ParsedQuestion(event_description.strip(), text, answer_type, text)

        # Không có mô tả sự kiện riêng: câu đầu thường mô tả bối cảnh, câu chứa
        # dấu hỏi là mục tiêu. Query một câu thì cả hai trùng nhau.
        parts = [item.strip() for item in _SPLIT_RE.split(text) if item.strip()]
        if len(parts) >= 2:
            target = next((item for item in reversed(parts) if "?" in item), parts[-1])
            context = " ".join(item for item in parts if item is not target)
            return ParsedQuestion(context, target, answer_type, text)
        return ParsedQuestion(text, text, answer_type, text)


# --------------------------------------------------------------------------
# Tools theo kiểu câu trả lời
# --------------------------------------------------------------------------


def _count_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    """Đếm bằng OCR số trước, rồi tới object detection.

    OCR được ưu tiên vì bản tin thường ghi rõ con số trên lower-third, còn đếm
    bounding box dễ sai do che khuất và trùng lặp giữa các frame.
    """

    candidates: list[AnswerCandidate] = []
    target_words = set(normalize_answer(parsed.question_target).split())
    for text in filter(None, [pack.ocr_text, pack.caption_text, pack.asr_window]):
        normalized = normalize_answer(text)
        if not (target_words & set(normalized.split())):
            continue
        for number in NUMBER_RE.findall(text):
            candidates.append(AnswerCandidate(
                canonical=number.strip(".,"), surface=number, confidence=0.72,
                answer_type="count", source="ocr_number",
            ))
        for word, value in _NUMBER_WORDS_VI.items():
            if word in text.casefold():
                candidates.append(AnswerCandidate(
                    canonical=str(value), surface=word, aliases=[word],
                    confidence=0.5, answer_type="count", source="ocr_number_word",
                ))
    if not candidates:
        labels = Counter(
            label for frame in pack.keyframes for label in frame.object_labels
        )
        for label, count in labels.most_common(1):
            if any(token in normalize_answer(label) for token in target_words):
                candidates.append(AnswerCandidate(
                    canonical=str(count), surface=str(count),
                    confidence=0.4, answer_type="count", source="object_detection",
                ))
    return candidates


def _color_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    counts = Counter(
        color for frame in pack.keyframes for color in frame.dominant_colors
    )
    total = sum(counts.values()) or 1
    return [
        AnswerCandidate(
            canonical=color, surface=color,
            aliases=[vi for vi, en in _COLOR_WORDS.items() if en == color],
            confidence=min(0.4 + count / total, 0.9),
            answer_type="color", source="color_metadata",
        )
        for color, count in counts.most_common(3)
    ]


def _ocr_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    texts = [text for frame in pack.keyframes for text in frame.ocr_texts]
    if not texts and pack.ocr_text:
        texts = [pack.ocr_text]
    # Chuỗi OCR dài nhất thường là tiêu đề/biển hiệu — đúng thứ câu hỏi nhắm tới.
    ordered = sorted({item.strip() for item in texts if item.strip()}, key=len, reverse=True)
    return [
        AnswerCandidate(
            canonical=text, surface=text,
            confidence=0.75 if index == 0 else 0.45,
            answer_type="ocr_text", source="ocr_exact",
        )
        for index, text in enumerate(ordered[:3])
    ]


def _asr_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    if not pack.asr_window:
        return []
    return [AnswerCandidate(
        canonical=pack.asr_window.strip()[:200], surface=pack.asr_window.strip()[:200],
        confidence=0.55, answer_type="asr_text", source="asr_transcript",
    )]


def _entity_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    labels = Counter(
        label for frame in pack.keyframes for label in frame.object_labels
    )
    candidates = [
        AnswerCandidate(
            canonical=label, surface=label, confidence=min(0.35 + 0.1 * count, 0.7),
            answer_type="entity", source="object_detection",
        )
        for label, count in labels.most_common(3)
    ]
    if not candidates and pack.caption_text:
        candidates.append(AnswerCandidate(
            canonical=pack.caption_text.strip()[:160], surface=pack.caption_text.strip()[:160],
            confidence=0.3, answer_type="entity", source="caption",
        ))
    return candidates


def _yes_no_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    """Yes/no dựa trên độ phủ từ khóa của câu hỏi trên evidence."""

    tokens = {
        token for token in normalize_answer(parsed.question_target).split() if len(token) >= 3
    }
    if not tokens:
        return []
    evidence = normalize_answer(pack.rerank_text(max_chars=4000))
    matched = sum(1 for token in tokens if token in evidence)
    ratio = matched / len(tokens)
    positive = ratio >= 0.6
    return [AnswerCandidate(
        canonical="có" if positive else "không",
        surface="có" if positive else "không",
        aliases=["yes"] if positive else ["no"],
        confidence=min(0.4 + abs(ratio - 0.5), 0.85),
        answer_type="yes_no", source="evidence_coverage",
    )]


def _temporal_tool(parsed: ParsedQuestion, pack: EvidencePack) -> list[AnswerCandidate]:
    """Câu hỏi thời điểm: trả về mốc giây của evidence, kèm ngữ cảnh liền kề."""

    candidates = [AnswerCandidate(
        canonical=f"{pack.start_sec:.0f}s", surface=f"{pack.start_sec:.1f}s",
        confidence=0.4, answer_type="temporal", source="scene_interval",
    )]
    if pack.next_context:
        candidates.append(AnswerCandidate(
            canonical=(pack.next_context.caption or "")[:160],
            surface=(pack.next_context.caption or "")[:160],
            confidence=0.3, answer_type="temporal", source="next_scene",
        ))
    return [item for item in candidates if item.canonical]


ANSWER_TOOLS = {
    "count": _count_tool,
    "color": _color_tool,
    "ocr_text": _ocr_tool,
    "asr_text": _asr_tool,
    "entity": _entity_tool,
    "yes_no": _yes_no_tool,
    "temporal": _temporal_tool,
    "other": _entity_tool,
}


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


def verify_answer(answer: AnswerCandidate, pack: EvidencePack) -> VerifierStatus:
    """Kiểm chứng độc lập: câu trả lời có thật sự nằm trong evidence không.

    Chạy TÁCH KHỎI tool sinh ra nó — nếu dùng chính tool để tự xác nhận thì
    verifier chỉ lặp lại niềm tin của tool.
    """

    evidence = normalize_answer(pack.rerank_text(max_chars=4000))
    if not evidence:
        return "INSUFFICIENT"
    surfaces = [answer.canonical, answer.surface, *answer.aliases]
    if any(normalize_answer(item) and normalize_answer(item) in evidence for item in surfaces):
        return "SUPPORTED"
    if answer.answer_type in ("yes_no", "temporal"):
        # Hai kiểu này suy luận từ evidence chứ không trích nguyên văn, nên
        # không tìm thấy chuỗi khớp là chuyện bình thường.
        return "PARTIAL"
    if answer.answer_type == "count":
        # Con số khác hẳn xuất hiện trong evidence => mâu thuẫn trực tiếp.
        numbers = set(NUMBER_RE.findall(pack.rerank_text(max_chars=4000)))
        if numbers and answer.canonical not in numbers:
            return "CONTRADICTED"
    return "INSUFFICIENT"


_VERIFIER_WEIGHT: dict[VerifierStatus, float] = {
    "SUPPORTED": 1.0,
    "PARTIAL": 0.6,
    "INSUFFICIENT": 0.25,
    "CONTRADICTED": 0.0,
}


# --------------------------------------------------------------------------
# Joint ranking
# --------------------------------------------------------------------------


class QaProcessor:
    """Sinh và xếp hạng bộ ba (video, frame, answer).

    `llm_answerer` (vd `FptQaAnswerer`, PR-15+) là tùy chọn: khi có, `answer_async`
    gọi thêm LLM trên `llm_top_n` evidence pack đứng đầu (theo joint score
    rule-based) để thay thế bằng answer ngữ nghĩa hơn — rule-based
    (`ANSWER_TOOLS`) vẫn luôn chạy trước làm baseline/fallback, vì `score_qa`
    (luật thi) chấm bất kỳ dòng nào trong submission list đúng cả ba điều
    kiện, không chỉ rank 1, nên giữ cả hai nguồn candidate tăng cơ hội trúng.
    """

    def __init__(
        self,
        parser: QuestionParser | None = None,
        *,
        llm_answerer=None,
        llm_top_n: int = 5,
    ) -> None:
        self.parser = parser or QuestionParser()
        self.llm_answerer = llm_answerer
        self.llm_top_n = llm_top_n

    async def answer_async(
        self,
        question: str,
        packs: list[EvidencePack],
        *,
        frame_scores: dict[str, float] | None = None,
        event_description: str | None = None,
        limit: int = 100,
        normalizers: "ScoreNormalizers | None" = None,
    ) -> tuple[list[QaResultItem], list[str]]:
        """Bản async của `answer()` — gọi thêm LLM nếu có cấu hình.

        Trả kèm `warnings`: LLM lỗi/JSON hỏng không làm hỏng kết quả rule-based
        đã có, chỉ bỏ qua việc nâng cấp dòng đó (đúng nguyên tắc no-silent-
        degradation đã áp dụng cho rerank_pipeline).
        """

        base = self.answer(
            question, packs, frame_scores=frame_scores,
            event_description=event_description, limit=limit, normalizers=normalizers,
        )
        if self.llm_answerer is None or not base:
            return base, []

        by_id = {pack.candidate_id: pack for pack in packs}
        warnings: list[str] = []
        seen_packs: set[str] = set()
        enhanced: list[QaResultItem] = []
        for item in base:
            pack = next((by_id[cid] for cid in item.evidence_ids if cid in by_id), None)
            if pack is None or pack.candidate_id in seen_packs or len(seen_packs) >= self.llm_top_n:
                enhanced.append(item)
                continue
            seen_packs.add(pack.candidate_id)
            try:
                llm_candidate = await self.llm_answerer.answer(question, pack)
            except DependencyUnavailableError as exc:
                warnings.append(f"FPT QA LLM bỏ qua {pack.candidate_id}: {exc}")
                enhanced.append(item)
                continue
            if llm_candidate is None:
                enhanced.append(item)
                continue
            status = verify_answer(llm_candidate, pack)
            if status == "CONTRADICTED":
                # LLM mâu thuẫn với evidence — giữ answer rule-based an toàn hơn.
                enhanced.append(item)
                continue
            enhanced.append(
                item.model_copy(
                    update={
                        "answer": llm_candidate.surface,
                        "canonical_answer": llm_candidate.canonical,
                        "answer_type": llm_candidate.answer_type,
                        "verifier_status": status,
                    }
                )
            )
        return enhanced, warnings

    def answer(
        self,
        question: str,
        packs: list[EvidencePack],
        *,
        frame_scores: dict[str, float] | None = None,
        event_description: str | None = None,
        limit: int = 100,
        normalizers: "ScoreNormalizers | None" = None,
    ) -> list[QaResultItem]:
        parsed = self.parser.parse(question, event_description)
        tool = ANSWER_TOOLS[parsed.answer_type]
        frame_scores = frame_scores or {}
        # Cùng lý do với kis.py: mẫu số lấy từ pool trước dedup, không phải
        # max của lát cắt hiện tại (EVAL-01 prefix invariance).
        best_frame_score = (
            normalizers.best_retrieval_score if normalizers is not None
            else (max(frame_scores.values(), default=1.0) or 1.0)
        )

        rows: list[tuple[float, QaResultItem]] = []
        for pack in packs:
            if pack.best_frame_idx is None and not pack.keyframes:
                continue
            frame_idx = pack.best_frame_idx
            if frame_idx is None:
                frame_idx = pack.keyframes[0].frame_idx
            evidence_confidence = frame_scores.get(pack.candidate_id, 0.0) / best_frame_score
            for candidate in tool(parsed, pack):
                if not candidate.canonical.strip():
                    continue
                status = verify_answer(candidate, pack)
                # Nhân chứ không cộng: sai một khâu là item mất giá trị, đúng
                # như cách luật chấm (sai video/frame/answer đều = 0).
                joint = (
                    max(evidence_confidence, 0.05)
                    * candidate.confidence
                    * _VERIFIER_WEIGHT[status]
                )
                if joint <= 0.0:
                    continue
                rows.append((
                    joint,
                    QaResultItem(
                        rank=1,
                        video_id=pack.video_id,
                        frame_idx=frame_idx,
                        answer=candidate.surface,
                        canonical_answer=candidate.canonical,
                        answer_type=candidate.answer_type,
                        joint_score=joint,
                        verifier_status=status,
                        scene_id=pack.scene_id,
                        evidence_ids=[pack.candidate_id],
                        evidence=pack,
                    ),
                ))

        rows.sort(key=lambda item: (-item[0], item[1].video_id, item[1].frame_idx))
        # Một (video, frame, answer) chỉ nên xuất hiện một lần trong submission.
        seen: set[tuple[str, int, str]] = set()
        output: list[QaResultItem] = []
        for _score, item in rows:
            key = (item.video_id, item.frame_idx, normalize_answer(item.canonical_answer))
            if key in seen:
                continue
            seen.add(key)
            output.append(item.model_copy(update={"rank": len(output) + 1}))
            if len(output) >= limit:
                break
        return output


__all__ = [
    "ANSWER_TOOLS",
    "ParsedQuestion",
    "QaProcessor",
    "QuestionParser",
    "answer_matches",
    "normalize_answer",
    "verify_answer",
]
