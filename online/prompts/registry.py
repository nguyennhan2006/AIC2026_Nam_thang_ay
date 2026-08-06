"""Nơi ĐIỀU PHỐI duy nhất cho mọi prompt gửi tới LLM/VLM.

Trước file này, prompt nằm rải rác dưới dạng hằng số module trong ba adapter
(`qa_llm.py`, `rerank.py`, `fpt_query.py`). Hậu quả cụ thể, đều đã xảy ra:

- **Không biết prompt nào đang chạy.** So hai lần eval mà không có phiên bản
  prompt trong trace thì không phân biệt được "đổi model" với "đổi prompt".
- **Vai model bị chọn nhầm ngay tại chỗ gọi.** `max_tokens=200` hard-code
  trong `qa_llm.py` gặp model reasoning là hỏng 100% lệnh gọi QA — mà không ai
  thấy, vì QA lặng lẽ rơi về rule-based. Ngân sách token là thuộc tính của
  PROMPT (nó cần bao nhiêu chỗ để trả lời), nên phải khai ngay cạnh prompt.
- **Không kiểm kê được.** Không có cách nào liệt kê "hệ thống đang dùng bao
  nhiêu prompt, cái nào cần model reasoning, cái nào chấp nhận model rẻ".

Mỗi prompt khai báo luôn VAI MODEL nó cần (`model_role`) chứ không khai tên
model. Tên model là chuyện cấu hình theo môi trường; còn "việc này có cần suy
luận nhiều bước không" là thuộc tính của chính việc đó, không đổi theo môi
trường.

Ba vai:

``fast``
    Việc máy móc, đầu ra ngắn, không cần suy luận. PHẢI là model trả thẳng
    `content`. Đo trên FPT: gemma-4-31B-it dịch xong một câu trong 9 token,
    còn Qwen3.6-27B tốn 1652 token cho đúng việc đó.

``reasoning``
    Cần bắc cầu nhiều bước hoặc cân nhắc mâu thuẫn (trả lời QA, chọn bằng
    chứng, đề xuất trọng số). Ngân sách token phải rộng vì phần suy luận tiêu
    trước, câu trả lời sinh sau.

``vlm``
    Cần nhìn ảnh. Trên FPT chỉ được **một ảnh mỗi prompt**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelRole = Literal["fast", "reasoning", "vlm"]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Một prompt kèm mọi thứ cần để gọi nó đúng cách."""

    prompt_id: str
    version: str
    model_role: ModelRole
    template: str
    temperature: float = 0.0
    max_tokens: int = 512
    json_output: bool = False
    notes: str = ""

    def render(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    @property
    def stamp(self) -> str:
        """Chuỗi đi vào trace để so hai lần chạy biết chính xác đổi gì."""

        return f"{self.prompt_id}@{self.version}"


# --------------------------------------------------------------------------
# Query understanding
# --------------------------------------------------------------------------

TRANSLATE_QUERY = PromptSpec(
    prompt_id="query.translate_vi_en",
    version="1",
    model_role="fast",
    max_tokens=120,
    notes=(
        "Vector ảnh sinh bằng openai/clip-vit-large-patch14, mà text tower của "
        "CLIP chỉ được huấn luyện trên tiếng Anh. Đây là thay đổi có tác động "
        "lớn nhất đo được: KIS MRR 0.547 -> 0.720, R@20 chạm trần 1.000."
    ),
    template="""Dịch truy vấn tìm kiếm video sau sang tiếng Anh.

Đây là câu mô tả một cảnh quay để đối chiếu với ảnh, nên hãy dịch thành cụm
mô tả THỊ GIÁC ngắn gọn, giữ nguyên danh từ cụ thể và hành động nhìn thấy được.
Bỏ các từ chỉ thao tác tìm kiếm ("tìm video về", "cảnh quay có").

Chỉ trả về BẢN DỊCH, không giải thích, không dấu ngoặc kép.

Truy vấn: {query}""",
)


EXPAND_QUERY = PromptSpec(
    prompt_id="query.expand_vi",
    version="2",
    model_role="fast",
    temperature=0.2,
    max_tokens=200,
    notes=(
        "v1 gây query drift: nó sinh từ đồng nghĩa tự do nên kéo theo candidate "
        "chỉ khớp term phụ — đo được KIS MRR 0.547 -> 0.512 và AVS nDCG 0.238 "
        "-> 0.201. v2 siết lại: chỉ nhận thứ NHÌN THẤY ĐƯỢC trong cảnh và hạ "
        "trần số term."
    ),
    template="""Bạn đang mở rộng truy vấn cho công cụ tìm kiếm BM25 trên kho
caption TIẾNG VIỆT mô tả cảnh trong video.

Truy vấn: {query}

Liệt kê tối đa {max_terms} từ/cụm từ TIẾNG VIỆT chỉ những thứ NHÌN THẤY ĐƯỢC
trong đúng cảnh mà truy vấn mô tả — vật thể, hành động, hoặc bối cảnh gần như
chắc chắn xuất hiện cùng.

CẤM:
- từ đồng nghĩa chung chung không thu hẹp được cảnh nào ("hình ảnh", "cảnh quay")
- suy đoán nguyên nhân, hậu quả, cảm xúc
- tên riêng, con số, địa danh không có trong truy vấn
- lặp lại từ đã có trong truy vấn

Thà trả mảng RỖNG còn hơn thêm từ không chắc chắn.

Trả về DUY NHẤT một mảng JSON các chuỗi, ví dụ: ["vòi nước", "tia nước"]""",
)


# --------------------------------------------------------------------------
# Xác minh & lọc cuối
# --------------------------------------------------------------------------

VLM_RERANK = PromptSpec(
    prompt_id="rerank.vlm_frame",
    version="1",
    model_role="vlm",
    max_tokens=300,
    json_output=True,
    notes=(
        "Cố ý KHÔNG đưa caption/OCR vào prompt: caption đã là thứ bm25_caption "
        "và text rerank dùng. Cho VLM đọc lại caption thì nhánh này chỉ lặp "
        "tín hiệu cũ và con số đo được không diễn giải được."
    ),
    template="""Bạn đang xác minh kết quả tìm kiếm video.

TRUY VẤN: {query}

Nhìn ẢNH và chấm xem khung hình này có đúng là cảnh mà truy vấn mô tả không.
Chỉ dựa vào những gì THẤY ĐƯỢC trong ảnh. Không suy đoán từ ngữ cảnh không có.

Trả về DUY NHẤT một object JSON, không kèm giải thích ngoài JSON:
{{"relevance": <0.0-1.0>,
  "must_match_coverage": <0.0-1.0, tỉ lệ các yếu tố bắt buộc của truy vấn nhìn thấy được>,
  "contradictions": [<những gì trong ảnh MÂU THUẪN với truy vấn, để rỗng nếu không có>],
  "evidence_summary": "<một câu, thấy gì trong ảnh>"}}""",
)


SELECT_EVIDENCE = PromptSpec(
    prompt_id="evidence.select",
    version="1",
    model_role="reasoning",
    max_tokens=1200,
    json_output=True,
    notes=(
        "Lý do tồn tại: `EvidencePack.rerank_text()` gộp NGUYÊN caption + OCR "
        "+ ASR + object + action, nên overlay của đài truyền hình lọt vào cùng "
        "hạng với nội dung cảnh. Quan sát thật: bằng chứng trả về là "
        "'HTV9 HD' và '06:33:29' trong khi truy vấn không hề hỏi kênh nào hay "
        "mấy giờ. Đó là watermark, không phải nội dung."
    ),
    template="""Bạn đang chọn BẰNG CHỨNG cho một kết quả tìm kiếm video.

TRUY VẤN: {query}

BẰNG CHỨNG THÔ (gộp máy móc từ caption/OCR/ASR/vật thể, chưa lọc):
{evidence}

Chọn ra CHỈ những mẩu thực sự chứng minh kết quả này khớp truy vấn.

LOẠI BỎ, kể cả khi chúng nổi bật trong văn bản thô:
- logo/tên kênh, tên chương trình ("HTV9 HD", "VTV1", "Bản tin thời sự")
- đồng hồ, ngày giờ hiển thị trên màn hình ("06:33:29")
- chữ chạy, tên người dẫn, thông tin bản quyền
- bất cứ thứ gì không ai hỏi tới và không giúp phân biệt cảnh này với cảnh khác

Những thứ trên là lớp phủ đồ hoạ của đài, không phải nội dung cảnh. Chúng
xuất hiện ở MỌI khung hình nên không chứng minh được gì.

Nếu không mẩu nào thực sự chứng minh được, trả `supports: false` và để
`evidence` rỗng — nói thẳng còn hơn dựng bằng chứng giả.

Trả về DUY NHẤT một object JSON:
{{"supports": <true|false>,
  "evidence": [<tối đa {max_items} mẩu NGUYÊN VĂN trích từ bằng chứng thô>],
  "reason": "<một câu: vì sao khớp, hoặc vì sao không>",
  "dropped_as_overlay": [<những mẩu bị loại vì là lớp phủ của đài>]}}""",
)


QA_ANSWER = PromptSpec(
    prompt_id="qa.answer",
    version="2",
    model_role="reasoning",
    max_tokens=3000,
    json_output=True,
    notes=(
        "max_tokens phải rộng: model reasoning tiêu ngân sách cho "
        "`reasoning_content` TRƯỚC rồi mới sinh câu trả lời. Bản trước "
        "hard-code 200 nên MỌI lệnh gọi QA đều hỏng và lặng lẽ rơi về "
        "rule-based — không ai phát hiện suốt nhiều đợt đo."
    ),
    template="",  # dựng động trong qa_llm.py (system + user tách riêng)
)


# --------------------------------------------------------------------------
# Đề xuất trọng số
# --------------------------------------------------------------------------

RECOMMEND_WEIGHTS = PromptSpec(
    prompt_id="search.recommend_weights",
    version="1",
    model_role="reasoning",
    max_tokens=2000,
    json_output=True,
    notes=(
        "Chỉ ĐỀ XUẤT, không tự áp: người dùng phải nhìn thấy lý do rồi mới "
        "quyết. Trọng số tự đổi ngầm giữa hai lần tìm là kiểu thay đổi khiến "
        "không ai tái lập được kết quả."
    ),
    template="""Bạn đang chỉnh trọng số cho một hệ tìm kiếm video đa nhánh.

TRUY VẤN: {query}
LOẠI TÁC VỤ: {task}

Các nhánh đang bật, mỗi nhánh tìm trên một loại dữ liệu khác nhau:
{branches}

Đặt trọng số 0.0–3.0 cho từng nhánh theo mức hữu ích của nó VỚI TRUY VẤN NÀY.

Nguyên tắc:
- 0.0 nghĩa là TẮT HẲN nhánh đó. Dùng khi truy vấn không có manh mối nào cho
  loại dữ liệu ấy — ví dụ không có chữ trong ngoặc kép thì nhánh OCR chỉ tạo
  kết quả sai, vì nó sẽ khớp bừa chữ chạy của bản tin.
- Nhánh chữ (OCR) chỉ đáng bật khi truy vấn nhắc tới chữ nhìn thấy trên màn hình.
- Nhánh lời nói (ASR) chỉ đáng bật khi truy vấn nhắc tới lời ai đó nói.
- Truy vấn thuần hình ảnh thì nhánh thị giác và caption phải nặng nhất.
- Đừng dàn đều mọi nhánh: dàn đều là không quyết định gì cả.

Trả về DUY NHẤT một object JSON:
{{"weights": {{"<branch_id>": <số>, ...}},
  "reason": "<một câu, vì sao phân bổ như vậy>",
  "disabled": [<branch_id bị đặt 0.0 và lý do ngắn>]}}""",
)


PROMPTS: dict[str, PromptSpec] = {
    spec.prompt_id: spec
    for spec in (
        TRANSLATE_QUERY,
        EXPAND_QUERY,
        VLM_RERANK,
        SELECT_EVIDENCE,
        QA_ANSWER,
        RECOMMEND_WEIGHTS,
    )
}


def prompts_by_role() -> dict[str, list[str]]:
    """Kiểm kê: vai model nào đang gánh những prompt nào.

    Dùng để trả lời "bật FPT thì thật ra hệ thống gọi bao nhiêu loại model" mà
    không phải đọc hết adapter.
    """

    grouped: dict[str, list[str]] = {}
    for spec in PROMPTS.values():
        grouped.setdefault(spec.model_role, []).append(spec.stamp)
    return {role: sorted(stamps) for role, stamps in sorted(grouped.items())}


__all__ = [
    "EXPAND_QUERY",
    "PROMPTS",
    "PromptSpec",
    "QA_ANSWER",
    "RECOMMEND_WEIGHTS",
    "SELECT_EVIDENCE",
    "TRANSLATE_QUERY",
    "VLM_RERANK",
    "prompts_by_role",
]
