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


PREPARE_QUERY_BUNDLE = PromptSpec(
    prompt_id="query.prepare_bundle",
    version="3",
    model_role="fast",
    temperature=0.0,
    max_tokens=420,
    json_output=True,
    notes=(
        "Soạn dữ liệu ĐẦU VÀO RIÊNG cho từng search engine theo đúng thế mạnh "
        "của engine đó, thay cho một query chung.\n\n"
        "Vì sao cần LLM ở đây (số đo 2026-08-27, gold L21_V023 frame 25995): "
        "hai truy vấn trỏ CÙNG MỘT FRAME cho thứ hạng lệch nhau 35 lần —\n"
        "  'Bàn tay đeo đồng hồ đổ chất lỏng vào bát trắng đặt trên cân điện "
        "tử'  -> rank 1\n"
        "  'một con cá được đặt lên cân, sau đó ... trên cân là bao nhiêu'"
        "        -> rank 35\n"
        "Khác biệt là MÔ TẢ FRAME so với KỂ CHUYỆN + hỏi. Tầng rule chỉ cắt "
        "được phần hỏi, nó không viết lại được câu kể thành mô tả thị giác — "
        "đó chính là việc của prompt này.\n\n"
        "Fallback là tầng rule (online/services/query/), nên prompt hỏng hay "
        "timeout chỉ làm mất phần cải thiện, không mất truy vấn.\n\n"
        "v2 — THÊM RÀNG BUỘC ĐỘ HIẾM. v1 chỉ bảo 'mô tả khung hình', nên model "
        "dùng từ phổ thông và query bị video 'nam châm' nuốt. Đo trên corpus "
        "873 video, số video chứa mỗi từ:\n"
        "    ca            823/873  94%   idf 0.06   <- vo dung\n"
        "    bat trang     436/873  50%   idf 0.69\n"
        "    man hinh      384/873  44%   idf 0.82\n"
        "    can            143/873  16%   idf 1.81\n"
        "    can dien tu     26/873   3%   idf 3.51   <- phan biet duoc\n"
        "    con so          13/873   1.5% idf 4.21   <- phan biet duoc\n"
        "Truy vấn cá/cân của gold L21_V023 luôn thua L29_V015 — một phóng sự "
        "thuỷ sản có 444 keyframe và 147 lần chữ 'cá', gấp đôi gold. Cạnh "
        "tranh bằng từ phổ thông thì video nhắc nhiều thắng, bất kể đúng sai. "
        "v2 yêu cầu ưu tiên danh từ ghép/cụ thể ('cân điện tử' thay vì 'cân') "
        "va bo tu chung chung.\n\n"
        "v3 - SUA HAI LOI DO DUOC tren gold L30_V078 frame 1788.\n"
        "(a) visual_vi ta nham canh. Truy van neu chi tiet chi doc duoc "
        "khi quay CAN CANH (chu tren giay), v2 ta luon canh can canh do. "
        "Nhung gold la frame HANH DONG (nguoi phu nu dang cam to giay co "
        "noi dung chu viet), con noi dung 200 gr nam o frame KHAC cach 60 "
        "frame. Khung hinh hanh dong va khung hinh noi dung la HAI khung "
        "hinh khac nhau.\n"
        "(b) ocr_terms chep nguyen cach viet cua truy van. Truy van ghi "
        "200g thit nac xay, OCR that ghi Nac dam xay 200 gr - lech ca don "
        "vi (g so voi gr) lan chinh ta (OCR doc sai). Khop chuoi nen truot "
        "hoan toan. v3 yeu cau sinh MOI cach viet va tach so khoi chu."
    ),
    template="""Bạn chuẩn bị dữ liệu tìm kiếm cho một hệ thống truy hồi video.
Hệ thống có 4 công cụ, MỖI CÔNG CỤ CẦN MỘT LOẠI DỮ LIỆU KHÁC NHAU. Nhiệm vụ
của bạn là soạn đúng loại dữ liệu cho từng công cụ.

Truy vấn gốc: {query}
Loại nhiệm vụ: {task}

NGUYÊN TẮC QUAN TRỌNG NHẤT — CHỌN TỪ PHÂN BIỆT ĐƯỢC, KHÔNG PHẢI TỪ ĐÚNG:

Kho có hàng trăm video. Một từ xuất hiện ở hầu hết video thì dù mô tả đúng
cảnh vẫn VÔ DỤNG để tìm — nó chỉ đẩy lên những video nhắc từ đó nhiều nhất,
chứ không phải video chứa đúng khoảnh khắc cần tìm.

Với mỗi từ định dùng, hãy tự hỏi: "bao nhiêu video trong kho có thứ này?"
  - Gần như video nào cũng có  -> BỎ, hoặc thay bằng dạng cụ thể hơn
  - Chỉ một số ít video có     -> GIỮ, đây là thứ giúp tìm ra

Cách làm cụ thể:
  - Ưu tiên DANH TỪ GHÉP thay vì danh từ trần:
        "cân điện tử" tốt hơn "cân";  "màn hình LED" tốt hơn "màn hình"
  - Ưu tiên chi tiết BẤT THƯỜNG của cảnh — thứ hiếm gặp ở video khác
  - BỎ các từ mô tả đúng nhưng có ở mọi nơi: "hình ảnh", "cảnh quay",
    "màu sắc", "người", và các danh từ phổ thông không kèm định ngữ
  - Một cảnh có 2-3 chi tiết hiếm thì tốt hơn một câu đầy đủ toàn từ chung

Soạn các trường sau:

"visual_vi" — cho công cụ so ẢNH với CÂU (CLIP).
  Nó so câu của bạn với NỘI DUNG NHÌN THẤY của MỘT khung hình đứng yên.
  Hãy viết MỘT câu mô tả khung hình đó TRÔNG NHƯ THẾ NÀO: vật thể, người,
  hành động, màu sắc, vị trí tương đối. Viết như đang chú thích một tấm ảnh.
  BỎ: câu hỏi, thứ tự thời gian ("sau đó", "cuối cùng"), suy luận.

  QUAN TRỌNG — mô tả cảnh NHÌN THẤY ĐƯỢC, không phải cảnh chứa đáp án.
  Nhiều truy vấn nêu một chi tiết chỉ đọc được khi quay CẬN CẢNH (chữ trên
  giấy, số trên màn hình). Khung hình cho thấy HÀNH ĐỘNG (một người cầm tờ
  giấy) và khung hình cho thấy NỘI DUNG (cận cảnh tờ giấy) là HAI khung hình
  khác nhau, thường cách nhau vài giây.

  Hãy mô tả khung hình HÀNH ĐỘNG — người đang làm gì, cầm gì, ở đâu — chứ
  ĐỪNG mô tả nội dung chi tiết mà chỉ cận cảnh mới thấy. Nội dung đó thuộc
  về "ocr_terms".
      truy vấn: "người cầm công thức ghi 200g thịt nạc xay"
      visual_vi ĐÚNG : "một người đang cầm tờ giấy có chữ viết, ngồi ở bàn"
      visual_vi SAI  : "cận cảnh tờ công thức ghi 200g thịt nạc xay"

"visual_en" — bản tiếng Anh của "visual_vi". Cụm mô tả ảnh, không phải câu
  dịch máy móc.

"caption_vi" — cho BM25 trên kho caption TIẾNG VIỆT.
  Là khớp TỪ nên đây là nơi nguyên tắc độ hiếm quan trọng nhất: một từ phổ
  thông ở đây sẽ kéo lên những video nhắc từ đó hàng trăm lần.
  Chỉ cho các danh từ ghép / cụm cụ thể phân biệt được cảnh này, kèm tối đa
  1 cách gọi khác cho mỗi khái niệm. Không cần đúng ngữ pháp.
  Thà cho 3 cụm hiếm còn hơn 10 từ mà đa số là từ chung.

"ocr_terms" — cho công cụ đọc CHỮ HIỆN TRÊN MÀN HÌNH.
  Chỉ liệt kê chuỗi ký tự thật sự có thể nhìn thấy trên hình: chữ trong ngoặc
  kép của truy vấn, đơn vị đo, nhãn, biển hiệu.
  Nếu cảnh không có lý do gì để chứa chữ, trả mảng RỖNG. Đừng nhét mô tả
  thị giác vào đây — nó chỉ tạo nhiễu.

  ĐỪNG chép nguyên cách viết của truy vấn — hãy cho MỌI CÁCH VIẾT mà chữ đó
  có thể xuất hiện trên màn hình. Người viết truy vấn và người làm phụ đề
  hiếm khi viết giống nhau, và OCR khớp theo chuỗi nên lệch một ký tự là trượt:
      "200g thịt nạc xay"
        -> "200 gr", "200gr", "200 g", "200g", "nạc xay", "thịt xay"
      "5kg"    -> "5 kg", "5kg", "5 kilogram"
      "30%"    -> "30 %", "30%", "30 phần trăm"
  Tách RIÊNG phần số kèm đơn vị và phần chữ, đừng gộp thành một cụm dài —
  cụm dài gần như không bao giờ khớp đúng nguyên văn.

"asr_vi" — cho công cụ tìm trong LỜI NÓI.
  Viết theo cách một người dẫn hoặc nhân vật sẽ NÓI RA nội dung này.
  Nếu đáp án không thể nằm trong lời nói, trả chuỗi rỗng.

"events" — nếu truy vấn mô tả nhiều khoảnh khắc NỐI TIẾP nhau, tách thành
  danh sách mô tả thị giác theo đúng thứ tự (mỗi phần tử viết như "visual_vi").
  Truy vấn một cảnh thì trả mảng rỗng.

"answer_type" — một trong: numeric, text, color, object, action, location,
  person, time, unknown.

Chỉ trả về JSON đúng khuôn sau, không thêm chữ nào khác:

{{"visual_vi": "...", "visual_en": "...", "caption_vi": "...",
"ocr_terms": ["..."], "asr_vi": "...", "events": ["..."],
"answer_type": "..."}}""",
)


PROMPTS: dict[str, PromptSpec] = {
    spec.prompt_id: spec
    for spec in (
        TRANSLATE_QUERY,
        EXPAND_QUERY,
        PREPARE_QUERY_BUNDLE,
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
    "PREPARE_QUERY_BUNDLE",
    "PROMPTS",
    "PromptSpec",
    "QA_ANSWER",
    "RECOMMEND_WEIGHTS",
    "SELECT_EVIDENCE",
    "TRANSLATE_QUERY",
    "VLM_RERANK",
    "prompts_by_role",
]
