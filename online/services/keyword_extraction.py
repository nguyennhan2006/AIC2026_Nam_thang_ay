"""Tách keyword từ TRUY VẤN — để nhánh `bm25_keyword` khớp đúng hạt.

Vấn đề đo được: hai phía đang lệch hạt hoàn toàn.

- **Tài liệu**: `scene.keywords` là 1651 mục, **100% từ nhãn object** — cụm 2–3
  từ như `"người dẫn chương trình"`, `"biển"`.
- **Truy vấn**: `bm25_keyword` nhận NGUYÊN câu, ví dụ *"Tìm video về giếng nước
  bất ngờ phun cao và căn chỉnh bốn khoảnh khắc: (1) cột nước được quay từ
  xa; ..."*

BM25 khớp một câu kể chuyện với một túi nhãn ngắn: phần lớn token của truy vấn
không thể khớp gì, còn các token phổ biến (`cảnh`, `người`, `nước`) thì khớp
bừa. ROUTE-01 đã đo được nền nhiễu này: một scene NGẪU NHIÊN trùng sẵn trung
bình 2.25 token OCR và 6.75 token ASR với truy vấn.

Bộ tách ở đây đưa phía truy vấn về đúng hạt của phía tài liệu.

Có ground truth để chấm: 12 truy vấn KIS trong gold có `sparse_terms` do người
gán (`"cảnh báo sạt lở nguy hiểm | riverbank erosion warning | Cần Thơ"`), nên
đo được bộ tách TRỰC TIẾP mà không phải chạy cả pipeline — xem
`scripts/eval_keyword_extraction.py`.
"""

from __future__ import annotations

import re

from online.adapters.ocr_fuzzy import normalize_vi

# Một nguồn duy nhất. Trước đây có BA bản rời rạc (avs.py 26 từ, kis.py 72,
# lexical_coverage.py 82) — cùng mục đích, khác nội dung, và không có gì giữ
# chúng đồng bộ. Đây là hợp của cả ba, viết KHÔNG DẤU vì mọi chỗ dùng đều so
# sau khi qua `normalize_vi`.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and any are as at be been but by for from has have in is it its of on or
    that the their then there these this to was were what when where which who
    will with showing find scenes segments video clip shot
    ai bang bao bi boi ca cac cai canh cho chua chung co con cua cung cuoi da dang
    day den deu di do doan duoc duoi gi gia gio hay hoac khi khong la lai len
    luc mot na nao nay nen ngay nhieu nhung no noi nua o phai qua ra rat roi sau
    se so su ta tai the theo thi tim toi tren tu tuy va vao vay ve voi vua vi xong
    y
    """.split()
)

# Động từ mở đầu của câu truy vấn thi đấu. Chúng mô tả THAO TÁC tìm kiếm chứ
# không mô tả nội dung cảnh, nên giữ lại chỉ làm loãng.
_LEAD_VERBS = re.compile(
    r"^\s*(tìm|tim|hãy tìm|hay tim|cho tôi|cho toi|xác định|xac dinh)\s+"
    r"(cảnh|canh|đoạn|doan|video|hiện trường|hien truong|phóng sự|phong su|"
    r"khoảnh khắc|khoanh khac|hình ảnh|hinh anh|clip)?\s*(có|co|về|ve|mà|ma)?\s*",
    re.IGNORECASE,
)

# Từ chỉ "một cảnh" chứ không chỉ nội dung — khác STOPWORDS ở chỗ chúng là
# danh từ đầy đủ nghĩa, chỉ vô dụng trong NGỮ CẢNH này.
SCENE_NOUNS = frozenset(
    "canh doan video clip hinh anh khoanh khac phong su hien truong dong bang".split()
)

_QUOTED = re.compile(r"[\"“”'']([^\"“”'']{2,80})[\"“”'']")
# Tách theo dấu câu VÀ liên từ: mỗi mệnh đề thường là một ý riêng, và giữ
# chúng riêng cho ra cụm ngắn đúng hạt hơn là cắt theo cửa sổ n-gram cố định.
_CLAUSE = re.compile(r"[,;.:()\[\]]|\bvà\b|\bva\b|\bhoặc\b|\bhoac\b|\brồi\b|\broi\b|\bsau đó\b|\bsau do\b")
_STEP_MARKER = re.compile(r"\(\s*\d+\s*\)")


def _content_tokens(clause: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-zÀ-ỹ]+", clause)
    kept: list[str] = []
    for token in tokens:
        normalized = normalize_vi(token)
        if not normalized or normalized in STOPWORDS or normalized in SCENE_NOUNS:
            continue
        # Số một chữ số hầu như luôn là chỉ mục bước ("(1)", "bốn khoảnh khắc")
        # chứ không phải nội dung; số dài hơn thì thường là dữ kiện thật
        # ("10 tỷ", "1A", "5 xe máy").
        if normalized.isdigit() and len(normalized) < 2:
            continue
        kept.append(token)
    return kept


class CorpusIdf:
    """IDF tính từ chính corpus đang tìm — để biết cụm nào ĐÁNG giữ.

    Vì sao cần: bản rule đầu tiên cắt 4 từ đầu mỗi mệnh đề và vớ phải chủ ngữ
    chung chung. Truy vấn *"người đàn ông mặc áo đen có nhiều hình xăm ở cánh
    tay"* cho ra `"người đàn ông mặc"`, trong khi keyword người gán là
    `"hình xăm cánh tay cổ"` và `"áo đen"`. Đo được: token_recall 0.374,
    precision 0.243.

    Tiếng Việt đặt danh từ chính trước, nhưng phần PHÂN BIỆT thường nằm ở bổ
    ngữ phía sau. Không suy được vị trí — phải hỏi dữ liệu: token nào hiếm
    trong corpus thì token đó phân biệt.
    """

    def __init__(self, documents: list[str]) -> None:
        import math
        from collections import Counter

        self.total = max(len(documents), 1)
        counts: Counter[str] = Counter()
        for text in documents:
            counts.update(set(normalize_vi(text).split()))
        self.counts = counts
        self._math = math

    def idf(self, token: str) -> float:
        # Smoothed IDF: token chưa từng thấy trong corpus vẫn được điểm cao
        # (nó có thể là tên riêng hiếm), nhưng không vô hạn.
        return self._math.log((self.total + 1) / (self.counts.get(token, 0) + 1)) + 1.0

    def phrase_score(self, phrase: str) -> float:
        """Điểm của một cụm = IDF TRUNG BÌNH, không phải tổng.

        Dùng tổng thì cụm dài luôn thắng, và ta lại quay về gửi cả câu.
        """

        tokens = [t for t in normalize_vi(phrase).split() if t and t not in STOPWORDS]
        if not tokens:
            return 0.0
        return sum(self.idf(t) for t in tokens) / len(tokens)

    @classmethod
    def from_scenes(cls, scenes) -> "CorpusIdf":
        """Dựng từ caption + nhãn keyword của scene — đúng văn bản mà
        `bm25_caption`/`bm25_keyword` index."""

        documents: list[str] = []
        for scene in scenes:
            parts = list(getattr(scene, "captions", []) or [])
            parts.extend(getattr(scene, "keywords", []) or [])
            if parts:
                documents.append(" ".join(parts))
        return cls(documents)


def _ngrams(tokens: list[str], max_words: int) -> list[str]:
    """Mọi cụm liên tiếp dài 1..max_words, giữ thứ tự xuất hiện."""

    out: list[str] = []
    for size in range(min(max_words, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            out.append(" ".join(tokens[start : start + size]))
    return out


def extract_keywords(
    query: str,
    *,
    max_terms: int = 8,
    max_words: int = 4,
    idf: "CorpusIdf | None" = None,
    mode: str = "content",
) -> list[str]:
    """Cụm keyword ngắn, giữ nguyên tiếng Việt có dấu.

    Trả về theo thứ tự ưu tiên: cụm trong ngoặc kép trước (chúng gần như luôn
    là chữ trên màn hình, tức tín hiệu mạnh nhất), rồi tới cụm nội dung theo
    thứ tự xuất hiện.

    Giữ NGUYÊN DẤU: caption và nhãn object trong export đều là tiếng Việt có
    dấu. Bỏ dấu ở đây sẽ khớp kém hơn, còn chuẩn hoá thì để BM25 lo.
    """

    if not query or not query.strip():
        return []

    terms: list[str] = []
    seen: set[str] = set()

    def push(text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        key = normalize_vi(cleaned)
        if not key or key in seen:
            return
        seen.add(key)
        terms.append(cleaned)

    # 1. Cụm trong ngoặc kép — chữ hiện trên màn hình, giữ nguyên văn.
    body = query
    for match in _QUOTED.finditer(query):
        push(match.group(1))
        body = body.replace(match.group(0), " ")

    # 2. Bỏ động từ mở đầu và chỉ mục bước.
    body = _LEAD_VERBS.sub("", body)
    body = _STEP_MARKER.sub(" ", body)

    # 3. Mỗi mệnh đề đóng góp cụm ứng viên. Có IDF thì chọn cụm PHÂN BIỆT
    #    nhất; không có thì lấy `max_words` từ đầu như cũ (kém hơn, nhưng
    #    không phụ thuộc corpus nên dùng được ở test và ở đường offline).
    for clause in _CLAUSE.split(body):
        tokens = _content_tokens(clause)
        if not tokens:
            continue
        if mode == "content":
            # Giữ TOÀN BỘ token nội dung của mệnh đề, mỗi mệnh đề một cụm.
            # Không cắt, không chọn: BM25 đã tự hạ trọng số token phổ biến bằng
            # IDF của chính nó, nên lọc thêm ở đây chỉ làm mất recall. Đo được:
            # cắt 4 từ đầu cho recall 0.374, chọn theo IDF cho 0.244.
            push(" ".join(tokens))
        elif mode == "head" or idf is None:
            push(" ".join(tokens[:max_words]))
        else:
            candidates = _ngrams(tokens, max_words)
            best = max(candidates, key=idf.phrase_score, default=None)
            if best:
                push(best)
                # Cụm tốt thứ hai của cùng mệnh đề, nếu nó không chồng lấn:
                # một mệnh đề dài thường chứa hai ý ("mặc áo đen" + "hình xăm
                # cánh tay") và giữ cả hai là đúng.
                used = set(normalize_vi(best).split())
                rest = [
                    c for c in candidates
                    if not (set(normalize_vi(c).split()) & used)
                ]
                second = max(rest, key=idf.phrase_score, default=None)
                if second and idf.phrase_score(second) > 0:
                    push(second)
        if len(terms) >= max_terms:
            break

    return terms[:max_terms]


def keyword_query(query: str, *, max_terms: int = 8) -> str:
    """Chuỗi đưa vào BM25 — các cụm nối bằng khoảng trắng.

    Rỗng thì trả lại truy vấn gốc: thà khớp thô còn hơn nhánh im lặng trả về 0
    candidate. Xảy ra với truy vấn toàn từ dừng.
    """

    terms = extract_keywords(query, max_terms=max_terms)
    return " ".join(terms) if terms else query


__all__ = ["CorpusIdf", "SCENE_NOUNS", "STOPWORDS", "extract_keywords", "keyword_query"]
