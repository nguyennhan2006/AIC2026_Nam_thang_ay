"""DIAG-01 — một nguồn duy nhất để lấy câu truy vấn từ một bản ghi gold.

Vì sao cần tách ra thành module công khai: gold ghi câu hỏi dưới BA bộ khoá
khác nhau tuỳ task, đo được trên `examples/gold_all3.jsonl`::

    36 bản ghi  query_vi, query_en, dense_query_en     (KIS)
    36 bản ghi  question_vi, question_en               (QA)
    48 bản ghi  query_vi, query_en                     (TRAKE, AVS)

`scripts/eval_tasks.py` xử lý đúng chuyện này từ đầu, nhưng nó giữ hàm ở dạng
private. Hệ quả thật: một script chẩn đoán viết vội chỉ đọc `query_vi` đã bỏ
sót toàn bộ 36 truy vấn QA — trong đó có `V001_VQA_M03`, đúng truy vấn mang
mệnh đề phủ định nguy hiểm nhất. Nó không báo lỗi, chỉ lặng lẽ đếm thiếu, nên
kết luận "0 truy vấn bị ảnh hưởng" trông vẫn hợp lệ.

Đó là kiểu hỏng tệ nhất cho một công cụ audit: không sai ở phép tính, sai ở
tập được đếm. Nên hàm này **không bao giờ trả chuỗi rỗng** — thiếu khoá là
`KeyError`, và trả kèm tên khoá đã dùng để bên gọi ghi lại được.
"""

from __future__ import annotations

from typing import NamedTuple

# Thứ tự ưu tiên cho từng task. QA dùng bộ khoá riêng và cần ghép mô tả sự
# kiện với câu hỏi — hỏi "Phương tiện nào bay trên không?" mà không kèm bối
# cảnh sự kiện thì không đủ để tìm.
_QA_CONTEXT_KEYS = ("event_description_vi", "event_description_en")
_QA_QUESTION_KEYS = ("question_vi", "question_en")
_GENERIC_KEYS = ("query_vi", "query_en", "raw_query")


class GoldText(NamedTuple):
    """Câu truy vấn kèm XUẤT XỨ của nó."""

    text: str
    #: Các khoá đã thực sự đóng góp, theo thứ tự ghép. Ghi lại được để một bản
    #: audit nói rõ nó đã đọc gì, thay vì để người đọc phải đoán.
    source_keys: tuple[str, ...]


def resolve_gold_text_detailed(record: dict) -> GoldText:
    """Câu truy vấn tiếng Việt của một bản ghi gold, kèm khoá đã dùng.

    Ném `KeyError` nếu không khoá nào dùng được. Trả chuỗi rỗng ở đây sẽ khiến
    truy vấn biến mất khỏi tập đo mà không ai biết.
    """

    for key in _GENERIC_KEYS:
        value = record.get(key)
        if value and str(value).strip():
            return GoldText(str(value).strip(), (key,))

    used: list[str] = []
    parts: list[str] = []
    for group in (_QA_CONTEXT_KEYS, _QA_QUESTION_KEYS):
        for key in group:
            value = record.get(key)
            if value and str(value).strip():
                parts.append(str(value).strip())
                used.append(key)
                break

    if parts:
        return GoldText(" ".join(parts), tuple(used))

    raise KeyError(
        f"gold {record.get('query_id')!r} (task={record.get('task')!r}) không có "
        f"trường truy vấn nào dùng được; đã thử "
        f"{list(_GENERIC_KEYS + _QA_CONTEXT_KEYS + _QA_QUESTION_KEYS)}"
    )


def resolve_gold_text(record: dict) -> str:
    """`resolve_gold_text_detailed` nhưng chỉ trả phần văn bản."""

    return resolve_gold_text_detailed(record).text


__all__ = ["GoldText", "resolve_gold_text", "resolve_gold_text_detailed"]
