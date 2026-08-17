"""Đổi truy vấn tiếng Việt thành nhãn vật thể tiếng Anh cho nhánh `bm25_object`.

Nhánh đó index `objects[].label` của Open Images — **tiếng Anh**, từ vựng đóng
514 chuỗi trên corpus hiện tại. Truy vấn thi đấu là tiếng Việt, nên trước bước
này nó khớp đúng 0 token: đo trên 96/120 truy vấn gold, tỉ lệ từ truy vấn xuất
hiện trong `object` của cảnh đúng là **0,0% ở mọi task**.

Không phải nhánh hỏng — dữ liệu vẫn đúng. Kiểm chứng: "xe đạp và người đi đường"
cho 0 ứng viên, "Bicycle Person Street" cho 100.

`AIC_ENABLE_QUERY_TRANSLATION` không giải quyết được: nó bọc text encoder của
nhánh dense, các nhánh BM25 vẫn nhận nguyên tiếng Việt.

Tra bảng, KHÔNG gọi LLM lúc truy vấn: ánh xạ này là một hàm hằng trên một từ
vựng đóng, dựng sẵn bằng `scripts/build_object_lexicon.py`. Gọi LLM mỗi truy vấn
để sinh lại cùng một kết quả là trả phí lặp, cộng độ trễ và một phụ thuộc mạng
vào đường nóng.

KHỚP CÓ DẤU LÀ MẶC ĐỊNH. Bản đầu bỏ dấu như
`online/services/lexical_coverage.py` để người gõ không dấu vẫn khớp, và nó sinh
ra nhãn SAI vì dấu tiếng Việt là âm vị chứ không phải trang trí:

    buổi (sáng)  -> buoi = bưởi  -> Grapefruit
    mặt (đất)    -> mat  = mắt   -> Human eye
    có           -> co   = cờ    -> Flag

"bản tin thời sự buổi sáng" khi ấy cho ra `Grapefruit`. Không có gì báo lỗi —
nhánh vẫn trả ứng viên, chỉ là ứng viên của một truy vấn khác hẳn.

Nên: khớp có dấu; chỉ khi CHÍNH truy vấn không có dấu nào mới rơi về bảng bỏ
dấu. Người gõ không dấu vẫn được phục vụ, mà người gõ đúng chính tả không bị
bảng không dấu làm hỏng.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

LEXICON_PATH = Path(__file__).resolve().parents[1] / "resources" / "object_lexicon_vi.json"

# Ranh giới từ: khớp cụm phải trùng nguyên từ, không phải chuỗi con. Không có nó
# thì "voi" (Elephant) khớp vào "với", và "bo" (Cattle) khớp vào "bỏ" — hai lỗi
# này đã gặp thật khi thử bản khớp bằng `in`.
_WORD = re.compile(r"[0-9a-zà-ỹ]+")
_MARK = re.compile(r"[̀-ͯ]")


def _fold(text: str) -> str:
    """Hạ chữ + gom khoảng trắng, GIỮ dấu."""

    return " ".join(_WORD.findall(unicodedata.normalize("NFC", text.casefold())))


def _strip(text: str) -> str:
    """Như `_fold` nhưng bỏ dấu — chỉ dùng cho truy vấn vốn đã không dấu."""

    decomposed = unicodedata.normalize("NFD", text.casefold())
    plain = _MARK.sub("", decomposed).replace("đ", "d")
    return " ".join(re.findall(r"[0-9a-z]+", plain))


def _has_diacritics(text: str) -> bool:
    decomposed = unicodedata.normalize("NFD", text)
    return bool(_MARK.search(decomposed)) or "đ" in text.casefold()


class ObjectQueryTransform:
    """Truy vấn tiếng Việt -> chuỗi nhãn tiếng Anh khớp được.

    Trả CHUỖI RỖNG khi không nhận ra vật thể nào. Đó là câu trả lời đúng: nhánh
    khi ấy báo `empty` chứ không đoán bừa một nhãn phổ biến, và `empty` là trạng
    thái hợp lệ mà fusion đã biết cách bỏ qua.
    """

    def __init__(self, accented: dict[str, list[str]], stripped: dict[str, list[str]]) -> None:
        # Cụm dài khớp trước: "xe cứu thương" phải ra Ambulance chứ không dừng ở
        # "xe" (Vehicle).
        self._accented = (sorted(accented, key=lambda p: -len(p.split())), accented)
        self._stripped = (sorted(stripped, key=lambda p: -len(p.split())), stripped)

    @classmethod
    def load(cls, path: Path | None = None) -> "ObjectQueryTransform | None":
        """`None` khi chưa có từ điển — caller giữ nguyên hành vi cũ.

        Thiếu file KHÔNG phải lỗi dừng: nhánh vẫn chạy như trước (khớp 0 với
        truy vấn tiếng Việt), chỉ là không tốt lên. Dừng hẳn ở đây sẽ biến một
        tính năng phụ thành điều kiện khởi động.
        """

        target = path or LEXICON_PATH
        if not target.exists():
            return None
        raw = json.loads(target.read_text(encoding="utf-8")).get("labels") or {}
        accented: dict[str, list[str]] = {}
        stripped: dict[str, list[str]] = {}
        for label, phrases in raw.items():
            for phrase in phrases:
                for table, key in ((accented, _fold(phrase)), (stripped, _strip(phrase))):
                    if not key:
                        continue
                    table.setdefault(key, [])
                    if label not in table[key]:
                        table[key].append(label)
        return cls(accented, stripped) if accented else None

    def __call__(self, query: str) -> str:
        phrases, table = self._accented if _has_diacritics(query) else self._stripped
        normalize = _fold if _has_diacritics(query) else _strip
        haystack = f" {normalize(query)} "
        labels: list[str] = []
        for phrase in phrases:
            if f" {phrase} " in haystack:
                for label in table[phrase]:
                    if label not in labels:
                        labels.append(label)
        return " ".join(labels)


__all__ = ["ObjectQueryTransform"]
