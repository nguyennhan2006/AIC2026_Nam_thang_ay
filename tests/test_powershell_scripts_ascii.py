"""`scripts/*.ps1` phải thuần ASCII.

Windows PowerShell 5.1 — bản có sẵn trên mọi máy Windows, và là bản chạy khi gõ
`powershell -File ...` — đọc file `.ps1` KHÔNG có BOM bằng codepage ANSI của máy
(cp1258 trên máy dự án), không phải UTF-8. Byte UTF-8 của `—` (E2 80 94) khi đó
giải thành `â€”`, mà 0x94 là dấu nháy kép cong U+201D — và PowerShell 5.1 CHẤP
NHẬN nháy cong làm dấu mở/đóng chuỗi. Một em dash trong `Write-Host "..."` đóng
chuỗi giữa chừng và làm hỏng cú pháp cả file:

    run_ui.ps1:20 Missing closing '}' in statement block or type definition.

Lỗi báo ở dòng cách chỗ sai cả chục dòng, nên rất tốn thời gian lần. Đã xảy ra
thật với `scripts/run_ui.ps1` (em dash ở dòng 21 và 31).

Cách chữa khác là lưu kèm BOM, nhưng cả hai script đã viết tiếng Việt không dấu
sẵn — giữ luật ASCII đơn giản hơn và không phụ thuộc editor có ghi BOM hay không.
"""

from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.ps1"))


def test_ps1_scripts_are_discovered():
    assert SCRIPTS, "không tìm thấy scripts/*.ps1 — test này mất tác dụng trong im lặng"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_ps1_script_is_pure_ascii(script: Path):
    text = script.read_text(encoding="utf-8")
    offenders = [
        (number, line.strip())
        for number, line in enumerate(text.splitlines(), 1)
        if any(ord(character) > 127 for character in line)
    ]
    assert not offenders, (
        f"{script.name} có ký tự ngoài ASCII — PowerShell 5.1 đọc file này bằng "
        f"codepage ANSI và có thể hỏng cú pháp:\n"
        + "\n".join(f"  dòng {number}: {line}" for number, line in offenders)
    )
