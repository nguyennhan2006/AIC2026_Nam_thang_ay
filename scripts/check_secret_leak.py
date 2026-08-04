"""Chặn commit lộ secret — Gate 0 của
AIC2026_FPT_API_SINGLE_VIDEO_TEST_TUNING_GUIDE.md §6.2/§27.

Quét đúng nội dung sắp được commit (staged diff), không quét toàn bộ working
tree — file `.env` thật vẫn nằm trên máy dev bình thường, chỉ cấm nó **lọt
vào commit**. Chạy trước khi `git commit`:

    python -m scripts.check_secret_leak

Thoát code 1 và in rõ file+dòng nghi vấn nếu tìm thấy; không tự động sửa hay
xóa gì — quyết định là của người commit.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Tên file tuyệt đối không được xuất hiện trong staged diff (chứa secret thật).
FORBIDDEN_STAGED_FILES = re.compile(r"^\.env(\.[\w.-]+)?\.local$|^\.env$")

# File CHỨA CHỦ Ý chuỗi trông giống secret để test chính bộ pattern ở trên —
# không phải secret thật. Không loại trừ theo path thì chính test suite của
# script này sẽ tự chặn commit của nó (giống bandit/gitleaks đều tự loại trừ
# test suite của chúng khỏi scan của chính chúng).
SELF_TEST_EXEMPT_FILES = frozenset({"tests/test_check_secret_leak.py"})

# Pattern giá trị trông như secret thật — không khớp placeholder kiểu
# "change-me"/""/"<SECRET>" mà .env.example dùng.
SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Không đặt \b trước nhóm tên biến: "AIC_FPT_API_KEY" có "_" ngay trước
    # "API_KEY" — "_" là ký tự \w nên \b sẽ KHÔNG khớp ở đó (không có ranh
    # giới word giữa hai ký tự \w), làm regex im lặng bỏ sót đúng biến hay
    # gặp nhất trong .env.
    ("generic_api_key_assignment", re.compile(
        r"(?i)(API[_-]?KEY|SECRET|TOKEN|PASSWORD)\s*=\s*['\"]?(?!change-me|<[^>]*>|\s*$)([A-Za-z0-9_\-./+=]{12,})"
    )),
    ("bearer_token", re.compile(r"(?i)Authorization['\"]?\s*[:=]\s*['\"]?Bearer\s+[A-Za-z0-9_\-.]{12,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


def _run_git(args: list[str]) -> str:
    """`subprocess.run(text=True)` decode theo locale mặc định của máy — trên
    Windows đó thường là một codepage (vd cp1258), không phải UTF-8. Diff của
    repo này chứa tiếng Việt UTF-8 nên phải tự chỉ định `encoding="utf-8"`,
    nếu không script vỡ ngay trên chính máy dev đang cần nó nhất."""

    result = subprocess.run(
        ["git", *args], capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    return result.stdout


def staged_files() -> list[str]:
    output = _run_git(["diff", "--cached", "--name-only"])
    return [line for line in output.splitlines() if line.strip()]


def staged_diff() -> str:
    return _run_git(["diff", "--cached", "-U0"])


def check_forbidden_filenames(files: list[str]) -> list[str]:
    return [f for f in files if FORBIDDEN_STAGED_FILES.match(f.rsplit("/", 1)[-1])]


def check_secret_patterns(diff: str) -> list[str]:
    """Chỉ quét dòng THÊM MỚI (bắt đầu bằng `+`, không phải `+++` header)."""

    issues: list[str] = []
    current_file = "?"
    skip_current_file = False
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[len("+++ b/"):]
            skip_current_file = current_file in SELF_TEST_EXEMPT_FILES
            continue
        if skip_current_file:
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:]
        for name, pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(added):
                issues.append(f"{current_file}: nghi vấn {name} — {added.strip()[:80]}")
    return issues


def main() -> None:
    files = staged_files()
    problems = [f"file bị cấm commit: {f}" for f in check_forbidden_filenames(files)]
    problems += check_secret_patterns(staged_diff())
    if problems:
        print("SECRET LEAK CHECK: FAIL", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(1)
    print(f"SECRET LEAK CHECK: OK ({len(files)} file staged, không phát hiện vấn đề)")


if __name__ == "__main__":
    main()
