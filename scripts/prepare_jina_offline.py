"""Nạp sẵn CODE mô hình jina vào cache HuggingFace để chạy được HF_HUB_OFFLINE=1.

Chạy MỘT LẦN trên máy còn mạng, trước khi khởi động backend:

    python -m scripts.prepare_jina_offline              # vá + tải + kiểm
    python -m scripts.prepare_jina_offline --verify-only  # chỉ kiểm, không mạng

Ba việc: vá `config.json`, kéo repo code về cache, và kiểm venv có đủ gói mà
code đó `import` (`einops`, `timm`, `torchvision`... — xem `missing_dependencies`).

## Vì sao cần

`storage/models/jina-clip-v2/config.json` khai:

    "auto_map": {"AutoModel": "jinaai/jina-clip-implementation--modeling_clip.JinaCLIPModel"}

Dấu `--` nghĩa là code mô hình nằm ở **repo KHÁC** trên HuggingFace. Dù thư mục
model đã nằm sẵn trên đĩa, `transformers` vẫn gọi `cached_file()` lên repo đó để
lấy file `.py`; `HF_HUB_OFFLINE=1` biến lời gọi ấy thành:

    OSError: We couldn't connect to 'https://huggingface.co' ... and couldn't
    find them in the cached files.

`jina-embeddings-v3` (text tower) cũng vậy, qua `jinaai/xlm-roberta-flash-implementation`.

## Vì sao chép `transformers_modules/` KHÔNG đủ

Gói chia dữ liệu (`04_hf_modules.zip`) chỉ có `~/.cache/huggingface/modules/`.
Đó là bản SAO DẪN XUẤT mà `transformers` tự sinh SAU khi `cached_file()` thành
công — đọc `dynamic_module_utils.get_cached_module_file`: `cached_file()` chạy
TRƯỚC, và nó chỉ nhìn `$HF_HOME/hub`. Thiếu `hub/models--jinaai--*-implementation`
thì có `modules/` cũng chết. Đúng hai repo code này, tổng ~350 KB.
"""

from __future__ import annotations

# Phải xoá cờ offline TRƯỚC khi import huggingface_hub: thư viện đọc hai biến
# này lúc import và đóng băng thành hằng số, đặt lại sau đó không có tác dụng.
# Script này tồn tại để TẢI, nên chạy nó trong shell đã export cờ offline (đúng
# shell thi đấu) mà vẫn phải tải được.
import os

for _flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(_flag, None)

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Script này hay được gọi trong shell TRẦN (trước khi `run_competition.*` đặt
# PYTHONIOENCODING), mà console Windows mặc định là cp1258 — in tiếng Việt vào
# đó ném UnicodeEncodeError và giết script trước khi nó kịp làm gì.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # stream bị thay thế / không hỗ trợ
        pass

ROOT = Path(__file__).resolve().parents[1]

# Thư mục model local cần code từ xa. Kiểm theo cả hai vì text tower hỏng thì
# lỗi nổ lúc DỰNG MODEL, muộn hơn và khó đọc hơn lỗi của clip.
MODEL_DIRS = ("jina-clip-v2", "jina-embeddings-v3")


def log(message: str) -> None:
    print(f"[jina-offline] {message}", flush=True)


def hf_home() -> Path:
    """Trên Vast.ai `HF_HOME` hay bị trỏ sang volume bền vững — không hard-code.

    Cùng lý do đã ghi ở `scripts/bootstrap_vast_from_kaggle.hf_modules_root`.
    """

    home = os.environ.get("HF_HOME", "").strip()
    return Path(home) if home else Path.home() / ".cache" / "huggingface"


def code_repos(root: Path) -> list[str]:
    """Đọc `auto_map` của từng model local, trả về repo id đứng trước `--`."""

    repos: list[str] = []
    for name in MODEL_DIRS:
        config_path = root / "storage" / "models" / name / "config.json"
        if not config_path.exists():
            log(f"[bỏ qua] thiếu {config_path}")
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for reference in (config.get("auto_map") or {}).values():
            if "--" not in str(reference):
                continue
            repo = str(reference).split("--", 1)[0]
            if repo not in repos:
                repos.append(repo)
    return repos


def snapshots(repos: list[str], *, download: bool) -> dict[str, Path | None]:
    """Trả về đường dẫn snapshot của từng repo code, tải về nếu được phép.

    Chỉ lấy `.py` và `.json`: repo implementation không có trọng số, nhưng lọc
    tường minh để một ngày nào đó họ thêm file lớn thì script không âm thầm kéo
    về vài GB giữa lúc chuẩn bị thi.
    """

    from huggingface_hub import snapshot_download

    resolved: dict[str, Path | None] = {}
    for repo in repos:
        try:
            path = snapshot_download(repo, allow_patterns=["*.py", "*.json"],
                                     local_files_only=not download)
            resolved[repo] = Path(path)
            log(f"[ok] {repo} -> {path}")
        except Exception as exc:  # noqa: BLE001 - in nguyên nhân rồi đi tiếp repo sau
            resolved[repo] = None
            log(f"[LỖI] {repo}: {exc}")
    return resolved


# Tên module lúc `import` khác tên lúc `pip install` — in nhầm là người chạy gõ
# `pip install PIL` rồi bí. Chỉ liệt kê những cái thật sự lệch.
PIP_NAME = {"PIL": "pillow", "cv2": "opencv-python-headless", "sklearn": "scikit-learn"}


def missing_dependencies(paths: dict[str, Path | None]) -> list[str]:
    """Những gói mà CODE MÔ HÌNH cần nhưng venv chưa có.

    `transformers.dynamic_module_utils.check_imports` quét `import` của file `.py`
    vừa tải và ném `ImportError` khi thiếu — mỗi lần một gói, mỗi lần tốn một
    lượt khởi động 4 phút (`einops`, rồi `timm`, rồi `torchvision`...). Quét
    TOÀN BỘ file ở đây để trả về một dòng `pip install` duy nhất.

    Dùng chính `get_imports` của transformers chứ không tự parse: nó đã bỏ các
    `import` nằm trong `try/except` (flash_attn, xformers — code jina cố ý cho
    phép thiếu), tự viết lại là chắc chắn lệch.
    """

    import importlib.util

    from transformers.dynamic_module_utils import get_imports

    needed: set[str] = set()
    for path in paths.values():
        if path is None:
            continue
        for module_file in sorted(path.rglob("*.py")):
            try:
                needed.update(get_imports(str(module_file)))
            except Exception as exc:  # noqa: BLE001 - file lạ không được chặn cả lượt kiểm
                log(f"[cảnh báo] không đọc được import của {module_file.name}: {exc}")

    missing = []
    for module in sorted(needed):
        if importlib.util.find_spec(module) is None:
            missing.append(PIP_NAME.get(module, module))
    return missing


_VERIFY_SNIPPET = """
import sys
from transformers import AutoConfig
AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
"""


def verify(root: Path) -> int:
    """Nạp `AutoConfig` trong tiến trình con ĐANG BẬT cờ offline.

    Phải là tiến trình CON: tiến trình này đã xoá cờ offline ở đầu file, tự kiểm
    trong nó thì luôn xanh dù cache rỗng — đúng kiểu kiểm tra vô dụng. `AutoConfig`
    thay vì `AutoModel` vì nó đi qua đúng đường `get_cached_module_file` đang
    hỏng mà không phải đọc 1,7 GB trọng số.
    """

    environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    failures = 0
    for name in MODEL_DIRS:
        directory = root / "storage" / "models" / name
        if not (directory / "config.json").exists():
            continue
        result = subprocess.run(
            [sys.executable, "-c", _VERIFY_SNIPPET, str(directory)],
            cwd=root, env=environment, capture_output=True, text=True,
        )
        if result.returncode == 0:
            log(f"[ok] offline nạp được {name}")
        else:
            failures += 1
            tail = (result.stderr or "").strip().splitlines()[-3:]
            log(f"[LỖI] offline KHÔNG nạp được {name}:")
            for line in tail:
                print(f"        {line}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT), help=f"gốc repo. Mặc định {ROOT}")
    parser.add_argument("--verify-only", action="store_true",
                        help="không tải gì, chỉ kiểm cache hiện có")
    arguments = parser.parse_args()
    root = Path(arguments.root).expanduser().resolve()

    log(f"gốc repo : {root}")
    log(f"HF_HOME  : {hf_home()}")

    # Vá config.json (text tower -> bản local) dùng lại đúng hàm của bootstrap:
    # tải lại model là ghi đè, nên cái vá này phải chạy mỗi lần chuẩn bị máy.
    from scripts.bootstrap_vast_from_kaggle import patch_jina_config

    patch_jina_config(root)

    repos = code_repos(root)
    if not repos:
        log("Không thấy auto_map trỏ repo ngoài — không có gì để tải.")
    else:
        log("repo code cần có trong cache: " + ", ".join(repos))

    paths = snapshots(repos, download=not arguments.verify_only) if repos else {}
    if any(path is None for path in paths.values()):
        # Ba nguyên nhân đã gặp thật: máy không ra được huggingface.co; trên
        # Windows là `WinError 1314` (cache hub dựng bằng symlink, cần Developer
        # Mode); và `--verify-only` trên máy chưa từng tải.
        log("Thiếu repo code trong cache. Xem lỗi bên trên: không ra được "
            "huggingface.co, hay WinError 1314 (bật Developer Mode trên Windows)?")
        return 1

    missing = missing_dependencies(paths)
    if missing:
        log("Code mô hình cần các gói sau mà venv chưa có:")
        log(f"    pip install {' '.join(missing)}")
        log("Cài xong chạy lại script này.")
        return 1

    if verify(root):
        log("VẪN HỎNG. Máy không ra được mạng thì chép từ máy đã chạy được:")
        for repo in repos:
            log(f"  {hf_home()}/hub/models--{repo.replace('/', '--')}")
        return 1

    log("Xong. Khởi động được với HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
