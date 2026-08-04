"""Tải model từ Hugging Face bằng `curl.exe` thay vì stack HTTP của Python.

VÌ SAO CẦN SCRIPT NÀY (máy Windows của dự án)

`huggingface_hub`, `transformers.from_pretrained`, `requests` — tất cả đều
chết với::

    SSLCertVerificationError: Basic Constraints of CA cert not marked critical

Đây KHÔNG phải lỗi mạng và KHÔNG phải thiếu token: `multilingual-e5-large` và
`BAAI/bge-m3` đều là repo công khai. Nguyên nhân là Python 3.13+/OpenSSL 3.x
bật `VERIFY_X509_STRICT` theo mặc định, và nó từ chối handshake nếu BẤT KỲ CA
nào trong chain có `basicConstraints` không đánh dấu `critical`. Windows
Schannel (mà `curl.exe` dùng) không áp check này nên tải bình thường.

`online/adapters/fpt_client.py` vá được vì nó tự dựng `SSLContext`. Không vá
được cho `huggingface_hub` vì urllib3 2.x dựng context ở tầng sâu hơn — đã
thử vá `create_urllib3_context` ở cả `urllib3.util.ssl_` lẫn
`urllib3.connection`, vẫn thất bại.

=> Dùng đường đã chứng minh chạy trên máy này: `curl.exe`.

Chỉ tải nhánh PyTorch. Repo E5 nặng 9.5 GB nhưng phần cần dùng chỉ ~2.3 GB —
phần còn lại là ONNX, OpenVINO, `pytorch_model.bin` (trùng nội dung với
`model.safetensors`) và kết quả benchmark.

Chạy::

    python -m scripts.download_hf_model intfloat/multilingual-e5-large
    python -m scripts.download_hf_model BAAI/bge-m3 --out storage/models/bge-m3

Repo gated (Llama, Gemma...) thì thêm `--token hf_...`; repo công khai KHÔNG
cần token.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HF_API = "https://huggingface.co/api/models"
HF_RESOLVE = "https://huggingface.co"

# Thư mục/hậu tố không cần cho nhánh PyTorch.
SKIP_PREFIXES = (".eval_results/", "onnx/", "openvino/", "coreml/", "tf_model", "flax_model")
SKIP_SUFFIXES = (".onnx", ".onnx_data", ".msgpack", ".h5", ".tflite")


def _curl(args: list[str], *, token: str | None = None) -> subprocess.CompletedProcess:
    command = ["curl.exe", "-L", "--fail", "--retry", "3", "--retry-delay", "2"]
    if token:
        command += ["-H", f"Authorization: Bearer {token}"]
    return subprocess.run(command + args, check=False)


def list_files(repo: str, token: str | None) -> list[dict]:
    target = Path("storage/models/_tmp")
    target.mkdir(parents=True, exist_ok=True)
    listing = target / "tree.json"
    result = _curl(
        ["-s", "--max-time", "60", f"{HF_API}/{repo}/tree/main?recursive=1", "-o", str(listing)],
        token=token,
    )
    if result.returncode != 0 or not listing.exists():
        raise SystemExit(f"không liệt kê được file của {repo} (curl exit {result.returncode})")
    data = json.loads(listing.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("error"):
        raise SystemExit(f"HF trả lỗi cho {repo}: {data['error']}")
    return [item for item in data if item.get("type") == "file"]


def select_files(files: list[dict]) -> list[dict]:
    """Giữ đúng phần cần cho `transformers`/`sentence-transformers`."""

    kept = [
        item
        for item in files
        if not item["path"].startswith(SKIP_PREFIXES)
        and not item["path"].endswith(SKIP_SUFFIXES)
    ]
    # safetensors và .bin là hai bản của CÙNG trọng số — chỉ lấy một.
    has_safetensors = any(item["path"].endswith(".safetensors") for item in kept)
    if has_safetensors:
        kept = [item for item in kept if not item["path"].endswith(".bin")]
    return kept


def size_of(item: dict) -> int:
    return int((item.get("lfs") or {}).get("size") or item.get("size") or 0)


def download(repo: str, out: Path, token: str | None, dry_run: bool) -> None:
    files = select_files(list_files(repo, token))
    total = sum(size_of(item) for item in files)
    print(f"{repo}: {len(files)} file, {total / 1e6:.0f} MB")
    for item in files:
        print(f"  {item['path']:44s} {size_of(item) / 1e6:8.1f} MB")
    if dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = out / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = size_of(item)
        if destination.exists() and expected and destination.stat().st_size == expected:
            print(f"  bỏ qua (đã đủ): {item['path']}")
            continue
        url = f"{HF_RESOLVE}/{repo}/resolve/main/{item['path']}"
        print(f"  tải {item['path']} ...")
        # `-C -` cho phép chạy lại là tiếp tục chỗ dở, không tải lại từ đầu.
        result = _curl(["-C", "-", "--progress-bar", url, "-o", str(destination)], token=token)
        if result.returncode != 0:
            raise SystemExit(f"tải hỏng: {item['path']} (curl exit {result.returncode})")

    print(f"\nxong -> {out}")
    print("Dùng offline bằng cách trỏ thẳng đường dẫn này và đặt HF_HUB_OFFLINE=1.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải model HF bằng curl (né lỗi SSL của Python)")
    parser.add_argument("repo", help="vd intfloat/multilingual-e5-large")
    parser.add_argument("--out", type=Path, default=None, help="mặc định storage/models/<tên repo>")
    parser.add_argument("--token", default=None, help="chỉ cần cho repo gated")
    parser.add_argument("--dry-run", action="store_true", help="chỉ liệt kê, không tải")
    args = parser.parse_args()

    out = args.out or Path("storage/models") / args.repo.split("/")[-1]
    try:
        download(args.repo, out, args.token, args.dry_run)
    except KeyboardInterrupt:
        print("\nđã dừng — chạy lại lệnh cũ để tiếp tục chỗ dở", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
