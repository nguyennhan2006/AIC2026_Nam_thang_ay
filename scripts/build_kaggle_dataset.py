"""Đóng dữ liệu thi đấu thành .zip theo TỪNG LOẠI để tải lên Kaggle.

Một archive cho mỗi loại dữ liệu, không cắt nhỏ theo dung lượng: tải lên bằng
`kaggle datasets` CLI hoặc trình duyệt đều nuốt được file lớn, và cắt nhỏ ra
chỉ làm phần ghép lại trên Kaggle rắc rối thêm.

Đường dẫn trong zip tính từ GỐC REPO (`storage/...`), nên giải nén đè lên một
bản clone là xong — không phải đoán file nào đi đâu.

Nén: ảnh JPEG, `.npy`, `.safetensors` đã là dữ liệu nén hoặc gần nhiễu; deflate
cho chúng tốn hàng chục phút CPU để lấy về 1-2%. Những đuôi đó lưu STORE, chỉ
JSONL và text mới nén thật.

Sáu archive, tổng ~35,4 GB::

    01_export.zip        1,10 GB  5 JSONL + manifest        BẮT BUỘC
    02_vectors.zip       0,34 GB  873 .npy vector jina      BẮT BUỘC
    03_models.zip        5,39 GB  jina-clip-v2 + e-v3       BẮT BUỘC
    04_hf_modules.zip    1,2 MB   cache trust_remote_code   BẮT BUỘC
    05_config.zip        ~60 KB   env ĐÃ BỎ KHOÁ + docs     BẮT BUỘC
    06_keyframes.zip    28,62 GB  176.722 ảnh               chỉ cần cho OCR/UI

Bốn archive đầu + config là đủ chạy backend và eval. Ảnh chỉ cần khi dựng lại
OCR, chạy VLM rerank, hoặc muốn UI hiện thumbnail.

    python -m scripts.build_kaggle_dataset --dry-run
    python -m scripts.build_kaggle_dataset --out-dir D:/aic_kaggle
    python -m scripts.build_kaggle_dataset --out-dir D:/aic_kaggle --skip-keyframes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

# Đuôi đã nén sẵn / không nén được — deflate chỉ tốn CPU mà không giảm được gì.
STORE_SUFFIXES = {".jpg", ".jpeg", ".png", ".npy", ".safetensors", ".bin", ".zip", ".mp4"}


def _mode_for(path: Path) -> int:
    return zipfile.ZIP_STORED if path.suffix.lower() in STORE_SUFFIXES else zipfile.ZIP_DEFLATED


def _files_under(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())


def _total(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths)


def _write_zip(target: Path, files: list[Path], root: Path) -> dict:
    """Ghi ra `.part` rồi mới đổi tên.

    Đứt giữa chừng mà để lại một zip nửa vời thì lần chạy sau sẽ thấy file tồn
    tại và BỎ QUA nó — hỏng âm thầm, chỉ lộ ra khi giải nén trên Kaggle.
    """

    temporary = target.with_name(target.name + ".part")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        for item in files:
            archive.write(item, arcname=item.relative_to(root).as_posix(),
                          compress_type=_mode_for(item))
    digest = hashlib.sha256()
    with temporary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    temporary.replace(target)
    return {
        "bytes": target.stat().st_size,
        "files": len(files),
        "sha256": digest.hexdigest(),
    }


def _redacted_env() -> tuple[str, int, list[str]]:
    """`.env.fpt.local` đã bỏ khoá, dùng lại đúng hai lớp lưới của
    `build_share_bundle`.

    Kaggle là dịch vụ của bên thứ ba; kể cả dataset để riêng tư cũng không phải
    chỗ cho khoá API nằm. Lớp thứ nhất soi TÊN biến, lớp thứ hai soi chính GIÁ
    TRỊ nên bắt được cả biến đặt tên không theo quy ước nào.
    """

    from scripts.build_share_bundle import audit_redacted, redact_env

    content, removed = redact_env(ROOT / ".env.fpt.local")
    return content, removed, audit_redacted(content)


def build_jobs(skip_keyframes: bool) -> list[dict]:
    hf_modules = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    jobs: list[dict] = [
        {
            "name": "01_export.zip",
            "files": _files_under(ROOT / "storage/exports_competition"),
            "root": ROOT,
            "note": "5 JSONL + manifest — BAT BUOC",
        },
        {
            "name": "02_vectors.zip",
            "files": _files_under(ROOT / "storage/processed/embeddings_pack"),
            "root": ROOT,
            "note": "873 .npy vector jina 1024 chieu — BAT BUOC",
        },
        {
            "name": "03_models.zip",
            "files": (_files_under(ROOT / "storage/models/jina-clip-v2")
                      + _files_under(ROOT / "storage/models/jina-embeddings-v3")),
            "root": ROOT,
            "note": "jina-clip-v2 + jina-embeddings-v3 — BAT BUOC",
        },
        {
            # Đường dẫn nằm NGOÀI repo nên root khác; người nhận chép vào
            # ~/.cache/huggingface/modules/ chứ không phải vào repo.
            "name": "04_hf_modules.zip",
            "files": _files_under(hf_modules),
            "root": hf_modules.parent,
            # CAN nhung CHUA DU: `cached_file()` doc `$HF_HOME/hub`, khong doc
            # thu muc nay. May nhan phai chay them
            # `python -m scripts.prepare_jina_offline` (~350 KB, mot lan).
            "note": "cache trust_remote_code — can, nhung con phai prepare_jina_offline",
        },
    ]
    if not skip_keyframes:
        jobs.append({
            "name": "06_keyframes.zip",
            "files": _files_under(ROOT / "storage/processed/keyframes"),
            "root": ROOT,
            "note": "176.722 anh — chi can cho OCR / VLM rerank / UI",
        })
    return jobs


def _write_config_zip(target: Path, content: str) -> dict:
    docs = [
        "docs/36_CHAY_HE_THONG.md",
        "docs/34_COMPETITION_PACK_IMPORT.md",
        "docs/35_KAGGLE_MEDIA.md",
    ]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(".env.fpt.local", content)
        for doc in docs:
            if (ROOT / doc).exists():
                archive.write(ROOT / doc, arcname=doc)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"bytes": target.stat().st_size, "files": 1 + len(docs), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dong goi du lieu thi dau cho Kaggle")
    parser.add_argument("--out-dir", type=Path, default=Path("D:/aic_kaggle"))
    parser.add_argument("--skip-keyframes", action="store_true",
                        help="bo 28,6 GB anh; van du de chay backend + eval")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    jobs = build_jobs(arguments.skip_keyframes)
    empty = [job["name"] for job in jobs if not job["files"]]
    if empty:
        sys.exit(f"THIEU du lieu, khong tim thay file nao cho: {empty}")

    print(f"{'archive':24s} {'GB':>8s} {'file':>9s}  ghi chu")
    grand = 0
    for job in jobs:
        size = _total(job["files"])
        grand += size
        print(f"{job['name']:24s} {size / (1 << 30):8.2f} {len(job['files']):9d}  {job['note']}")
    print(f"{'TONG':24s} {grand / (1 << 30):8.2f} {sum(len(j['files']) for j in jobs):9d}")

    content, removed, flagged = _redacted_env()
    print(f"\n05_config.zip: env da bo {removed} khoa — lưới thứ hai: "
          f"{'SACH' if not flagged else 'CON SOT'}")
    if flagged:
        sys.exit(f"DUNG LAI: con gia tri trong nhu khoa: {flagged}")

    if arguments.dry_run:
        print("\n(dry-run — chua ghi gi)")
        return

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.out_dir / "MANIFEST.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})

    def record(name: str, entry: dict, note: str) -> None:
        entry["note"] = note
        manifest[name] = entry
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
        print(f"{entry['bytes'] / (1 << 30):8.2f} GB  sha256={entry['sha256'][:16]}")

    for job in jobs:
        target = arguments.out_dir / job["name"]
        if target.exists():
            print(f"[bo qua] {job['name']} — da co")
            continue
        print(f"[dong  ] {job['name']:24s} ", end="", flush=True)
        record(job["name"], _write_zip(target, job["files"], job["root"]), job["note"])

    config_zip = arguments.out_dir / "05_config.zip"
    if config_zip.exists():
        print("[bo qua] 05_config.zip — da co")
    else:
        print(f"[dong  ] {'05_config.zip':24s} ", end="", flush=True)
        record("05_config.zip", _write_config_zip(config_zip, content),
               "env DA BO KHOA + docs — BAT BUOC")

    print(f"\n-> {arguments.out_dir}")
    print("   MANIFEST.json giu sha256 tung file de doi chieu sau khi tai len Kaggle.")
    print("   KHOA API khong nam trong goi — gui rieng cho dong doi.")


if __name__ == "__main__":
    main()
