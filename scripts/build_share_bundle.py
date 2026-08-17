"""Đóng một file .zip để chia cho đồng đội qua Google Drive.

Nguyên tắc chọn nội dung: **chỉ gói thứ người khác KHÔNG tự lấy được.** Ảnh
keyframe (28,7 GB) và objects (610 MB) tải thẳng từ mirror ban tổ chức nên không
gói — gói vào là bắt mỗi người tải hai lần cùng một thứ.

Còn lại đúng bốn thứ, và ba trong số đó nhỏ tới mức dễ bị quên:

1.  **Pack thi đấu** (~561 MB) — bản do chính nhóm dựng trên Kaggle. Không có ở
    đâu khác. Thiếu nó thì không có caption, ASR, vector, và không có bảng đổi
    tên `source_keyframe_index -> frame_idx` nên ảnh tải về cũng không đặt đúng
    chỗ được.

2.  **Cache `transformers_modules`** (~1,2 MB) — thứ chặn cứng dễ mất nhất.
    `storage/models/jina-clip-v2/` KHÔNG tự chứa đủ code: `config.json` của nó
    trỏ `auto_map` sang repo `jinaai/jina-clip-implementation`, và
    `transformers` tải phần đó về `~/.cache/huggingface/modules/` qua đúng cái
    stack HTTP mà `scripts/download_hf_model.py` đã ghi rõ là chết trên máy
    Windows của dự án. Máy nào đã chạy được một lần thì cache ấm và không ai để
    ý; máy mới thì kẹt ở 1,2 MB.

3.  **File env đã BỎ KHOÁ** — nơi chứa toàn bộ tham số đã đo (fusion `norm_max`,
    trọng số nhánh, `AVS cap=40`, `QA_TOP_N=10`) kèm lý do. Mất file này là mất
    phần lớn công đo đạc, mà `.gitignore` chặn nó nên nó không đi theo git.
    Mọi giá trị của biến có tên chứa KEY/TOKEN/SECRET/PASSWORD bị thay bằng
    rỗng; phần chú thích giữ nguyên.

4.  **CSV link của ban tổ chức** (~2 KB) — để chạy `fetch_aic_data.py`.

`--include-export` gói thêm export đã dựng sẵn + vector (1,1 GB). Cân nhắc:
importer dựng lại được trong ~6 phút, NHƯNG kết quả không trùng byte (provenance
đóng dấu thời gian lúc chạy), nên không có cách nào đối chiếu hai bên đã dựng
giống nhau chưa. Gói theo thì cả nhóm chắc chắn chạy trên cùng một corpus.

    python -m scripts.build_share_bundle --pack "D:/.../pack.zip" --out D:/aic_share.zip
    python -m scripts.build_share_bundle --pack ... --out ... --include-export
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sys
import zipfile

# Tên biến kết thúc bằng một trong các TỪ này (theo đoạn cuối, không phải chuỗi
# con) thì bỏ giá trị.
#
# Không dùng khớp chuỗi con: bản đầu viết vậy và nó xoá luôn
# `AIC_FPT_QA_MAX_TOKENS`, `AIC_CAPTION_MAX_TOKENS`, `AIC_KEYFRAMES_PER_SCENE`,
# `AIC_REDACT_SECRETS` — vì "KEY" nằm trong "KEYFRAMES" và "TOKEN" trong
# "MAX_TOKENS". Đó đúng là những giá trị ĐÃ ĐO mà gói này sinh ra để giữ, và
# `AIC_FPT_QA_MAX_TOKENS` rỗng thì QA rơi về mặc định 200 -> mọi lời gọi trả
# None -> im lặng dùng rule-based. "Cẩn thận quá" ở đây không phải an toàn hơn,
# chỉ là hỏng theo kiểu khác.
SECRET_SUFFIXES = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
# Tên không kết thúc bằng các từ trên nhưng vẫn là danh tính cá nhân.
SECRET_EXTRA = {"KAGGLE_USERNAME"}
ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# Lưới an toàn thứ hai, ĐỘC LẬP với danh sách tên: giá trị nào còn sót mà trông
# như khoá thì dừng hẳn để người dùng tự nhìn. Mẫu lấy từ
# `scripts/check_secret_leak.py` để hai chỗ không lệch nhau.
SUSPICIOUS_VALUE = (
    re.compile(r"^sk-[A-Za-z0-9]{16,}$"),
    re.compile(r"^AKIA[0-9A-Z]{16}$"),
    re.compile(r"^hf_[A-Za-z0-9]{20,}$"),
    re.compile(r"^[A-Za-z0-9_\-]{40,}$"),
)


def _is_secret_name(name: str) -> bool:
    return name.upper() in SECRET_EXTRA or name.upper().split("_")[-1] in SECRET_SUFFIXES

HF_MODULES = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"


def redact_env(path: Path) -> tuple[str, int]:
    """Nội dung file env với mọi giá trị khoá bị xoá. Trả `(noi_dung, so_dong)`."""

    out: list[str] = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ASSIGNMENT.match(line.strip())
        if match and _is_secret_name(match.group(1)) and match.group(2).strip():
            out.append(f"{match.group(1)}=")
            removed += 1
        else:
            out.append(line)
    return "\n".join(out) + "\n", removed


def audit_redacted(content: str) -> list[str]:
    """Giá trị còn sót mà trông như khoá. Rỗng là qua.

    Lưới thứ hai này KHÔNG nhìn tên biến — nó soi chính giá trị, nên bắt được cả
    trường hợp đặt tên không theo quy ước nào (`AIC_X=sk-...`).
    """

    flagged: list[str] = []
    for line in content.splitlines():
        match = ASSIGNMENT.match(line.strip())
        if not match:
            continue
        value = match.group(2).strip().strip("'\"")
        if any(pattern.match(value) for pattern in SUSPICIOUS_VALUE):
            flagged.append(match.group(1))
    return flagged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readme(entries: list[tuple[str, int]], pack_name: str, has_export: bool) -> str:
    listing = "\n".join(f"| `{name}` | {size/2**20:,.1f} MB |" for name, size in entries)
    export_step = (
        """
### 3b. Đã có sẵn export — bỏ qua bước 3

Gói này kèm `export/` và `embeddings_pack/`. Chép thẳng:

```powershell
Copy-Item -Recurse bundle\\export        storage\\exports_competition
Copy-Item -Recurse bundle\\embeddings_pack storage\\processed\\embeddings_pack
```
"""
        if has_export
        else ""
    )
    return f"""# Gói dữ liệu AIC2026 — dùng cùng branch `full-runnable`

Đóng ngày {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.

| Nội dung | Dung lượng |
|---|---:|
{listing}

Gói này **cố tình không kèm** ảnh keyframe (28,7 GB) và objects (610 MB): hai
thứ đó tải thẳng từ mirror ban tổ chức nhanh hơn (~29 MB/s), gói vào chỉ làm bạn
tải hai lần.

---

## 1. Lấy code

```bash
git clone -b full-runnable https://github.com/nguyennhan2006/AIC2026_Nam_thang_ay.git
cd AIC2026_Nam_thang_ay
python -m venv .venv
.venv\\Scripts\\activate
pip install -e ".[api,test]"
python -m pytest tests/ -q --ignore=tests/test_caption_qwen3vl_config.py
```

623 test phải PASS. (`test_caption_qwen3vl_config` cần `cv2`, không liên quan.)

## 2. Khoá và cấu hình

Chép `env/.env.team.template` thành `.env.fpt.local` ở gốc repo, rồi **điền các
dòng khoá đang để trống** (`AIC_FPT_API_KEY`, `AIC_ONLINE_API_KEY`, …).

⚠️ Mọi chú thích trong file đó là kết quả đo thật — đọc trước khi đổi số nào.

## 3. Dựng export từ pack

```powershell
python -m scripts.import_competition_pack `
    --pack <duong dan>\\{pack_name} `
    --objects-zip D:\\aic_archive\\objects-aic25-b1.zip `
    --out storage/exports_competition `
    --merge-embeddings-from storage/exports_multivideo
```

Ra 873 video / 87.742 scene / 176.707 keyframe, ~6 phút.
{export_step}
## 4. Model (9,2 GB)

```powershell
python -m scripts.download_hf_model jinaai/jina-clip-v2 --out storage/models/jina-clip-v2
python -m scripts.download_hf_model openai/clip-vit-large-patch14 --out storage/models/clip-vit-large-patch14
```

**Rồi chép thư mục `hf_modules/` trong gói này vào:**

```powershell
Copy-Item -Recurse bundle\\hf_modules\\* "$env:USERPROFILE\\.cache\\huggingface\\modules\\transformers_modules\\"
```

Bước chép này **không bỏ qua được**. `jina-clip-v2` dùng `trust_remote_code`:
code mô hình nằm ở repo khác và `transformers` tải nó qua `huggingface_hub` —
đúng cái stack chết với `SSLCertVerificationError` trên máy Windows của dự án
(xem docstring `scripts/download_hf_model.py`). Không có thư mục này thì container
chết ngay lúc dựng, ở một file 1,2 MB.

## 5. Ảnh keyframe (28,7 GB) và objects (610 MB)

```powershell
$csv = "<duong dan>\\links\\aic2026_batch1.csv"
$pack = "<duong dan>\\{pack_name}"

python -m scripts.fetch_aic_data --what objects   --pack $pack --csv $csv --keep-zip
python -m scripts.fetch_aic_data --what keyframes --pack $pack --csv $csv
```

Đứt giữa chừng thì chạy lại — nối tiếp được. Xác nhận:

```powershell
python -m scripts.fetch_kaggle_media --verify --export storage/exports_competition
```

## 6. Chạy

```powershell
$env:AIC_ENV_FILE                 = ".env.fpt.local"
$env:AIC_METADATA_JSONL           = "storage/exports_competition/scenes.jsonl"
$env:AIC_VISUAL_EMBEDDING_NAME    = "jina_clip_v2"
$env:AIC_VISUAL_EMBEDDING_MODEL   = "storage/models/jina-clip-v2"
$env:AIC_ENABLE_QUERY_TRANSLATION = "false"
$env:AIC_BRANCH_TIMEOUT_MS        = "30000"
$env:HF_HUB_OFFLINE               = "1"
$env:TRANSFORMERS_OFFLINE         = "1"

python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

Khởi động ~4 phút, cần ~5 GB RAM. Mọi endpoint nằm dưới `/v1`, và `/v1/*` trừ
`/v1/health` đòi header `Authorization: Bearer <AIC_ONLINE_API_KEY>`.

`AIC_BRANCH_TIMEOUT_MS=30000` **không phải tuỳ chọn**: ở 87.742 scene nhánh
`dense_visual` mất 5,2–11,8 s, để mặc định 8000 là nó bị cắt **trong im lặng**.

UI: `cd online/ui-react && npm run dev`, đặt API base `http://localhost:8000`.

## 7. Đọc trước khi đổi bất cứ thứ gì

| Tài liệu | Nội dung |
|---|---|
| `README_BRANCH.md` | tổng quan, dựng lại dữ liệu |
| `docs/34_COMPETITION_PACK_IMPORT.md` | pack thiếu gì, importer quyết định gì |
| `docs/35_KAGGLE_MEDIA.md` | tải ảnh, và vì sao không được chép thẳng |
| `docs/20_EXPERIMENT_LOG.md` | mọi thí nghiệm đã chạy, gồm cả cái **DROP** |
| `docs/27_SYSTEM_ISSUES.md` | cấu hình nào bật/tắt và vì sao |

## 8. Chưa làm, biết trước để khỏi mất công

- **OCR 0%** trên cả 873 video, mà 75/120 truy vấn gold cần OCR và `bm25_ocr`
  đang để trọng số 1.0.
- **Không có events** → `event_search` rỗng, và dedup theo event của AVS không
  chạy (mà `AIC_AVS_MAX_RESULTS_PER_VIDEO=40` được chọn dựa trên giả định dedup
  hoạt động).
- **Chưa đo chất lượng ở quy mô 873 video.** Mọi tham số trong file env đo trên
  765 scene / 3 video. Một cái đã hỏng vì quy mô rồi (`BRANCH_TIMEOUT_MS`).
- `color_search` rỗng cho tới khi chạy `scripts/backfill_color_quality.py` trên
  ảnh đã tải.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pack", required=True, help="AIC2026_competition_clean_v3*.zip")
    parser.add_argument("--out", required=True, help="file .zip se tao")
    parser.add_argument("--env-file", default=".env.fpt.local")
    parser.add_argument("--links-csv", help="CSV link ban to chuc")
    parser.add_argument("--export-dir", default="storage/exports_competition")
    parser.add_argument("--vectors-dir", default="storage/processed/embeddings_pack")
    parser.add_argument(
        "--include-export",
        action="store_true",
        help="gói thêm export + vector da dung san (1,1 GB) de ca nhom chac chan cung corpus",
    )
    arguments = parser.parse_args()

    pack = Path(arguments.pack)
    if not pack.exists():
        raise SystemExit(f"khong thay pack: {pack}")
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, int]] = []
    # ZIP_STORED: pack/vector/npy đều đã nén sẵn, ép deflate lần nữa chỉ tốn
    # hàng chục phút CPU để giảm dưới 1%.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
        print(f"  pack        {pack.name} ({pack.stat().st_size/2**20:,.0f} MB) …", flush=True)
        bundle.write(pack, f"pack/{pack.name}")
        entries.append((f"pack/{pack.name}", pack.stat().st_size))

        if HF_MODULES.is_dir():
            total = 0
            for item in HF_MODULES.rglob("*"):
                if item.is_file() and "__pycache__" not in item.parts:
                    bundle.write(item, f"hf_modules/{item.relative_to(HF_MODULES).as_posix()}")
                    total += item.stat().st_size
            entries.append(("hf_modules/", total))
            print(f"  hf_modules  {total/2**20:.1f} MB")
        else:
            print(f"  CANH BAO: khong thay {HF_MODULES} — nguoi nhan se ket o buoc 4")

        env_path = Path(arguments.env_file)
        if env_path.exists():
            content, removed = redact_env(env_path)
            leftover = [
                match.group(1)
                for line in content.splitlines()
                if (match := ASSIGNMENT.match(line.strip()))
                and _is_secret_name(match.group(1))
                and match.group(2).strip()
            ]
            suspicious = audit_redacted(content)
            if leftover or suspicious:
                raise SystemExit(
                    "BO KHOA THAT BAI — dung lai truoc khi ghi file chia se.\n"
                    f"  con gia tri theo TEN : {leftover}\n"
                    f"  gia tri trong nhu khoa: {suspicious}"
                )
            bundle.writestr("env/.env.team.template", content)
            entries.append(("env/.env.team.template", len(content.encode())))
            print(f"  env         da bo gia tri cua {removed} bien khoa")
        else:
            print(f"  CANH BAO: khong thay {env_path}")

        if arguments.links_csv and Path(arguments.links_csv).exists():
            bundle.write(Path(arguments.links_csv), "links/aic2026_batch1.csv")
            entries.append(("links/aic2026_batch1.csv", Path(arguments.links_csv).stat().st_size))
            print("  links       CSV ban to chuc")

        if arguments.include_export:
            for source, prefix in (
                (Path(arguments.export_dir), "export"),
                (Path(arguments.vectors_dir), "embeddings_pack"),
            ):
                if not source.is_dir():
                    print(f"  CANH BAO: khong thay {source}")
                    continue
                total = 0
                for item in sorted(source.rglob("*")):
                    if item.is_file():
                        bundle.write(item, f"{prefix}/{item.relative_to(source).as_posix()}")
                        total += item.stat().st_size
                entries.append((f"{prefix}/", total))
                print(f"  {prefix:11s} {total/2**20:,.0f} MB")

        bundle.writestr("README.md", _readme(entries, pack.name, arguments.include_export))

    size = out.stat().st_size
    print()
    print(f"  {out}  —  {size/2**30:.2f} GB")
    print(f"  sha256 {_sha256(out)}")
    print()
    print("  Tai len Google Drive roi chia link. Nguoi nhan doc README.md trong goi.")
    sys.exit(0)


if __name__ == "__main__":
    main()
