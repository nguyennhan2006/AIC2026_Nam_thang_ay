# Khoi dong server tren corpus thi dau (873 video / 87.742 scene / 176.707 keyframe).
#
# Gom lai mot cho toan bo cau hinh da kiem chung trong docs/34 muc 7, vi ngay thi
# khong phai luc go lai 7 bien moi truong tu tai lieu. Chay:
#
#     .\scripts\run_competition.ps1
#
# Cong mo NGAY (vai giay). Viec nap ~4 phut chay o luong nen, nen
# "Application startup complete." KHONG con nghia la da san sang.
# Hoi tien do bang:  curl -s http://127.0.0.1:8080/v1/startup
# UI mo duoc luon va tu hien thanh cho; truy van bam ngay cung duoc,
# no se tu chay khi nap xong.

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

# --- Kiem tra truoc khi ton 4 phut khoi dong -------------------------------
$must = @(
    "storage/exports_competition/scenes.jsonl",
    "storage/exports_competition/keyframes.jsonl",
    "storage/exports_competition/videos.jsonl",
    "storage/exports_competition/events.jsonl",
    "storage/models/jina-clip-v2/config.json",
    "storage/models/jina-embeddings-v3/config.json",
    ".env.fpt.local"
)
foreach ($f in $must) {
    if (-not (Test-Path -LiteralPath $f)) { throw "THIEU: $f" }
}

# jina-clip-v2 mac dinh tro text tower toi repo HuggingFace tu xa. Voi
# HF_HUB_OFFLINE=1 thi `model_info` NEM LOI thay vi roi ve cache, nen container
# chet ngay luc dung. Da va config.json tro sang ban local; kiem lai o day vi
# tai ve model moi se ghi de va loi quay lai ma khong ai nho.
$cfgPath = "storage/models/jina-clip-v2/config.json"
$cfg = Get-Content -Raw -LiteralPath $cfgPath | ConvertFrom-Json
if ($cfg.text_config.hf_model_name_or_path -like "jinaai/*") {
    # Tai model moi tu HuggingFace/Drive se ghi de lai gia tri goc, nen VA O DAY
    # thay vi bat nguoi dung tu sua: tren may dong doi day la loi dau tien gap
    # phai, va thong bao cua transformers khong he goi y nguyen nhan.
    if (-not (Test-Path "$cfgPath.orig")) { Copy-Item $cfgPath "$cfgPath.orig" }
    $cfg.text_config.hf_model_name_or_path = "storage/models/jina-embeddings-v3"
    $cfg | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $cfgPath -Encoding utf8
    Write-Host "[va] jina-clip-v2/config.json: text tower -> storage/models/jina-embeddings-v3 (ban goc luu o config.json.orig)"
}

$free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($free -lt 6) {
    Write-Warning ("Chi con {0:N1} GB RAM trong. Container can ~5 GB (dinh 5,1). Dong bot ung dung truoc." -f $free)
}

if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }

# --- Cau hinh ---------------------------------------------------------------
$env:AIC_ENV_FILE                 = ".env.fpt.local"
$env:AIC_METADATA_JSONL           = "storage/exports_competition/scenes.jsonl"
# Pack chi co vector jina 1024 chieu. Dung VISUAL_EMBEDDING_NAME chu KHONG dung
# AIC_DENSE_INDEXES: bien kia doi ten nhanh thanh `dense_jina_clip_v2`, lam moi
# trong so nhanh va moi --disable-branch da luu tro sai cho.
$env:AIC_VISUAL_EMBEDDING_NAME    = "jina_clip_v2"
$env:AIC_VISUAL_EMBEDDING_MODEL   = "storage/models/jina-clip-v2"
# jina co text tower da ngu - dich VI->EN cho no la mat thong tin, va bo dich
# thi duong truy van khong can mang, khong ton loi goi FPT nao.
$env:AIC_ENABLE_QUERY_TRANSLATION = "false"
# 8000 ms la con so chon cho 765 scene. O 87.742 scene, dense_visual mat
# 5,2-11,8 s; vuot han thi nhanh bien mat KHONG BAO LOI (branch_status=timeout,
# API van 200, UI van co ket qua - chi la tang ngu nghia da tat).
$env:AIC_BRANCH_TIMEOUT_MS        = "300000"
# Dat TUONG MINH "true", khong phai bo dong nay di. Pack co OCR 0% nen tat hai
# nhanh nay nghe hop ly, nhung tat chung DOI TOPOLOGY nhanh:
# `/v1/search/capabilities` tu choi moi `search_options` tro toi nhanh khong
# ton tai bang 422, ma UI gui kem trong so da luu (`AIC_BRANCH_WEIGHTS` co
# `bm25_ocr:1.0`) - nen MOI truy van tu UI deu 422. Da xay ra that.
#
# Phai gan "true" chu khong phai xoa dong: `$env:` song suot phien PowerShell,
# nen chay lai script trong CUNG terminal se giu nguyen gia tri "false" cu.
#
# Tat cung chang duoc gi: do tren 120 truy van, `bm25_ocr` p50 0 ms,
# `ocr_fuzzy` p50 0 ms / max 10 ms. Nhanh rong gan nhu mien phi.
$env:AIC_ENABLE_OCR_BRANCH        = "true"
$env:AIC_ENABLE_OCR_FUZZY         = "true"
# KHONG dat AIC_OCR_OVERLAY_DF o day: .env.fpt.local da giu 0.10, va do lai
# tren 1236 scene sau khi nap OCR Qwen thi 0.10 van la lua chon dung (xem ghi
# chu tai cho khai bao). Dat lai o day chi tao hai nguon su that.
$env:HF_HUB_OFFLINE               = "1"
$env:TRANSFORMERS_OFFLINE         = "1"
$env:PYTHONIOENCODING             = "utf-8"
# Retrieval lay nhieu hon de cai thien recall, fusion se chon top-k
$env:AIC_RETRIEVAL_MULTIPLIER     = "10"

Write-Host "metadata : $env:AIC_METADATA_JSONL"
Write-Host "embedding: $env:AIC_VISUAL_EMBEDDING_NAME ($env:AIC_VISUAL_EMBEDDING_MODEL)"
Write-Host "Cong mo ngay; nap ~4 phut o luong nen."
Write-Host "Tien do: curl -s http://127.0.0.1:8080/v1/startup`n"

python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8080
