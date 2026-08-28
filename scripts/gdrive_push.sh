#!/usr/bin/env bash
# =============================================================================
# gdrive_push.sh - Đóng gói dữ liệu đang chạy trên máy Vast rồi đẩy lên Drive
#
# Chạy TRÊN MÁY VAST, không phải máy Windows: dữ liệu đã nằm sẵn ở đó, và
# đường lên của máy thuê nhanh hơn đường lên nhà vài chục lần. Đây là chiều
# ngược của `bootstrap_vast_from_gdrive.sh`.
#
#   ./scripts/gdrive_push.sh --setup      # một lần: nối rclone vào Drive
#   ./scripts/gdrive_push.sh --plan       # xem sẽ đóng gì, bao nhiêu, chưa làm
#   ./scripts/gdrive_push.sh              # đóng + đẩy
#   ./scripts/gdrive_push.sh --skip-keyframes
#   ./scripts/gdrive_push.sh --verify     # đối chiếu sha256 remote vs MANIFEST
#
# ĐĨA: không bao giờ cần chỗ cho cả 34 GB. Ảnh keyframe được cắt thành nhiều
# phần ~6 GB, mỗi lần chỉ một archive nằm trên đĩa rồi xoá ngay sau khi đẩy —
# nên đỉnh chiếm thêm bằng đúng archive lớn nhất, không phải tổng.
# =============================================================================
set -euo pipefail

ROOT="${AIC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE="${AIC_GDRIVE_REMOTE:-gdrive}"
REMOTE_DIR="${AIC_GDRIVE_DIR:-AIC2026_pack}"
SPOOL="${AIC_SPOOL:-}"          # mac dinh tinh SAU khi doc --root, xem duoi
PART_BYTES=$(( 6 * 1024 * 1024 * 1024 ))

SKIP_KEYFRAMES=0
DO_SETUP=0
PLAN_ONLY=0
VERIFY_ONLY=0
FORCE=0
WITH_VIDEOS=0
VIDEOS_ONLY=0
TRANSFERS=8

while [ $# -gt 0 ]; do
    case "$1" in
        --setup)          DO_SETUP=1 ;;
        --plan)           PLAN_ONLY=1 ;;
        --verify)         VERIFY_ONLY=1 ;;
        --skip-keyframes) SKIP_KEYFRAMES=1 ;;
        --videos)         WITH_VIDEOS=1 ;;
        --videos-only)    WITH_VIDEOS=1; VIDEOS_ONLY=1 ;;
        --transfers)      TRANSFERS="$2"; shift ;;
        --force)          FORCE=1 ;;
        --root)           ROOT="$2"; shift ;;
        --remote)         REMOTE="$2"; shift ;;
        --remote-dir)     REMOTE_DIR="$2"; shift ;;
        --spool)          SPOOL="$2"; shift ;;
        --part-size)      PART_BYTES=$(( $2 * 1024 * 1024 * 1024 )); shift ;;
        -h|--help)        sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Tham so la: $1" >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '[push] %s\n' "$*"; }
die()  { printf '[push] DUNG: %s\n' "$*" >&2; exit 1; }
human(){ numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 B"; }

# Spool nam CANH repo chu khong trong repo: tren Vast la /workspace/_gdrive_spool,
# tuc van tren volume ben vung, nhung khong lam ban cay git. Tinh o day - sau
# vong doc tham so - de `--root` keo theo ca spool.
[ -n "$SPOOL" ] || SPOOL="$(dirname "$ROOT")/_gdrive_spool"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_MODULES="$HF_HOME/modules"

MANIFEST="$SPOOL/MANIFEST.json"

# --- Công cụ -----------------------------------------------------------------

ensure_tools() {
    command -v bsdtar >/dev/null 2>&1 || {
        log "Cai bsdtar (libarchive-tools)..."
        apt-get update -qq && apt-get install -y -qq libarchive-tools
    }
    command -v rclone >/dev/null 2>&1 || {
        log "Cai rclone..."
        curl -fsSL https://rclone.org/install.sh | bash
    }
}

# --- Setup: OAuth trên máy không có trình duyệt ------------------------------

if [ "$DO_SETUP" -eq 1 ]; then
    ensure_tools
    cat <<'EOF'

=== Noi rclone vao Google Drive (may nay khong co trinh duyet) ===

Buoc 1 - TREN MAY WINDOWS cua ban, mo PowerShell va chay:

    D:\aic_kaggle\_tools\rclone.exe authorize "drive"

  Trinh duyet mo ra -> dang nhap DUNG nick Drive muon chua data -> Allow.
  rclone in ra mot khoi JSON mot dong, giua hai dong
  "Paste the following into your remote machine --->" va "<---End paste".
  Copy nguyen khoi do, tu dau { den cuoi }.

DUNG LUONG: pack day ~34,4 GiB. Drive mien phi chi 15 GB va dung chung voi
Gmail + Photos. Khong du thi dung --skip-keyframes (~5,8 GiB) hoac nick truong.

EOF
    # PHAI hoi token o DAY roi truyen thang vao lam tham so config.
    #
    # Duong `config create ... config_is_local=false` roi cho no hoi
    # "config_token>" KHONG chay: rclone ghi ro "if the config process would
    # normally ask a question the default is taken (unless --non-interactive)".
    # Tuc la no lay mac dinh RONG va tao ra remote khong co token, im lang,
    # roi moi lenh sau do chet voi "empty token found". Da dinh that.
    printf 'Dan khoi JSON vao day roi Enter (de trong = huy):\n> '
    read -r TOKEN
    case "$TOKEN" in
        "") die "chua co token - chay lai sau khi lam xong Buoc 1." ;;
        *access_token*) : ;;
        *) die "chuoi vua dan khong chua \"access_token\" - copy thieu? Lay ca dong tu { den }." ;;
    esac

    # config_refresh_token=false: chan rclone tu dong di lam lai vong OAuth
    # (may nay khong co trinh duyet nen no se treo hoac hong).
    if rclone listremotes | grep -qx "${REMOTE}:"; then
        log "Remote '${REMOTE}:' da co - cap nhat token."
        rclone config update "$REMOTE" token "$TOKEN" config_refresh_token=false >/dev/null
    else
        rclone config create "$REMOTE" drive scope=drive \
            token="$TOKEN" config_refresh_token=false >/dev/null
    fi

    log "Kiem lai bang cach hoi dung luong Drive:"
    if rclone about "${REMOTE}:"; then
        log "Noi thanh cong. Tiep theo: ./scripts/gdrive_push.sh --plan"
    else
        die "noi that bai. Token co the da het han (lay lai o Buoc 1) hoac dan thieu."
    fi
    exit 0
fi

# --- Bảng nội dung -----------------------------------------------------------
#
# "store" cho ảnh JPEG / .npy / .safetensors: chúng đã nén hoặc gần nhiễu, ép
# deflate lên 176k ảnh tốn hàng chục phút CPU để lấy về 1-2%. Chỉ JSONL và text
# mới nén thật.
dir_bytes() { [ -d "$1" ] && du -sb "$1" 2>/dev/null | cut -f1 || echo 0; }

EXPORT_DIR="$ROOT/storage/exports_competition"
VECTOR_DIR="$ROOT/storage/processed/embeddings_pack"
MODEL_DIR="$ROOT/storage/models"
KEYFRAME_DIR="$ROOT/storage/processed/keyframes"

check_sources() {
    [ -d "$EXPORT_DIR" ]  || die "khong thay $EXPORT_DIR - may nay chua co du lieu?"
    [ -d "$VECTOR_DIR" ]  || die "khong thay $VECTOR_DIR"
    [ -d "$MODEL_DIR/jina-clip-v2" ] || die "khong thay $MODEL_DIR/jina-clip-v2"
    [ -d "$HF_MODULES/transformers_modules" ] || die \
        "khong thay $HF_MODULES/transformers_modules (HF_HOME dang la $HF_HOME?)"
    local n
    n=$(find "$VECTOR_DIR" -maxdepth 1 -name '*.npy' | wc -l)
    [ "$n" -eq 873 ] || log "CANH BAO: co $n/873 vector .npy - pack se thieu."
}

# --- Cắt ảnh keyframe thành phần ---------------------------------------------
#
# Cắt theo THƯ MỤC VIDEO chứ không theo file: mỗi phần vẫn là một cây
# `storage/processed/keyframes/<video>/...` giải nén đè lên nhau được, và không
# video nào bị xẻ đôi giữa hai phần.
plan_keyframe_parts() {
    local list_dir="$1"
    rm -rf "$list_dir"; mkdir -p "$list_dir"
    # `du` mot lan duy nhat: quet 176k anh mat gan mot phut, va bang tom tat o
    # duoi can dung con so nay chu khong phai quet lai lan hai.
    ( cd "$ROOT" && du -sb storage/processed/keyframes/*/ 2>/dev/null | sort -k2 ) \
    | awk -v limit="$PART_BYTES" -v out="$list_dir" '
        BEGIN { part = 1; acc = 0; total = 0 }
        {
            size = $1
            path = $2
            sub(/\/$/, "", path)
            if (acc > 0 && acc + size > limit) { part++; acc = 0 }
            printf "%s\n", path >> sprintf("%s/part%02d.txt", out, part)
            acc += size
            total += size
        }
        END {
            printf "%d\n", part  > out "/count"
            printf "%d\n", total > out "/total"
        }
    '
}

# --- Đóng một archive rồi đẩy -------------------------------------------------

remote_has() {
    rclone lsjson "${REMOTE}:${REMOTE_DIR}/$1" >/dev/null 2>&1
}

# Chan TRUOC khi dong, khong phai sau: dong mot phan 6 GB roi moi chet vi het
# dia la mat ca luot nen do CPU. Nhan 1.1 vi zip cua anh JPEG (store) gan bang
# dung tong nguon, chi hon phan muc luc.
need_space() {
    local want="$1" free
    mkdir -p "$SPOOL"
    free=$(df -B1 --output=avail "$SPOOL" | tail -1)
    if [ "$free" -lt $(( want * 11 / 10 )) ]; then
        die "spool $SPOOL chi con $(human "$free"), can ~$(human "$want"). Dung --spool <cho khac> hoac --part-size nho hon."
    fi
}

record_manifest() {
    local name="$1" file="$2" note="$3"
    local bytes sha
    bytes=$(stat -c %s "$file")
    sha=$(sha256sum "$file" | cut -d' ' -f1)
    python3 - "$MANIFEST" "$name" "$bytes" "$sha" "$note" <<'PY'
import json, sys, pathlib
path, name, size, sha, note = sys.argv[1:6]
p = pathlib.Path(path)
data = json.loads(p.read_text()) if p.exists() else {}
data[name] = {"bytes": int(size), "sha256": sha, "note": note}
p.write_text(json.dumps(data, ensure_ascii=False, indent=1))
PY
    log "  sha256=${sha:0:16}  $(human "$bytes")"
}

# Dat PACK_EXCLUDE=(--exclude PAT ...) NGAY TRUOC mot lan goi pack_and_push de
# loai file khoi archive do. Ham tu xoa lai sau khi dung, nen khong ro ri sang
# lan goi ke tiep.
PACK_EXCLUDE=()

pack_and_push() {
    local name="$1" base="$2" compression="$3" note="$4"
    shift 4
    local excludes=(${PACK_EXCLUDE[@]+"${PACK_EXCLUDE[@]}"})
    PACK_EXCLUDE=()

    if [ "$FORCE" -eq 0 ] && remote_has "$name"; then
        log "[bo qua] $name - da co tren Drive"
        return 0
    fi

    local want
    want=$( cd "$base" && du -scb "$@" 2>/dev/null | tail -1 | cut -f1 )
    need_space "${want:-0}"

    local out="$SPOOL/$name"
    log "[dong  ] $name ($(human "${want:-0}") nguon)"

    # Ghi ra .part roi moi doi ten: dut giua chung ma de lai zip nua voi thi luot
    # sau thay file ton tai va BO QUA no - hong am tham, chi lo ra luc giai nen.
    bsdtar -c -f "$out.part" --format zip \
           --options "zip:compression=$compression" \
           ${excludes[@]+"${excludes[@]}"} -C "$base" "$@"
    mv "$out.part" "$out"

    record_manifest "$name" "$out" "$note"

    log "[day   ] $name -> ${REMOTE}:${REMOTE_DIR}/"
    rclone copyto "$out" "${REMOTE}:${REMOTE_DIR}/$name" \
        --drive-chunk-size 128M --progress --stats 15s

    # Xoa ngay: dinh chiem dia bang archive LON NHAT, khong phai tong.
    rm -f "$out"
}

# --- 05_config: bỏ khoá trước khi rời máy ------------------------------------
#
# `.env.fpt.local` TRÊN MÁY VAST có khoá FPT thật. Drive là chỗ lưu của bên thứ
# ba; đẩy nguyên bản lên đó là rò khoá. Dùng lại đúng hai lớp lưới của
# `build_share_bundle`: lớp một soi TÊN biến, lớp hai soi chính GIÁ TRỊ nên bắt
# được cả biến đặt tên không theo quy ước.
build_config_zip() {
    local out="$1"
    ( cd "$ROOT" && python3 - "$out" <<'PY'
import sys, zipfile, pathlib
sys.path.insert(0, ".")
from scripts.build_share_bundle import redact_env, audit_redacted

out = pathlib.Path(sys.argv[1])
env = pathlib.Path(".env.fpt.local")
if not env.exists():
    raise SystemExit("DUNG: khong thay .env.fpt.local")

content, removed = redact_env(env)
flagged = audit_redacted(content)
if flagged:
    raise SystemExit(f"DUNG: con gia tri trong nhu khoa: {flagged}")

docs = ["docs/36_CHAY_HE_THONG.md", "docs/38_VASTAI_RUNBOOK.md",
        "docs/42_GDRIVE_PACK.md", "docs/34_COMPETITION_PACK_IMPORT.md"]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(".env.fpt.local", content)
    for doc in docs:
        if pathlib.Path(doc).exists():
            archive.write(doc)
print(f"[push]   da bo {removed} khoa khoi .env.fpt.local")
PY
    )
}

push_config() {
    if [ "$FORCE" -eq 0 ] && remote_has "05_config.zip"; then
        log "[bo qua] 05_config.zip - da co tren Drive"
        return 0
    fi
    mkdir -p "$SPOOL"
    log "[dong  ] 05_config.zip"
    build_config_zip "$SPOOL/05_config.zip"
    record_manifest "05_config.zip" "$SPOOL/05_config.zip" "env DA BO KHOA + docs"
    rclone copyto "$SPOOL/05_config.zip" "${REMOTE}:${REMOTE_DIR}/05_config.zip" --progress
    rm -f "$SPOOL/05_config.zip"
}

# --- Video thô: copy thẳng, KHÔNG nén ----------------------------------------
#
# Khác hẳn mọi archive ở trên, và cố ý:
#
#   * `.mp4` đã là dữ liệu nén. Bọc thêm một lớp zip tốn hàng chục phút CPU để
#     lấy về gần 0% — cùng lý do các archive kia để STORE cho ảnh với `.npy`.
#   * Zip 78 GB cần 78 GB spool trống bên cạnh; máy chỉ còn ~109 GB, mà phần
#     lớn còn phải dành cho chính dữ liệu đang chạy.
#   * Copy thẳng thì rclone tự bỏ qua file đã có (so kích thước + hash), nên
#     đứt mạng chạy lại chỉ đẩy tiếp phần thiếu — zip một khối thì hỏng là mất
#     cả 78 GB.
#   * 874 file là số Drive nuốt thoải mái. Ảnh keyframe phải nén vì có tới
#     176.722 file: ở mức đó, chi phí mỗi-file của Drive API mới thành nút thắt.
push_videos() {
    local src="$ROOT/storage/raw/videos"
    [ -d "$src" ] || die "khong thay $src"

    local count bytes
    count=$(find "$src" -maxdepth 1 -type f -name '*.mp4' | wc -l)
    bytes=$(du -sb "$src" | cut -f1)
    log "Video tho: $count file, $(human "$bytes") -> ${REMOTE}:${REMOTE_DIR}/videos/"
    log "Copy THANG, khong nen (mp4 nen san roi)."

    if [ "$PLAN_ONLY" -eq 1 ]; then return 0; fi

    # --transfers: so file day song song. Moi transfer giu --drive-chunk-size
    # trong RAM, nen 8 x 128M = 1 GB - khong dang ke voi 125 GB cua may nay.
    # Cao hon nua thi Drive bat dau tra 403 rateLimitExceeded; rclone co pacer
    # tu lui lai nen khong hong, chi la khong nhanh them.
    #
    # KHONG dung --drive-chunk-size lon hon: file trung binh ~90 MB, chunk 128M
    # nghia la phan lon file di trong mot luot, khong con gi de cat nho.
    rclone copy "$src" "${REMOTE}:${REMOTE_DIR}/videos" \
        --transfers "$TRANSFERS" \
        --checkers 16 \
        --drive-chunk-size 128M \
        --exclude '*.part' \
        --progress --stats 15s

    log "Xong video. Chay lai lenh nay bat cu luc nao - rclone bo qua file da co."
}

# --- Verify ------------------------------------------------------------------

if [ "$VERIFY_ONLY" -eq 1 ]; then
    ensure_tools
    [ -f "$MANIFEST" ] || die "khong thay $MANIFEST - chua day lan nao?"
    log "Doi chieu sha256 remote vs MANIFEST"
    # Drive tu tinh sha256 phia server, nen day la kiem tra THAT chu khong phai
    # so sanh kich thuoc suong.
    #
    # Ghi ra file roi moi doc, KHONG pipe vao python: heredoc '<<PY' da chiem
    # stdin de dua ma nguon, nen du lieu di qua pipe se bi nuot mat.
    rclone lsjson "${REMOTE}:${REMOTE_DIR}" --hash --hash-type sha256 \
        > "$SPOOL/_remote.json"
    python3 - "$MANIFEST" "$SPOOL/_remote.json" <<'PY'
import json, sys
remote = {item["Name"]: item for item in json.load(open(sys.argv[2]))}
want = json.load(open(sys.argv[1]))
bad = 0
for name, entry in sorted(want.items()):
    got = remote.get(name)
    if got is None:
        print(f"  [--] {name:26s} chua co tren Drive"); bad += 1; continue
    sha = (got.get("Hashes") or {}).get("sha256")
    if got["Size"] != entry["bytes"]:
        print(f"  [!!] {name:26s} lech size {got['Size']} vs {entry['bytes']}"); bad += 1
    elif sha and sha != entry["sha256"]:
        print(f"  [!!] {name:26s} LECH sha256 - day lai file nay"); bad += 1
    elif sha:
        print(f"  [OK] {name:26s} sha256 khop")
    else:
        print(f"  [~ ] {name:26s} size khop, Drive chua tra hash")
sys.exit(1 if bad else 0)
PY
    exit $?
fi

# --- Chạy --------------------------------------------------------------------

ensure_tools
# `--videos-only` khong dung toi export/vector/model, nen dung bat chung phai co.
[ "$VIDEOS_ONLY" -eq 1 ] || check_sources

# `--plan` khong cham toi Drive nen khong doi remote: dung no de kiem du lieu
# nguon TRUOC khi bo cong noi OAuth.
if [ "$PLAN_ONLY" -eq 0 ]; then
    rclone listremotes | grep -qx "${REMOTE}:" || die \
        "rclone chua co remote '${REMOTE}:'. Chay truoc: ./scripts/gdrive_push.sh --setup"
fi

# Video khong di qua spool, khong nen, khong dung MANIFEST - nen `--videos-only`
# ra thang o day, bo qua ca pha quet 176k anh (ton gan mot phut cho khong).
if [ "$VIDEOS_ONLY" -eq 1 ]; then
    push_videos
    exit 0
fi

mkdir -p "$SPOOL"
LISTS="$SPOOL/_parts"
parts=0
keyframe_bytes=0
if [ "$SKIP_KEYFRAMES" -eq 0 ] && [ -d "$KEYFRAME_DIR" ]; then
    log "Dang quet $KEYFRAME_DIR de chia phan (176k anh, hoi lau)..."
    plan_keyframe_parts "$LISTS"
    parts=$(cat "$LISTS/count" 2>/dev/null || echo 0)
    keyframe_bytes=$(cat "$LISTS/total" 2>/dev/null || echo 0)
fi

export_bytes=$(dir_bytes "$EXPORT_DIR")
vector_bytes=$(dir_bytes "$VECTOR_DIR")
model_bytes=$(( $(dir_bytes "$MODEL_DIR/jina-clip-v2") + $(dir_bytes "$MODEL_DIR/jina-embeddings-v3") ))
hf_bytes=$(dir_bytes "$HF_MODULES/transformers_modules")
video_bytes=0
[ "$WITH_VIDEOS" -eq 1 ] && video_bytes=$(dir_bytes "$ROOT/storage/raw/videos")
source_total=$(( export_bytes + vector_bytes + model_bytes + hf_bytes + keyframe_bytes + video_bytes ))

echo
printf '%-26s %12s  %s\n' "archive" "nguon" "noi dung"
printf '%-26s %12s  %s\n' "01_export.zip"     "$(human "$export_bytes")" "5 JSONL + manifest"
printf '%-26s %12s  %s\n' "02_vectors.zip"    "$(human "$vector_bytes")" "873 .npy"
printf '%-26s %12s  %s\n' "03_models.zip"     "$(human "$model_bytes")"  "jina-clip-v2 + e-v3"
printf '%-26s %12s  %s\n' "04_hf_modules.zip" "$(human "$hf_bytes")"     "cache trust_remote_code"
printf '%-26s %12s  %s\n' "05_config.zip"     "-"                        "env DA BO KHOA + docs"
if [ "$parts" -gt 0 ]; then
    printf '%-26s %12s  %s\n' "06_keyframes_part*.zip ($parts)" \
        "$(human "$keyframe_bytes")" "anh keyframe, cat ~$(human "$PART_BYTES")/phan"
fi
if [ "$WITH_VIDEOS" -eq 1 ]; then
    printf '%-26s %12s  %s\n' "videos/ (KHONG nen)" \
        "$(human "$video_bytes")" "mp4 copy thang, $TRANSFERS file song song"
fi
printf '%-26s %12s\n' "TONG nguon" "$(human "$source_total")"
echo
log "Spool: $SPOOL  (moi lan chi mot archive nam o day roi xoa)"
log "Dia con trong: $(human "$(df -B1 --output=avail "$SPOOL" | tail -1)")"

if [ "$PLAN_ONLY" -eq 1 ]; then echo; log "(--plan - chua lam gi)"; exit 0; fi

# Chan TRUOC khi day, khong phai giua chung: Drive khong canh bao sap day, no
# chi tra 403 storageQuotaExceeded o giua roi bo do.
#
# Do bang KICH THUOC NGUON chu khong phai kich thuoc zip. Phan lon khoi luong
# (anh JPEG, .npy, .safetensors) luu STORE nen zip xap xi nguon; rieng JSONL nen
# duoc nhieu, nen con so nay la uoc luong THUA - sai ve phia an toan.
about_free=$(rclone about "${REMOTE}:" --json 2>/dev/null \
    | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("free", -1))
except Exception: print(-1)')
if [ "${about_free:-0}" -ge 0 ]; then
    log "Drive con trong: $(human "$about_free")"
    if [ "$about_free" -lt "$source_total" ]; then
        die "Drive chi con $(human "$about_free") ma pack can ~$(human "$source_total").
  - Bo anh keyframe : ./scripts/gdrive_push.sh --skip-keyframes
  - Hoac --setup lai bang nick nhieu dung luong hon (nick truong thuong 100 GB+)"
    fi
else
    log "CANH BAO: khong doc duoc dung luong Drive (rclone about) - day thang, neu day se hong giua chung."
fi

echo

pack_and_push "01_export.zip" "$ROOT" deflate \
    "5 JSONL + manifest - BAT BUOC" storage/exports_competition
pack_and_push "02_vectors.zip" "$ROOT" store \
    "873 .npy vector jina 1024 chieu - BAT BUOC" storage/processed/embeddings_pack
# Cung danh sach loai tru voi `scripts/download_hf_model.py`. `snapshot_download`
# trong setup_vastai_clean.sh keo NGUYEN repo HF, nen jina-clip-v2 tren dia phinh
# 14 GB: rieng thu muc onnx/ da ~10 GB (6 ban fp16/q4/bnb4/uint8/quantized), cong
# pytorch_model.bin 1,7 GB trung noi dung voi model.safetensors.
# `AutoModel.from_pretrained` chi doc safetensors; da soat online/ va pyproject:
# khong cho nao dung onnxruntime. Bo di dua 03_models tu ~19 GiB ve ~5,4 GiB.
PACK_EXCLUDE=(
    --exclude 'onnx/*'    --exclude '*/onnx/*'
    --exclude 'openvino/*' --exclude '*/openvino/*'
    --exclude 'coreml/*'   --exclude '*/coreml/*'
    --exclude '.eval_results/*' --exclude '*/.eval_results/*'
    --exclude '*pytorch_model.bin' --exclude '*tf_model*' --exclude '*flax_model*'
    --exclude '*.onnx' --exclude '*.onnx_data'
    --exclude '*.msgpack' --exclude '*.h5' --exclude '*.tflite'
)
pack_and_push "03_models.zip" "$ROOT" store \
    "jina-clip-v2 + jina-embeddings-v3 (bo onnx/bin) - BAT BUOC" \
    storage/models/jina-clip-v2 storage/models/jina-embeddings-v3
pack_and_push "04_hf_modules.zip" "$HF_MODULES" deflate \
    "cache trust_remote_code - BAT BUOC" transformers_modules
push_config

if [ "$parts" -gt 0 ]; then
    for i in $(seq 1 "$parts"); do
        name=$(printf '06_keyframes_part%02d.zip' "$i")
        list=$(printf '%s/part%02d.txt' "$LISTS" "$i")
        [ -f "$list" ] || continue
        if [ "$FORCE" -eq 0 ] && remote_has "$name"; then
            log "[bo qua] $name - da co tren Drive"
            continue
        fi
        want=$( cd "$ROOT" && du -scb -- $(cat "$list") 2>/dev/null | tail -1 | cut -f1 )
        need_space "${want:-0}"
        log "[dong  ] $name ($(wc -l < "$list") thu muc video, $(human "${want:-0}"))"
        bsdtar -c -f "$SPOOL/$name.part" --format zip \
               --options zip:compression=store \
               -C "$ROOT" -T "$list"
        mv "$SPOOL/$name.part" "$SPOOL/$name"
        record_manifest "$name" "$SPOOL/$name" "anh keyframe phan $i/$parts"
        log "[day   ] $name"
        rclone copyto "$SPOOL/$name" "${REMOTE}:${REMOTE_DIR}/$name" \
            --drive-chunk-size 128M --progress --stats 15s
        rm -f "$SPOOL/$name"
    done
fi

# MANIFEST di cuoi cung: no la thu duy nhat cho biet pack da tron ven.
rclone copyto "$MANIFEST" "${REMOTE}:${REMOTE_DIR}/MANIFEST.json"

# Video di SAU MANIFEST, co y: no khong nam trong MANIFEST (khong nen, khong co
# sha256 rieng - rclone tu doi chieu hash tung file khi copy). Dat truoc thi mot
# lan dut mang o 78 GB se chan mat ca phan archive von nho va quan trong hon.
if [ "$WITH_VIDEOS" -eq 1 ]; then push_videos; fi

echo
log "Xong. Doi chieu: ./scripts/gdrive_push.sh --verify"
log "Lay ve tren may Vast moi: ./scripts/bootstrap_vast_from_gdrive.sh"
