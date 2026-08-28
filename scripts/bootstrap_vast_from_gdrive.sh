#!/usr/bin/env bash
# =============================================================================
# bootstrap_vast_from_gdrive.sh - Kéo pack thi đấu từ Google Drive về đúng chỗ
#
# Chiều ngược của `scripts/gdrive_push.sh`. Thay cho
# `scripts/bootstrap_vast_from_kaggle.py`; khác biệt đáng giá nhất: đường Kaggle
# tải NGUYÊN khối dataset rồi lọc trong lúc chảy, nên `--skip-keyframes` vẫn kéo
# đủ 35 GB qua mạng dù chỉ ghi ra đĩa 6,8 GB. Trên Drive mỗi archive là một file
# riêng - bỏ ảnh là thật sự không tải 28,6 GB đó.
#
# Cần trước: rclone đã nối vào Drive (xem `gdrive_push.sh --setup`) và bsdtar.
#
#   ./scripts/bootstrap_vast_from_gdrive.sh --plan
#   ./scripts/bootstrap_vast_from_gdrive.sh --skip-keyframes
#   ./scripts/bootstrap_vast_from_gdrive.sh
#   ./scripts/bootstrap_vast_from_gdrive.sh --verify-only
# =============================================================================
set -euo pipefail

ROOT="${AIC_ROOT:-/workspace/AIC2026_Nam_thang_ay}"
REMOTE="${AIC_GDRIVE_REMOTE:-gdrive}"
REMOTE_DIR="${AIC_GDRIVE_DIR:-AIC2026_pack}"
SKIP_KEYFRAMES=0
PLAN_ONLY=0
VERIFY_ONLY=0
WITH_VIDEOS=0
VIDEOS_ONLY=0
TRANSFERS=8
MODE="auto"          # auto | stream | download

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-keyframes) SKIP_KEYFRAMES=1 ;;
        --videos)         WITH_VIDEOS=1 ;;
        --videos-only)    WITH_VIDEOS=1; VIDEOS_ONLY=1 ;;
        --transfers)      TRANSFERS="$2"; shift ;;
        --plan)           PLAN_ONLY=1 ;;
        --verify-only)    VERIFY_ONLY=1 ;;
        --stream)         MODE="stream" ;;
        --download)       MODE="download" ;;
        --root)           ROOT="$2"; shift ;;
        --remote)         REMOTE="$2"; shift ;;
        --remote-dir)     REMOTE_DIR="$2"; shift ;;
        -h|--help)        sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Tham so la: $1" >&2; exit 2 ;;
    esac
    shift
done

log()  { printf '[gdrive] %s\n' "$*"; }
die()  { printf '[gdrive] DUNG: %s\n' "$*" >&2; exit 1; }
human(){ numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "$1 B"; }

# HF_HOME trên Vast phải nằm trên /workspace (volume bền vững). Đặt lệch giữa
# lúc tải và lúc chạy là kiểu hỏng âm thầm nhất: file đủ cả, transformers nhìn
# sang chỗ khác rồi đòi ra mạng.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_MODULES="$HF_HOME/modules"

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

ensure_config() {
    if ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE}:"; then
        cat >&2 <<EOF

DUNG: rclone chua co remote '${REMOTE}:'.

Chon mot trong hai:
  a) Noi lai tu dau tren may nay:
         ./scripts/gdrive_push.sh --setup
  b) Chep rclone.conf tu may da noi roi:
         scp -P <PORT> ~/.config/rclone/rclone.conf root@<HOST>:~/.config/rclone/

File rclone.conf chua refresh token Google - dung commit, dung dan len chat chung.
EOF
        exit 1
    fi
}

# --- Định tuyến --------------------------------------------------------------
#
# Đích của 04 nằm NGOÀI repo: arcname của nó là `transformers_modules/...`,
# thuộc cache HuggingFace. Giải nén vào gốc repo là hỏng âm thầm - file có mặt,
# transformers không bao giờ nhìn tới chỗ đó. Mọi archive còn lại có arcname
# tính từ gốc repo (`storage/...`) nên giải nén tại gốc repo là ra đúng cây.
destination_for() {
    case "$1" in
        04_hf_modules.zip) printf '%s\n' "$HF_MODULES" ;;
        *)                 printf '%s\n' "$ROOT" ;;
    esac
}

# --- Lấy một archive ---------------------------------------------------------

fetch_one() {
    local name="$1" dest="$2" size="$3"
    mkdir -p "$dest"

    local how="$MODE"
    if [ "$how" = "auto" ]; then
        # `rclone copy` mo nhieu luong song song nen nhanh hon han `rclone cat`
        # (mot luong), nhung can cho chua zip BEN CANH phan da giai nen. Chi
        # chon no khi dia con du thoai mai; het dia giua chung la mat ca luot.
        local free
        free=$(df -B1 --output=avail "$dest" | tail -1)
        if [ "$free" -gt $(( size * 5 / 2 )) ]; then how="download"; else how="stream"; fi
    fi

    log "$(printf '%-26s %10s -> %s [%s]' "$name" "$(human "$size")" "$dest" "$how")"

    if [ "$how" = "download" ]; then
        local tmp="$dest/.$name.part"
        rclone copyto "${REMOTE}:${REMOTE_DIR}/$name" "$tmp" \
            --multi-thread-streams 4 --transfers 1 --progress --stats 15s
        bsdtar -x -f "$tmp" -C "$dest" --no-same-owner --no-same-permissions
        rm -f "$tmp"
    else
        # Zip khong bao gio cham dia: rclone day thang vao stdin cua bsdtar.
        # `set -o pipefail` o dau file khien rclone hong cung lam ca luot hong,
        # thay vi bsdtar nuot luong cut roi bao thanh cong.
        rclone cat "${REMOTE}:${REMOTE_DIR}/$name" \
            | bsdtar -x -f - -C "$dest" --no-same-owner --no-same-permissions
    fi
}

# --- Video thô: copy thẳng, không giải nén ------------------------------------
#
# Video nằm ngoài hệ archive: `gdrive_push.sh` đẩy chúng nguyên dạng vào
# `<remote>/videos/` vì `.mp4` nén sẵn rồi. Nên chiều về cũng là `rclone copy`
# thuần — không zip để mở, không cần spool, và rclone tự bỏ qua file đã có nên
# đứt mạng chạy lại chỉ kéo tiếp phần thiếu.
fetch_videos() {
    local dest="$ROOT/storage/raw/videos"
    local size
    size=$(rclone size "${REMOTE}:${REMOTE_DIR}/videos" --json 2>/dev/null \
        | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin); print(str(d["bytes"]) + " " + str(d["count"]))
except Exception: print("0 0")')
    set -- $size
    local bytes="$1" count="$2"
    if [ "${bytes:-0}" -eq 0 ]; then
        log "Khong thay video tren ${REMOTE}:${REMOTE_DIR}/videos - bo qua."
        return 0
    fi
    log "Video: $count file, $(human "$bytes") -> $dest"
    if [ "$PLAN_ONLY" -eq 1 ]; then return 0; fi

    mkdir -p "$dest"
    local free
    free=$(df -B1 --output=avail "$dest" | tail -1)
    if [ "$free" -lt "$bytes" ]; then
        die "video can $(human "$bytes") nhung dia chi con $(human "$free")."
    fi
    rclone copy "${REMOTE}:${REMOTE_DIR}/videos" "$dest" \
        --transfers "$TRANSFERS" --checkers 16 --progress --stats 15s
}

# --- Kiểm tra ----------------------------------------------------------------

verify() {
    # Dùng lại verify() của đường Kaggle: nó kiểm ĐÚNG những đường dẫn container
    # sẽ mở, gồm cả hai thư mục `$HF_HOME/hub` mà `transformers_modules` không
    # thay được. Hai đường tải khác nhau, cùng một chuẩn "đã đủ để chạy".
    local py="$ROOT/.venv/bin/python"
    [ -x "$py" ] || py="python3"
    ( cd "$ROOT" && HF_HOME="$HF_HOME" "$py" -m scripts.bootstrap_vast_from_kaggle \
        --root "$ROOT" --verify-only )
}

# --- Chạy --------------------------------------------------------------------

log "Goc repo   : $ROOT"
log "HF modules : $HF_MODULES"
log "Nguon      : ${REMOTE}:${REMOTE_DIR}"

if [ "$VERIFY_ONLY" -eq 1 ]; then verify; exit $?; fi

ensure_tools
ensure_config

# Video khong phai archive nen khong di qua duong liet-ke-.zip ben duoi.
if [ "$VIDEOS_ONLY" -eq 1 ]; then
    fetch_videos
    exit 0
fi

# Liệt kê thẳng từ Drive thay vì dùng bảng cứng: `gdrive_push.sh` cắt ảnh thành
# `06_keyframes_part01.zip`, `..._part02.zip`... mà số phần phụ thuộc dữ liệu.
# Đọc remote là luôn khớp, kể cả khi pack được đóng lại với --part-size khác.
listing=$(rclone lsjson "${REMOTE}:${REMOTE_DIR}" 2>/dev/null) \
    || die "khong doc duoc ${REMOTE}:${REMOTE_DIR} - sai ten thu muc?"

mapfile -t rows < <(printf '%s' "$listing" | python3 -c '
import json, sys
for item in sorted(json.load(sys.stdin), key=lambda x: x["Name"]):
    if item["Name"].endswith(".zip"):
        print(item["Name"] + "|" + str(item["Size"]))
')

[ "${#rows[@]}" -gt 0 ] || die "khong thay archive .zip nao trong ${REMOTE}:${REMOTE_DIR}"

total=0
declare -a plan=()
for row in "${rows[@]}"; do
    IFS='|' read -r name size <<< "$row"
    if [ "$SKIP_KEYFRAMES" -eq 1 ] && [[ "$name" == *keyframes* ]]; then
        continue
    fi
    total=$(( total + size ))
    plan+=("$name|$(destination_for "$name")|$size")
done

[ "${#plan[@]}" -gt 0 ] || die "loc xong khong con archive nao"

printf '\n%-26s %12s  %s\n' "archive" "kich thuoc" "dich"
for row in "${plan[@]}"; do
    IFS='|' read -r name dest size <<< "$row"
    printf '%-26s %12s  %s\n' "$name" "$(human "$size")" "$dest"
done
printf '%-26s %12s\n\n' "TONG" "$(human "$total")"

# Voi --plan, goi o day chi de IN ra dong video (ham tu thoat khi PLAN_ONLY).
# Luot tai that thi video di SAU archive, o cuoi file.
if [ "$PLAN_ONLY" -eq 1 ]; then
    if [ "$WITH_VIDEOS" -eq 1 ]; then fetch_videos; fi
    log "(--plan - chua tai gi)"
    exit 0
fi

mkdir -p "$ROOT"
free=$(df -B1 --output=avail "$ROOT" | tail -1)
if [ "$free" -lt "$total" ]; then
    die "can ~$(human "$total") nhung chi con $(human "$free"). Dung --skip-keyframes hoac gan them disk."
fi

for row in "${plan[@]}"; do
    IFS='|' read -r name dest size <<< "$row"
    fetch_one "$name" "$dest" "$size"
done

# Video di SAU archive: archive nho nhung la thu quyet dinh backend chay duoc
# hay khong. Dat truoc thi mot lan dut mang o 78 GB chan mat phan quan trong.
if [ "$WITH_VIDEOS" -eq 1 ]; then echo; fetch_videos; fi

echo
verify
code=$?

cat <<EOF

Tiep theo:
  .venv/bin/python -m scripts.prepare_jina_offline
  ./scripts/run_competition.sh
EOF
exit $code
