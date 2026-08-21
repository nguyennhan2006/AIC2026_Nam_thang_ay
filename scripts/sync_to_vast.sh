#!/usr/bin/env bash
# Đẩy các file ĐÃ SỬA lên máy thuê (Vast.ai) bằng một lệnh.
#
#     VAST_HOST=1.2.3.4 VAST_PORT=12345 ./scripts/sync_to_vast.sh --dry-run
#     VAST_HOST=1.2.3.4 VAST_PORT=12345 ./scripts/sync_to_vast.sh
#     ./scripts/sync_to_vast.sh --since 89e768a --dry-run
#
# ĐỌC TRƯỚC: nếu code đã commit VÀ push, `git pull` trên máy thuê gọn hơn hẳn
# script này — có version, rollback được, và không phụ thuộc máy local đang mở.
# Script này dành cho hai tình huống mà git không giải quyết được:
#   * máy thuê không vào được GitHub (repo riêng, thiếu credential, chặn mạng);
#   * cần thử một sửa đổi CHƯA commit.
#
# Vì sao tar-qua-ssh chứ không phải scp từng file hay rsync:
#   * scp từng file mở một phiên SSH mỗi file — 19 file là 19 lần bắt tay, và
#     một file lỗi giữa chừng để lại cây thư mục nửa vời không ai biết.
#   * rsync không có trong Git Bash trên Windows.
#   * tar gửi MỘT luồng, giữ nguyên đường dẫn tương đối, và hoặc là bung được
#     hết hoặc là hỏng ngay từ đầu.
#
# Danh sách file lấy từ git, KHÔNG gõ tay: `git diff --name-only HEAD` cho phần
# đã theo dõi, `git ls-files --others --exclude-standard` cho file mới. Gõ tay
# là cách chắc chắn nhất để quên đúng một file rồi mất một buổi tìm lỗi.

set -euo pipefail
cd "$(dirname "$0")/.."

VAST_HOST="${VAST_HOST:-}"
VAST_PORT="${VAST_PORT:-22}"
VAST_USER="${VAST_USER:-root}"
VAST_DIR="${VAST_DIR:-/workspace/AIC2026_Nam_thang_ay}"

DRY_RUN=0
RUN_TESTS=0
SINCE=""
EXPLICIT=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --since)   SINCE="$2"; shift ;;
        --test)    RUN_TESTS=1 ;;
        --host)    VAST_HOST="$2"; shift ;;
        --port)    VAST_PORT="$2"; shift ;;
        --dir)     VAST_DIR="$2"; shift ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)         EXPLICIT+=("$1") ;;
    esac
    shift
done

# Không đẩy dữ liệu và kết quả đo: chúng nặng hàng trăm MB, đã có sẵn trên máy
# thuê, và ghi đè `storage/exports*` bằng bản stub của máy local là cách nhanh
# nhất để giết một server đang chạy tốt.
is_excluded() {
    case "$1" in
        storage/*|outputs/*|bt/*|.venv/*|node_modules/*|*/node_modules/*|\
        *.pyc|__pycache__/*|*/__pycache__/*|aic_debug_bundle/*|.env*)
            return 0 ;;
    esac
    return 1
}

# Gốc so sánh, theo thứ tự: --since người dùng đặt -> nhánh trên remote (tức
# "những gì remote CHƯA có") -> HEAD (chỉ còn thay đổi chưa commit).
if [ -z "$SINCE" ]; then
    upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
    if [ -n "$upstream" ] && [ -n "$(git log --oneline "$upstream..HEAD" 2>/dev/null)" ]; then
        SINCE="$upstream"
    else
        SINCE="HEAD"
    fi
fi

if [ ${#EXPLICIT[@]} -gt 0 ]; then
    CANDIDATES=("${EXPLICIT[@]}")
else
    echo "(so voi $SINCE)"
    mapfile -t CANDIDATES < <(
        { git diff --name-only "$SINCE"; git ls-files --others --exclude-standard; } | sort -u
    )
fi

FILES=()
SKIPPED=()
for path in "${CANDIDATES[@]}"; do
    [ -f "$path" ] || continue          # file đã xoá thì không gửi
    if is_excluded "$path"; then SKIPPED+=("$path"); continue; fi
    FILES+=("$path")
done

if [ ${#FILES[@]} -eq 0 ]; then
    echo "Khong co file nao de day." >&2
    exit 1
fi

echo "=== ${#FILES[@]} file se day len $VAST_USER@${VAST_HOST:-<chua dat>}:$VAST_DIR ==="
for path in "${FILES[@]}"; do printf '  %s\n' "$path"; done
if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo "--- bo qua ${#SKIPPED[@]} file (du lieu/ket qua do) ---"
    for path in "${SKIPPED[@]}"; do printf '  %s\n' "$path"; done
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "(--dry-run: chua gui gi)"
    exit 0
fi

if [ -z "$VAST_HOST" ]; then
    echo "THIEU VAST_HOST. Vi du:" >&2
    echo "    VAST_HOST=1.2.3.4 VAST_PORT=12345 $0" >&2
    exit 2
fi

# `-C .` để đường dẫn trong gói là tương đối gốc repo, bung ra đúng chỗ.
printf '%s\0' "${FILES[@]}" \
  | tar --null -T - -czf - -C . \
  | ssh -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" \
        "mkdir -p '$VAST_DIR' && tar -xzf - -C '$VAST_DIR' && echo '  bung xong'"

echo "=== da day ${#FILES[@]} file ==="

if [ "$RUN_TESTS" -eq 1 ]; then
    echo "=== chay test tren server ==="
    ssh -p "$VAST_PORT" "$VAST_USER@$VAST_HOST" \
        "cd '$VAST_DIR' && python -m pytest tests/ -q 2>&1 | tail -5"
fi

cat <<EOF

Code moi CHUA chay: uvicorn nap module luc khoi dong, khong doc lai file.
Khoi dong lai:

    ssh -p $VAST_PORT $VAST_USER@$VAST_HOST
    pkill -f "uvicorn online.api.app:app"
    cd $VAST_DIR && bash scripts/run_competition.sh
EOF
