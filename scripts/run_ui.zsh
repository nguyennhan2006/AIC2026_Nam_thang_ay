#!/usr/bin/env zsh
set -e

# Khoi dong UI THI DAU (React) o http://localhost:5173
#
# Chay o TERMINAL RIENG, song song voi backend.
# Backend co the chay local hoac tren Vast.ai qua SSH tunnel.
#
# Chay:
#   ./scripts/run_ui.zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
UI_DIR="$REPO_DIR/online/ui-react"

cd "$UI_DIR"

# Kiem tra Node.js
if ! command -v node >/dev/null 2>&1; then
    echo "Khong tim thay node. Cai Node.js roi chay lai."
    exit 1
fi

# Kiem tra npm
if ! command -v npm >/dev/null 2>&1; then
    echo "Khong tim thay npm. Cai Node.js/npm roi chay lai."
    exit 1
fi

# Cai node_modules neu chua co
if [[ ! -d "node_modules" ]]; then
    echo "node_modules chua co - dang chay npm install..."
    npm install
fi

# ---------------------------------------------------------
# Kiem tra dist/ co cu hon src/ hay khong
# ---------------------------------------------------------

newest_src=""
newest_dist=""

if [[ -d "src" ]]; then
    newest_src="$(
        find src -type f -print0 2>/dev/null \
        | xargs -0 stat -f "%m %Sm" -t "%Y-%m-%d %H:%M:%S" 2>/dev/null \
        | sort -n \
        | tail -n 1
    )"
fi

if [[ -d "dist" ]]; then
    newest_dist="$(
        find dist -type f -print0 2>/dev/null \
        | xargs -0 stat -f "%m %Sm" -t "%Y-%m-%d %H:%M:%S" 2>/dev/null \
        | sort -n \
        | tail -n 1
    )"
fi

if [[ -n "$newest_src" && -n "$newest_dist" ]]; then
    src_ts="$(echo "$newest_src" | awk '{print $1}')"
    dist_ts="$(echo "$newest_dist" | awk '{print $1}')"

    src_date="$(echo "$newest_src" | cut -d' ' -f2-)"
    dist_date="$(echo "$newest_dist" | cut -d' ' -f2-)"

    if (( dist_ts < src_ts )); then
        echo "[luu y] dist/ ($dist_date) cu hon src/ ($src_date)"
        echo "        Dang chay UI tu source bang npm run dev, khong dung dist/."
    fi
fi

# ---------------------------------------------------------
# Kiem tra backend localhost:8000
#
# Neu Vast.ai dang duoc tunnel:
#
# ssh -N -p <SSH_PORT> \
#   -L 8000:127.0.0.1:8000 \
#   root@<VAST_IP>
#
# thi localhost:8000 tren Mac se tro toi backend Vast.
# ---------------------------------------------------------

backend_up=false

if curl -fsS \
    --connect-timeout 2 \
    --max-time 2 \
    "http://localhost:8000/v1/health" \
    >/dev/null 2>&1; then

    backend_up=true
fi

echo ""
echo "UI:      http://localhost:5173"

if [[ "$backend_up" == true ]]; then
    echo "Backend: http://localhost:8000  (dang song)"
else
    echo "Backend: KHONG thay gi o http://localhost:8000"
    echo ""
    echo "Neu backend chay tren Vast.ai, mo TERMINAL KHAC va giu tunnel:"
    echo ""
    echo "  ssh -N -p <SSH_PORT> \\"
    echo "    -L 8000:127.0.0.1:8000 \\"
    echo "    root@<VAST_IP>"
    echo ""
    echo "Voi Vast hien tai cua ban:"
    echo ""
    echo "  ssh -N -p 41044 \\"
    echo "    -L 8000:127.0.0.1:8000 \\"
    echo "    root@173.62.207.124"
fi

echo ""
echo "Lan dau mo UI, dien trong QueryStudio:"
echo "  API base : http://localhost:8000"
echo "  Token    : AIC_ONLINE_API_KEY"
echo ""

# ---------------------------------------------------------
# Doc token tu .env.fpt.local
# ---------------------------------------------------------

ENV_FILE="$REPO_DIR/.env.fpt.local"

if [[ -f "$ENV_FILE" ]]; then
    TOKEN_LINE="$(
        grep -E '^AIC_ONLINE_API_KEY=' "$ENV_FILE" \
        | head -n 1 || true
    )"

    if [[ -n "$TOKEN_LINE" ]]; then
        TOKEN="${TOKEN_LINE#AIC_ONLINE_API_KEY=}"

        if [[ -n "$TOKEN" ]]; then
            echo "  -> Token: $TOKEN"
        else
            echo "  -> AIC_ONLINE_API_KEY dang rong:"
            echo "     de trong o Token la chay duoc."
        fi
    else
        echo "  -> Khong tim thay AIC_ONLINE_API_KEY trong .env.fpt.local"
    fi
else
    echo "  -> Khong tim thay $ENV_FILE"
fi

echo ""

# ---------------------------------------------------------
# Chay Vite/React dev server
# ---------------------------------------------------------

exec npm run dev
