#!/usr/bin/env bash
# Khoi dong backend tren corpus thi dau, ban LINUX (Vast.ai / may thue GPU).
#
#     ./scripts/run_competition.sh
#
# Ban song sinh cua scripts/run_competition.ps1. Khac ba diem, deu la khac biet
# THAT giua hai moi truong, khong phai tuy tien:
#   - bind 0.0.0.0 (mac dinh) thay vi 127.0.0.1: may thue truy cap qua mang;
#   - va config.json bang scripts/prepare_jina_offline.py thay vi va tay;
#   - canh bao khi API mo ra Internet ma khong co token.
#
# Khoi dong mat ~4 phut. Doi dong "Application startup complete."

set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# .venv cua du an neu co, khong thi python3 cua image.
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

# --- Kiem tra truoc khi ton 4 phut khoi dong -------------------------------
missing=0
for f in \
    "storage/exports_competition/scenes.jsonl" \
    "storage/exports_competition/keyframes.jsonl" \
    "storage/exports_competition/videos.jsonl" \
    "storage/exports_competition/events.jsonl" \
    "storage/models/jina-clip-v2/config.json" \
    "storage/models/jina-embeddings-v3/config.json"
do
    if [ ! -f "$f" ]; then
        echo "THIEU: $f" >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "Tai du lieu bang: python -m scripts.bootstrap_vast_from_kaggle" >&2
    exit 1
fi

if [ ! -f ".env.fpt.local" ]; then
    echo "THIEU .env.fpt.local (nam trong 05_config.zip cua dataset Kaggle)." >&2
    exit 1
fi

# jina-clip-v2 tai CODE mo hinh tu mot repo HuggingFace KHAC qua trust_remote_code.
# Thieu cache do -> container chet giua chung startup bang mot OSError kho doc.
# Kiem o day (khong dung mang, khong tai gi) de biet TRUOC khi ton 4 phut.
if ! "$PYTHON" -m scripts.prepare_jina_offline --verify-only; then
    echo "" >&2
    echo "Cache code mo hinh chua du. May con mang thi chay:" >&2
    echo "    $PYTHON -m scripts.prepare_jina_offline" >&2
    echo "Xem docs/36_CHAY_HE_THONG.md muc 11-12." >&2
    exit 1
fi

# --- Cau hinh ---------------------------------------------------------------
# Y nghia tung bien: docs/36_CHAY_HE_THONG.md muc 4. Dung xoa dong nao ma khong
# doc muc do truoc - moi dong o day deu la mot lan da hong that.
export AIC_ENV_FILE="${AIC_ENV_FILE:-.env.fpt.local}"
export AIC_METADATA_JSONL="${AIC_METADATA_JSONL:-storage/exports_competition/scenes.jsonl}"
export AIC_VISUAL_EMBEDDING_NAME="${AIC_VISUAL_EMBEDDING_NAME:-jina_clip_v2}"
export AIC_VISUAL_EMBEDDING_MODEL="${AIC_VISUAL_EMBEDDING_MODEL:-storage/models/jina-clip-v2}"
export AIC_ENABLE_QUERY_TRANSLATION="${AIC_ENABLE_QUERY_TRANSLATION:-false}"
export AIC_BRANCH_TIMEOUT_MS="${AIC_BRANCH_TIMEOUT_MS:-30000}"
export AIC_ENABLE_OCR_BRANCH="${AIC_ENABLE_OCR_BRANCH:-true}"
export AIC_ENABLE_OCR_FUZZY="${AIC_ENABLE_OCR_FUZZY:-true}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# 05_config.zip da BO KHOA truoc khi len Kaggle, nen tren may thue
# AIC_ONLINE_API_KEY thuong RONG - ma khoa rong nghia la api_key_guard tat han
# (xem online/api/app.py). Bind 0.0.0.0 voi khoa rong la mo toan bo API ra
# Internet: ai quet trung cong deu goi duoc.
if [ "$HOST" = "0.0.0.0" ] && ! grep -qE '^AIC_ONLINE_API_KEY=.+' .env.fpt.local; then
    echo "[CANH BAO] AIC_ONLINE_API_KEY rong va dang bind 0.0.0.0 - API khong co" >&2
    echo "           xac thuc. Dat khoa trong .env.fpt.local, hoac chay voi" >&2
    echo "           HOST=127.0.0.1 roi noi bang SSH tunnel (docs/36 muc 13)." >&2
fi

echo "metadata : $AIC_METADATA_JSONL"
echo "embedding: $AIC_VISUAL_EMBEDDING_NAME ($AIC_VISUAL_EMBEDDING_MODEL)"
echo "lang nghe: $HOST:$PORT"
echo "Khoi dong ~4 phut. Doi 'Application startup complete.'"
echo ""

exec "$PYTHON" -m uvicorn online.api.app:app --host "$HOST" --port "$PORT"
