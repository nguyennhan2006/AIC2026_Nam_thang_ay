#!/bin/bash
# ==============================================================================
# setup_vastai_clean.sh — Cài đặt sạch AIC2026 server trên VastAI
# Chạy một lần khi bắt đầu instance mới
#
# Usage:
#   bash scripts/setup_vastai_clean.sh
#
# ==============================================================================
set -euo pipefail

echo "=== 1. Thiết lập HF_HOME trên /workspace (tránh đầy phân vùng root) ==="
export HF_HOME=/workspace/.hf_home
mkdir -p "$HF_HOME"
echo "export HF_HOME=/workspace/.hf_home" >> ~/.bashrc

echo "=== 2. Cài đặt system dependencies ==="
apt-get update
apt-get install -y git rsync curl

echo "=== 3. Clone repo ==="
cd /workspace
if [ -d AIC2026_Nam_thang_ay ]; then
    echo "Repo đã tồn tại, pull mới nhất..."
    cd AIC2026_Nam_thang_ay
    git fetch origin
    git reset --hard origin/full-runnable
else
    echo "Clone repo mới..."
    git clone -b full-runnable https://github.com/nguyynchnngn1/AIC2026_Nam_thang_ay.git
    cd AIC2026_Nam_thang_ay
fi

echo "=== 4. Cài Python dependencies ==="
pip install -U pip setuptools wheel
pip install -e ".[api,faiss]"
pip install "transformers>=4.49,<5" "huggingface_hub>=0.24,<1"
pip install einops timm stream-unzip requests tqdm

echo "=== 5. Cài Kaggle API ==="
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"nguynchnngn1","key":"KGAT_1cc8d02aa1680245f0b8980552bfaaa4"}
EOF
chmod 600 ~/.kaggle/kaggle.json
pip install kaggle

echo "=== 6. Tải HuggingFace models cần thiết ==="

# 6a. CLIP cho dense_visual (nếu cần)
if [ ! -d "storage/models/clip-vit-large-patch14" ]; then
    echo "Tải CLIP model..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/clip-vit-large-patch14', local_dir='storage/models/clip-vit-large-patch14')
"
else
    echo "CLIP model đã có sẵn"
fi

# 6b. jina-clip-v2 cho caption_dense (nếu cần)
if [ ! -d "storage/models/jina-clip-v2" ]; then
    echo "Tải jina-clip-v2 model..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('jinaai/jina-clip-v2', local_dir='storage/models/jina-clip-v2')
"
else
    echo "jina-clip-v2 model đã có sẵn"
fi

# 6c. jina-embeddings-v3 (nếu cần)
if [ ! -d "storage/models/jina-embeddings-v3" ]; then
    echo "Tải jina-embeddings-v3 model..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('jinaai/jina-embeddings-v3', local_dir='storage/models/jina-embeddings-v3')
"
else
    echo "jina-embeddings-v3 model đã có sẵn"
fi

echo "=== 7. Chuẩn bị jina offline modules (code repo HuggingFace) ==="
# jina dùng auto_map với dấu "--" nên cần code repo riêng
python -m scripts.prepare_jina_offline

echo "=== 8. Tải caption_dense artifacts (embeddings.npy) ==="
# Tạo thư mục nếu chưa có
mkdir -p storage/caption_embedding_jina_v2

# Nếu có local artifacts, upload qua SCP:
#   scp -i ~/.ssh/vastai_key embeddings.npy root@ssh3.vast.ai:/workspace/AIC2026_Nam_thang_ay/storage/caption_embedding_jina_v2/
# Sau đó chạy convert script nếu cần:
# python scripts/convert_jina2_to_backend.py

echo "=== 9. Tạo .env.fpt.local ==="
cat > .env.fpt.local <<'EOF'
# AIC 2026 — VastAI Production

# Runtime
AIC_RUNTIME_PROFILE=fpt_acceptance
AIC_PIPELINE_VERSION=aic-v1.0.0-vastai

# Data paths
AIC_DATA_ROOT=storage
AIC_INPUT_DIR=storage/raw/videos
AIC_EXPORT_DIR=storage/exports_competition
AIC_METADATA_JSONL=storage/exports_competition/scenes.jsonl
AIC_STATE_DIR=storage/state/vastai

# FPT API (điền key thật)
AIC_FPT_ENABLED=true
AIC_FPT_BASE_URL=https://mkp-api.fptcloud.com
AIC_FPT_API_KEY=sk-eEpMgMHCWGiNHZXSR5hOe0zNAlEtpixxYUM6gcGYCv0=
AIC_FPT_TIMEOUT_SEC=90
AIC_FPT_CONNECT_TIMEOUT_SEC=10
AIC_FPT_MAX_RETRIES=3
AIC_FPT_MAX_CONCURRENCY=2

# Visual embedding — dùng local CLIP
AIC_VISUAL_EMBEDDING_PROVIDER=local
AIC_VISUAL_EMBEDDING_MODEL=storage/models/clip-vit-large-patch14

# Caption dense (jina v2) — THUMB CHO caption_dense branch
AIC_CAPTION_DENSE_INDEX=storage/caption_embedding_jina_v2
AIC_CAPTION_DENSE_ENCODER=jina_clip_v2
AIC_CAPTION_DENSE_MODEL=storage/models/jina-clip-v2

# Offline
AIC_OFFLINE_PROVIDER=mock
AIC_CAPTION_PROVIDER=fpt
AIC_CAPTION_MODEL=Qwen2.5-VL-7B-Instruct

# Search branches
AIC_ENABLE_QUERY_TRANSLATION=true
AIC_ENABLE_EXPANSION=true
AIC_ENABLE_OCR_FUZZY=true
AIC_ENABLE_OCR_BRANCH=true
AIC_ENABLE_EVENT_SEARCH=true

# Timeout — tăng cho dense branch trên CPU
AIC_BRANCH_TIMEOUT_MS=30000

# Fusion
AIC_FUSION_METHOD=norm_max
AIC_FUSION_METHOD_QA=rrf

# CORS
AIC_CORS_ORIGINS=*
AIC_ONLINE_API_KEY=

# Offline mode
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
PYTHONIOENCODING=utf-8
EOF

# Fix CRLF
sed -i 's/\r$//' .env.fpt.local

echo "=== 10. Verify setup ==="
echo "Models:"
ls -la storage/models/

echo ""
echo "Caption dense artifacts:"
ls -la storage/caption_embedding_jina_v2/ 2>/dev/null || echo "(chưa có embeddings.npy — upload qua SCP)"

echo ""
echo "Jina modules:"
ls -la ~/.cache/huggingface/modules/transformers_modules/jinaai/ 2>/dev/null || echo "THIEU jina modules"

echo ""
echo "=== SETUP HOÀN TẤT ==="
echo "Bước tiếp theo:"
echo "  1. Upload embeddings.npy qua SCP nếu chưa có"
echo "  2. Chạy server:"
echo "     AIC_ENV_FILE=.env.fpt.local nohup python -m uvicorn online.api.app:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
