#!/bin/bash
# ==============================================================================
# upload_embeddings.sh — Upload caption_dense artifacts từ local lên VastAI
#
# Chạy trên MÁY LOCAL (Windows/Mac/Linux)
#
# Usage:
#   bash scripts/upload_embeddings.sh
#
# ==============================================================================
set -euo pipefail

# SSH key path
SSH_KEY="$HOME/.ssh/vastai_key"

# VastAI target
VAST_HOST="root@ssh3.vast.ai"
VAST_PATH="/workspace/AIC2026_Nam_thang_ay/storage/caption_embedding_jina_v2"

# Local source
LOCAL_PATH="D:/Sinh viên CNhan/AIC/Data/AIC2026_Nam_thang_ay/storage/caption_embedding_jina_v2"

echo "=== Upload caption_dense artifacts lên VastAI ==="
echo "Local: $LOCAL_PATH"
echo "Remote: $VAST_HOST:$VAST_PATH"
echo ""

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY"
    echo "Tạo key mới:"
    echo "  ssh-keygen -t ed25519 -f $SSH_KEY -C 'asus@ChonNhan'"
    exit 1
fi

# Check if local files exist
if [ ! -f "$LOCAL_PATH/embeddings.npy" ]; then
    echo "ERROR: embeddings.npy not found at $LOCAL_PATH"
    exit 1
fi

echo "Files to upload:"
ls -lh "$LOCAL_PATH/"

echo ""
echo "Uploading..."

# Create remote directory
ssh -i "$SSH_KEY" "$VAST_HOST" "mkdir -p $VAST_PATH"

# Upload files one by one (embeddings.npy is large)
echo "Uploading manifest.json..."
scp -i "$SSH_KEY" "$LOCAL_PATH/manifest.json" "$VAST_HOST:$VAST_PATH/"

echo "Uploading scene_ids.json..."
scp -i "$SSH_KEY" "$LOCAL_PATH/scene_ids.json" "$VAST_HOST:$VAST_PATH/"

echo "Uploading embeddings.npy (large file, may take a few minutes)..."
scp -i "$SSH_KEY" "$LOCAL_PATH/embeddings.npy" "$VAST_HOST:$VAST_PATH/"

echo ""
echo "Verifying on remote..."
ssh -i "$SSH_KEY" "$VAST_HOST" "ls -lh $VAST_PATH/"

echo ""
echo "=== UPLOAD HOÀN TẤT ==="
