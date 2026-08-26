#!/bin/bash
# ==============================================================================
# run_server.sh — Chạy AIC2026 server trên VastAI
#
# Usage:
#   bash scripts/run_server.sh
#
# ==============================================================================
set -euo pipefail

cd /workspace/AIC2026_Nam_thang_ay

# Kill existing server
pkill -f "uvicorn online.api.app" 2>/dev/null || true
sleep 2

# Export env
export AIC_ENV_FILE=.env.fpt.local
export HF_HOME=/workspace/.hf_home
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting server at $(date)..."

# Run in background
nohup python -m uvicorn online.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --timeout-keep-alive 300 \
  > /workspace/server.log 2>&1 &

echo "Server PID: $!"
echo "Log: /workspace/server.log"
echo "Waiting for startup..."

# Wait for server to be ready
for i in {1..60}; do
    if curl -s http://127.0.0.1:8000/v1/startup > /dev/null 2>&1; then
        echo "Server is UP after ${i}s"
        exit 0
    fi
    sleep 5
done

echo "Server may not be ready. Check /workspace/server.log"
tail -30 /workspace/server.log

# Quick fix env - thêm FPT models vào .env.fpt.local
