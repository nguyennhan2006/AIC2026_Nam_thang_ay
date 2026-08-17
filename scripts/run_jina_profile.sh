#!/usr/bin/env sh
# Chạy online API với `dense_visual` ăn vector jina-clip-v2 thay CLIP.
#
# Ba biến dưới đây đặt TRÊN dòng lệnh chứ không nằm trong một file env riêng:
# `load_env_file(override=False)` cho biến shell thắng file, nên profile này
# chồng lên `.env.fpt.local` mà không phải nhân bản file chứa khoá thật.
#
# jina-clip-v2 là CC-BY-NC-4.0 (phi thương mại) — khác giấy phép với CLIP.
#
# Đọc `docs/20_EXPERIMENT_LOG.md` § VISUAL-01 trước khi coi đây là mặc định:
# trên bộ gold hiện tại jina HOÀ với CLIP+dịch (R@20 67 cả hai), nên lý do dùng
# nó là VẬN HÀNH (bỏ lời gọi LLM khỏi đường truy vấn), không phải chất lượng.
set -eu
cd "$(dirname "$0")/.."

: "${AIC_ENV_FILE:=.env.fpt.local}"
export AIC_ENV_FILE

# Export đa video là export DUY NHẤT mang vector jina (855 keyframe, 3 video).
# `exports_l21_enriched` chỉ có CLIP, 1 video — trỏ vào đó là khởi động hỏng
# ngay ở chốt AIC_VISUAL_EMBEDDING_NAME, không phải hỏng âm thầm.
export AIC_METADATA_JSONL="${AIC_METADATA_JSONL:-storage/exports_multivideo/scenes.jsonl}"
export AIC_VISUAL_EMBEDDING_MODEL="${AIC_VISUAL_EMBEDDING_MODEL:-storage/models/jina-clip-v2}"
export AIC_VISUAL_EMBEDDING_NAME="${AIC_VISUAL_EMBEDDING_NAME:-jina_clip_v2}"

# Text tower của jina nặng hơn CLIP nhiều và đây là CPU: đo trên máy này p50
# 4.5s/truy vấn khi chạy MỘT MÌNH, so với 294ms của CLIP. Deadline 8000ms mặc
# định là quá sát — 10 nhánh chạy song song tranh CPU đã từng đẩy CLIP từ 200ms
# lên 1.6-3.9s (xem online/services/retrieval_orchestrator.py), và nhánh vượt
# deadline thì BIẾN MẤT TRONG IM LẶNG chứ không báo lỗi.
export AIC_BRANCH_TIMEOUT_MS="${AIC_BRANCH_TIMEOUT_MS:-30000}"

exec python -m uvicorn online.api.app:app \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
