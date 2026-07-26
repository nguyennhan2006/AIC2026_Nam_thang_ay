# 05. Triển khai Vast.ai

Hướng dẫn này khớp với `infra/docker-compose.vast.yml` — topology **dùng được thật**
hôm nay (mock hoặc `AIC_GPU_PROVIDER=transformers` với model hiện có, xem
`offline/gpu_engine.py`). `infra/docker-compose.production.yml` (vLLM-32b/14b, ES,
Redis, rq-worker) **chưa test trên server thật** và code phía offline chưa có gì gọi
tới các service đó — KHÔNG dùng doc này để deploy compose production, xem
`docs/11_SERVER_IMPLEMENTATION.md` §5 phase plan.

## Topology khuyến nghị

Một instance GPU chạy worker và có thể chạy Online/Qdrant trong compose. Public
duy nhất cổng 8000 của Online API; worker 8010 và Qdrant 6333 chỉ internal
(`expose`, không `ports`, trong `docker-compose.vast.yml`). Gắn volume bền vững cho
`storage`, Qdrant và model cache.

## Checklist trước khi thuê máy

- **GPU/driver**: `nvidia-smi` chạy được trong instance template đã chọn; driver hỗ trợ
  CUDA khớp với `torch` cài trong `infra/Dockerfile.worker` (`pyproject.toml` extra
  `gpu`, xem `docs/13_PRODUCTION_READINESS_INFO.md` mục 3).
- **Disk**: đủ chỗ cho model cache (Qwen2.5-VL-7B + CLIP ViT-L/14 + OWLv2 +
  Whisper-large-v3-turbo ~25-30GB, volume `model_cache:/root/.cache` đã khai báo sẵn
  trong compose) cộng corpus video/keyframe/index thật — không ước tính, đo lúc dựng.
- **Port**: chỉ cần forward/public đúng 1 cổng backend (`AIC_PUBLIC_PORT`, mặc định
  8000) qua giao diện port-forward của Vast.ai — 8010 (worker) và 6333 (Qdrant) không
  bao giờ public, compose đã tự giới hạn bằng `expose`.
- **Revision model**: điền các biến `AIC_*_MODEL_REVISION` mới trong `.env` (xem
  `.env.example`) trước khi enrich full corpus — tránh checkpoint đổi ngầm giữa các lần
  chạy (đúng yêu cầu provenance, doc 13 mục 1).

## Trình tự

1. Tạo instance có Docker, NVIDIA runtime và đủ disk (xem checklist trên).
2. Copy repo, `cp .env.example .env`, điền model revision đã pin.
3. Tạo hai token khác nhau: `AIC_GPU_API_KEY`, `AIC_ONLINE_API_KEY`.
4. Đặt `AIC_CORS_ORIGINS=http://IP_LOCAL:5173` hoặc origin thực.
5. Build/start compose (`docker compose -f infra/docker-compose.vast.yml up -d --build`).
6. Chạy Offline/export/index trước khi chuyển Online sang `qdrant`.
7. Chạy `python -m scripts.preflight --export-dir storage/exports --check-gpu-warmup`
   — gọi thử caption/ocr/object/embedding qua worker thật trước khi coi backend sẵn sàng
   nhận traffic (đóng gap "model warmup" ở doc 11 §4.G25); fail thì đọc `error` theo
   từng task trong output JSON trước khi mở traffic thật.
8. Kiểm tra `/v1/health`, query smoke, thumbnail và log.

## Không cần thuê máy để làm ngay

`scripts/caption_qwen3vl.py` (Qwen3-VL-32B caption qua OpenRouter, xem
`docs/14_TECHNICAL_PREPARATION.md` mục "Đã làm") chạy được từ máy local, không cần
instance Vast.ai — chỉ cần `OPENROUTER_API_KEY` và `pip install -e .[caption-qwen3vl]`.
Việc này không thay thế worker GPU (vẫn cần cho OCR/object/embedding/ASR), chỉ giúp
nâng chất lượng caption ngay trong lúc chưa thuê máy.

## Mode an toàn khi chưa có index

Giữ `AIC_ONLINE_BACKEND=local` để kiểm tra Data/API/UI. Sau khi
`python -m offline index --qdrant` hoàn tất, đổi sang `qdrant` và restart backend.
Không để backend qdrant khởi động trước collection/vector cùng dimension.

## Network

Dùng TLS reverse proxy hoặc tunnel; không truyền token qua HTTP công cộng. Giới
hạn IP inbound nếu có thể. Không public dashboard Qdrant. Rotate token khi log
hoặc shell history có nguy cơ lộ.
