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

## Triển khai 1×A100 tuần tự 3 pha (không cần 2×A100)

Bảng VRAM ở `docs/11_SERVER_IMPLEMENTATION.md` §3 giả định **enrich và serving chạy đồng
thời** nên cần 2×A100. Thực tế offline (enrich) và online (serving) không cần chạy cùng
lúc — tách theo thời gian, mỗi pha chỉ tải đúng model nó cần, thì **1×A100 80GB đủ dùng**
cho cả 2, miễn là chấp nhận: (a) rerank tầng 2 (Qwen3-VL evidence rerank) chưa bật — bản
thân nó cũng chưa có code (`docs/14_TECHNICAL_PREPARATION.md` mục rerank cascade), nên đây
không phải đánh đổi mới, chỉ là giữ nguyên trạng thái hiện tại; (b) mỗi lần đổi pha phải
dừng hẳn pha trước để giải phóng VRAM (không chạy song song 2 pha).

### Pha 1 — Enrich cơ bản (Qwen2.5-VL-7B + OWLv2 + CLIP + Whisper + color)

VRAM ước tính: ~15-16GB (Qwen2.5-VL-7B fp16) + ~1GB (OWLv2) + ~1.7GB (CLIP) + ~3GB
(Whisper-large-v3-turbo) ≈ **~22GB** — dư nhiều so với 80GB.

```bash
docker compose -f infra/docker-compose.vast.yml up -d --build worker
python -m offline run           # AIC_OFFLINE_PROVIDER=remote, AIC_GPU_PROVIDER=transformers
python -m offline index --encoder remote --qdrant   # nếu đã sẵn sàng dùng backend qdrant
docker compose -f infra/docker-compose.vast.yml stop worker   # GIẢI PHÓNG VRAM trước khi sang Pha 2
```

### Pha 2 — Caption làm giàu Qwen3-VL-32B (`scripts/caption_qwen3vl.py`)

VRAM ước tính: ~66GB bf16 + KV cache — cần gần như toàn bộ 80GB, đây là lý do PHẢI dừng
worker Pha 1 trước (chạy chung sẽ tràn VRAM: 22GB + 66GB+KV > 80GB).

```bash
# Cài vllm trên máy thuê (không nằm trong pyproject.toml của repo này — đây là service
# riêng, không phải dependency của online/offline package):
pip install vllm
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8001 --dtype bfloat16

# Ở .env: AIC_QWEN3VL_USE_OPENROUTER=false, AIC_QWEN3VL_SERVER_URL=http://127.0.0.1:8001/v1,
# AIC_QWEN3VL_MODEL=Qwen/Qwen3-VL-32B-Instruct. Chạy thử LIMIT nhỏ trước (AIC_QWEN3VL_LIMIT=1
# hoặc vài scene) để verify JSON parse_ok + chất lượng trước khi để trống LIMIT chạy hết:
python -m scripts.caption_qwen3vl
python -m scripts.import_qwen3vl_captions --captions storage/exports/qwen3vl_captions/scene_captions_selfhosted.jsonl --export-dir storage/exports

# Xong thì dừng vLLM để giải phóng VRAM trước khi sang Pha 3 (Ctrl+C hoặc kill process).
```

### Pha 3 — Online serving

VRAM cần: chỉ khi bật `AIC_ONLINE_BACKEND=qdrant` (`RemoteTextEncoder` gọi worker
`/v1/embed/text` — có thể dùng lại CLIP đã load, ~1.7GB) hoặc `local` (không cần GPU,
`HashingTextEncoder` chạy CPU). Qwen3-14B query parser và BGE reranker **chưa có code**
(`docs/14_TECHNICAL_PREPARATION.md` — `online/services/llm_planner.py` không tồn tại,
`POST /v1/rerank` chưa có ở worker) nên serving hôm nay không cần thêm VRAM cho 2 việc đó.

```bash
docker compose -f infra/docker-compose.vast.yml up -d --build backend
python -m scripts.preflight --export-dir storage/exports --check-gpu-warmup
```

### Vì sao KHÔNG chạy Pha 1 + Pha 2 song song trong cùng 1 lần enrich

Có thể sẽ muốn hỏi "sao không caption bằng Qwen3-VL-32B luôn trong `offline run` cho đỡ 1
bước" — không được, vì `offline/gpu_engine.py::_load_qwen` hard-code
`Qwen2_5_VLForConditionalGeneration` (transformers, in-process `.generate()`), còn
`scripts/caption_qwen3vl.py` gọi Qwen3-VL-32B qua HTTP OpenAI-compatible (vLLM hoặc
OpenRouter) — hai đường code khác nhau, chưa hợp nhất (xem quyết định "giữ 2 nguồn OCR/
caption tách biệt" trong docstring `scripts/import_qwen3vl_captions.py`). Việc tách pha ở
trên tận dụng đúng 2 đường code có sẵn, không cần viết thêm code mới.

## Mode an toàn khi chưa có index

Giữ `AIC_ONLINE_BACKEND=local` để kiểm tra Data/API/UI. Sau khi
`python -m offline index --qdrant` hoàn tất, đổi sang `qdrant` và restart backend.
Không để backend qdrant khởi động trước collection/vector cùng dimension.

## Network

Dùng TLS reverse proxy hoặc tunnel; không truyền token qua HTTP công cộng. Giới
hạn IP inbound nếu có thể. Không public dashboard Qdrant. Rotate token khi log
hoặc shell history có nguy cơ lộ.
