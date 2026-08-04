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

## Triển khai 1×A100 80GB tuần tự 3 pha (không cần 2×A100)

Bảng VRAM ở `docs/11_SERVER_IMPLEMENTATION.md` §3 giả định **enrich và serving chạy đồng
thời** nên cần 2×A100. Thực tế offline (enrich) và online (serving) không cần chạy cùng
lúc — tách theo thời gian, mỗi pha chỉ tải đúng model nó cần, thì **1×A100 80GB** (không
phải 40GB — xem cảnh báo VRAM ở Pha 2) đủ dùng cho cả 2, miễn là chấp nhận: (a) rerank tầng
2 (Qwen3-VL evidence rerank) chưa bật — bản thân nó cũng chưa có code
(`docs/14_TECHNICAL_PREPARATION.md` mục rerank cascade), nên đây không phải đánh đổi mới,
chỉ là giữ nguyên trạng thái hiện tại; (b) mỗi lần đổi pha phải dừng hẳn pha trước để giải
phóng VRAM (không chạy song song 2 pha).

### Pha 1 — Enrich cơ bản (Qwen2.5-VL-7B + OWLv2 + CLIP + Whisper + color)

`offline/gpu_engine.py::TransformersGpuEngine` load lazy từng model đúng 1 lần (mỗi
`_load_*` tự cache vào `self._qwen`/`self._object`/`self._clip`/`self._asr`, không load lại
mỗi video/frame) và cả 4 model cùng sống suốt vòng đời process worker — không phải load-
rồi-giải-phóng tuần tự trong cùng pha này, nên VRAM ổn định (không tăng dần/fragment theo
số video đã xử lý). VRAM ước tính ở steady-state: ~15-16GB (Qwen2.5-VL-7B fp16) + ~1GB
(OWLv2) + ~1.7GB (CLIP) + ~3GB (Whisper-large-v3-turbo) ≈ **~22GB** — đây vẫn là ước tính
theo thông số công bố của từng model, CHƯA đo thật trên A100; xác nhận bằng
`nvidia-smi`/preflight lúc dựng máy trước khi coi là chắc chắn.

```bash
docker compose -f infra/docker-compose.vast.yml up -d --build worker
python -m offline run           # AIC_OFFLINE_PROVIDER=remote, AIC_GPU_PROVIDER=transformers
docker compose -f infra/docker-compose.vast.yml stop worker   # GIẢI PHÓNG VRAM trước khi sang Pha 2
```

(Chưa `offline index --qdrant` ở đây — đợi Pha 2.5, sau khi caption Qwen3-VL đã gộp vào,
để không phải build index 2 lần.)

### Pha 2 — Caption làm giàu Qwen3-VL-32B (`scripts/caption_qwen3vl.py`)

**Cảnh báo VRAM**: `Qwen/Qwen3-VL-32B-Instruct` là checkpoint ~33B tham số BF16, kho HF
chính thức ~66.7GB. Cộng KV cache + CUDA context + vision-encoder activations + tiền xử lý
multimodal + bộ nhớ tạm của vLLM scheduler, tổng thực tế **sát trần 80GB**, không có nhiều
dư. Vì vậy:
- **A100 40GB: không đủ** cho BF16 nguyên bản — không dùng kết luận "1×A100" ở mục này cho
  bản 40GB.
- **A100 80GB: khả thi nhưng sát giới hạn** — PHẢI giới hạn `--max-model-len` (khuyến nghị
  8K–16K, tác vụ caption không cần context dài) và bắt đầu concurrency thấp (xem
  `AIC_QWEN3VL_MAX_WORKERS` dưới), verify bằng smoke test thật trước khi chạy full corpus,
  không giả định trước.
- Muốn dư VRAM hơn: quantize (AWQ/FP8) — phải benchmark lại chất lượng caption trước khi
  dùng cho enrich thật (chất lượng caption ảnh hưởng trực tiếp mọi branch retrieval sau).

Vì cần gần hết 80GB, đây là lý do PHẢI dừng worker Pha 1 trước (chạy chung sẽ tràn VRAM:
22GB + 66GB+KV > 80GB).

```bash
# Cài vllm trên máy thuê (không nằm trong pyproject.toml của repo này — đây là service
# riêng, không phải dependency của online/offline package):
pip install vllm
vllm serve Qwen/Qwen3-VL-32B-Instruct --port 8001 --dtype bfloat16 --max-model-len 16384

# Ở .env: AIC_QWEN3VL_PROVIDER=vllm, AIC_QWEN3VL_SERVER_URL=http://127.0.0.1:8001/v1,
# AIC_QWEN3VL_MODEL=Qwen/Qwen3-VL-32B-Instruct. Giữ AIC_QWEN3VL_MAX_WORKERS=1 cho lần chạy
# đầu tiên — chỉ tăng dần (1 -> 2 -> 4...) sau khi xác nhận từng bước: peak VRAM ổn định,
# không OOM, throughput thật sự tăng, latency đuôi không tăng bất thường, output không bị
# retry hàng loạt. Chạy thử LIMIT nhỏ trước (AIC_QWEN3VL_LIMIT=1 hoặc vài scene) để verify
# JSON parse_ok + chất lượng caption trước khi để RỖNG (không xoá dòng) LIMIT để chạy hết:
python -m scripts.caption_qwen3vl
python -m scripts.import_qwen3vl_captions --captions storage/exports/qwen3vl_captions/scene_captions_selfhosted.jsonl --export-dir storage/exports

# Xong thì dừng vLLM để giải phóng VRAM trước khi sang Pha 2.5 (Ctrl+C hoặc kill process).
```

### Pha 2.5 — Finalize dataset (merge, validate, index)

Ba pha enrich (1, 2) tạo ra output từ nhiều model khác nhau nhưng **chưa chắc đã là một
dataset thống nhất, sẵn sàng cho online search** cho tới khi:

```bash
python -m datasection.cli storage/exports          # validate coverage/checksum của export đã gộp Qwen3-VL
python -m scripts.preflight --export-dir storage/exports --check-gpu-warmup
python -m offline index --encoder remote --qdrant  # build/publish index Qdrant SAU khi đã có caption Qwen3-VL
```

Chạy `offline index` ở đây (không phải cuối Pha 1) để tránh phải index lại lần 2 — dù vector
ảnh (CLIP) trong `scene_rows_remote` không phụ thuộc caption text nên về mặt kỹ thuật thứ tự
này không bắt buộc, nhưng gộp về 1 lần cho rõ ràng, tránh nhầm "đã sẵn sàng" khi dữ liệu
online thật ra vẫn còn thiếu bước merge.

### Pha 3 — Online serving

Không phải hoàn toàn "không cần VRAM" — cần làm rõ theo backend:
- `AIC_ONLINE_BACKEND=qdrant`: mỗi query vẫn cần encode qua `RemoteTextEncoder` gọi
  `/v1/embed/text` — tức là CLIP text encoder (~1.7GB) phải chạy ở đâu đó (GPU hoặc CPU với
  latency cao hơn tương ứng), không phải "0 VRAM".
- `AIC_ONLINE_BACKEND=local`: `HashingTextEncoder` chạy thuần CPU, thật sự không cần GPU.

Qwen3-14B query parser và BGE reranker **chưa có code**
(`docs/14_TECHNICAL_PREPARATION.md` — `online/services/llm_planner.py` không tồn tại,
`POST /v1/rerank` chưa có ở worker) nên serving hôm nay không cần thêm VRAM cho 2 việc đó —
đây là phần đúng của khẳng định trước, chỉ riêng phần embedding-cho-query là cần làm rõ lại.

Về kinh tế: giữ nguyên A100 80GB chỉ để chạy online lightweight (đặc biệt nếu dùng backend
`local`, hoàn toàn không cần GPU) không hiệu quả chi phí. Sau khi Pha 2.5 xong và dữ liệu đã
lên Qdrant, cân nhắc: đồng bộ `storage/exports`, tắt A100, chạy online backend trên máy
local hoặc một instance rẻ hơn (chỉ cần CPU nếu dùng backend `local`, hoặc GPU nhỏ nếu vẫn
muốn CLIP text encoder thật cho `qdrant`).

```bash
docker compose -f infra/docker-compose.vast.yml up -d --build backend
python -m scripts.preflight --export-dir storage/exports --check-gpu-warmup
```

### Vì sao KHÔNG chạy Pha 1 + Pha 2 song song trong cùng 1 lần enrich

Có thể sẽ muốn hỏi "sao không caption bằng Qwen3-VL-32B luôn trong `offline run` cho đỡ 1
bước" — không được, vì `offline/gpu_engine.py::_load_qwen` hard-code
`Qwen2_5_VLForConditionalGeneration` (transformers, in-process `.generate()`, có fail-fast
check nếu `AIC_CAPTION_MODEL` không thuộc họ Qwen2.5-VL), còn `scripts/caption_qwen3vl.py`
gọi Qwen3-VL-32B qua HTTP OpenAI-compatible (vLLM hoặc OpenRouter) — hai đường code khác
nhau, chưa hợp nhất (xem quyết định "giữ 2 nguồn OCR/caption tách biệt" trong docstring
`scripts/import_qwen3vl_captions.py`). Việc tách pha ở trên tận dụng đúng 2 đường code có
sẵn, không cần viết thêm code mới.

## Mode an toàn khi chưa có index

Giữ `AIC_ONLINE_BACKEND=local` để kiểm tra Data/API/UI. Sau khi
`python -m offline index --qdrant` hoàn tất, đổi sang `qdrant` và restart backend.
Không để backend qdrant khởi động trước collection/vector cùng dimension.

## Network

Dùng TLS reverse proxy hoặc tunnel; không truyền token qua HTTP công cộng. Giới
hạn IP inbound nếu có thể. Không public dashboard Qdrant. Rotate token khi log
hoặc shell history có nguy cơ lộ.
