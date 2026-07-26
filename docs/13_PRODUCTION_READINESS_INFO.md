# 13. Thông tin cần chuẩn bị trước khi lên máy production

Doc này là checklist **thông tin/tài nguyên** — không lặp lại thiết kế đã có ở
`docs/11_SERVER_IMPLEMENTATION.md` (topology, bảng model, GPU budget). Mỗi mục ghi:
trạng thái hôm nay, việc cần làm, và mức chặn (blocker thật sự trước khi thuê máy hay
có thể làm song song).

## 1. Checkpoint & revision cần pin

Nguyên tắc: mọi checkpoint dùng cho enrich phải ghi revision cụ thể vào
`ModelProvenance`/manifest — đổi checkpoint mà không đổi version là lỗi âm thầm khó
phát hiện (encoder cũ trộn với encoder mới trong cùng index).

| Model | Dùng ở đâu | Trạng thái | Việc cần làm | Chặn |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct | caption/VQA/rerank tầng 2 (§4.B6, F22, E19 doc 11) | chưa thử — hôm nay dùng Qwen2.5-VL-7B (`.env.example:AIC_CAPTION_MODEL`) | tải thử trên máy thuê, pin revision, benchmark VRAM thật so với ước tính ~66GB bf16 | có, trước Phase 1 §4.B6 |
| Qwen3-14B-Instruct AWQ | query parser (§4.E16) | chưa có code (`online/services/llm_planner.py` chưa tồn tại) | quyết định AWQ source (chính chủ Qwen hay tự quantize), benchmark JSON-mode | không chặn Phase 1-2, chặn Phase 3 |
| SigLIP2 | dense embedding chính (§4.B5) | chưa dùng — hôm nay CLIP ViT-L/14 (`AIC_EMBEDDING_MODEL`) | chọn checkpoint (google/siglip2-*), so kích thước vector với `AIC_QDRANT_VECTOR_NAME` hiện tại (đổi tên vector nếu đổi chiều) | chặn trước khi đổi §4.B5 |
| OpenCLIP ViT-L/14 | đối chứng ensemble (§4.B5) | chưa dùng | chọn pretrained tag (openai vs laion2b), giữ tách biệt named vector `frame_clip` | không chặn — làm song song SigLIP2 |
| TransNetV2 tuned | scene boundary (§4.A2) | chưa dùng — hôm nay uniform 8s | weight PyTorch port cộng đồng (không có bản chính chủ PyTorch) — xác minh nguồn, license, độ chính xác trước khi thay uniform | chặn trước §4.A2 |
| PaddleOCR vi+en | OCR exact (§4.B7) | chưa dùng — hôm nay OCR qua Qwen (semantic, không exact) | tải model vi+en, kiểm tra xung đột CUDA context với torch (đã ghi ở §6 doc 11 — có thể phải chạy CPU/process riêng) | chặn trước §4.B7 |
| Grounding DINO | open-vocab detection (§4.B8) | chưa dùng — hôm nay OWLv2 | chọn checkpoint, so latency với OWLv2 hiện tại | không chặn — thay thế trực tiếp |
| Whisper large-v3 (full) + WhisperX | ASR + word-align (§4.A4/B9) | chưa dùng — hôm nay `whisper-large-v3-turbo` chunk-level | WhisperX cần forced-alignment model riêng theo ngôn ngữ (vi) — xác minh có bản tiếng Việt đủ tốt không | chặn trước §4.B9 nếu cần word-level timestamp |
| BGE reranker-v2-m3 | rerank tầng 1 (§4.E19) | chưa dùng | endpoint `POST /v1/rerank` chưa tồn tại trong `offline/worker.py` | chặn trước §4.E19 |
| Elasticsearch + plugin analyzer tiếng Việt | sparse index (§4.D14) | chưa dùng — hôm nay BM25 in-memory (`online/adapters/bm25.py`) | xác nhận plugin ICU/analyzer vi chạy được trên image ES định dùng | chặn trước §4.D14 |

## 2. License / token gate

- **Hugging Face token**: cần cho gated checkpoint (Qwen3-VL/14B nếu gated, SigLIP2).
  Chuẩn bị trước khi dựng máy — không phát hiện lúc chạy job mới đi xin token.
- **pyannote** (diarization, optional theo §4.B9): pyannote 3.x yêu cầu accept
  license trên HuggingFace Hub + token riêng cho từng model trong pipeline. Chỉ cần
  nếu bật diarization — không chặn Phase 1-3 mặc định.
- **PaddleOCR / Grounding DINO / WhisperX**: license mở (Apache/MIT), không cần token,
  chỉ cần xác minh lúc dựng máy weight tải được (một số mirror bị chặn ở VN).

## 3. Hạ tầng máy thuê

- Rút từ §3 doc 11: tối thiểu 2×A100 (80GB + 40/80GB), ≥64GB RAM, NVMe ≥100GB (corpus +
  index + model cache). Việc cần làm: chốt nhà cung cấp (Vast.ai theo doc 05
  `docs/05_VAST_DEPLOYMENT.md`), xác nhận giá theo giờ cho cấu hình này trước khi thi.
- A100 không có FP8 native — nếu dùng checkpoint FP8 phải verify throughput qua Marlin
  (weight-only) lúc dựng máy, không giả định trước.

## 4. Dữ liệu ground-truth cho eval

Hiện có **14 query** ground-truth (`examples/kis_groundtruth.jsonl`: 4 dòng,
`examples/kis_groundtruth_L16_V001.jsonl`: 10 dòng) — đủ để smoke-test
`scripts/eval_kis.py` chạy không lỗi, **không đủ** để kết luận ablation có ý nghĩa
thống kê. Doc 11 §5 Phase 2 đặt mục tiêu ≥50 query/loại task (KIS/VQA/AVS/SEQUENCE)
trên dev set BTC thật.

Việc cần làm trước Phase 2:
- Lấy dev set chính thức từ BTC (video + câu hỏi mẫu) khi công bố.
- Gán nhãn `scene_ids`/`start_sec`/`end_sec` chuẩn theo format `GroundTruthItem`
  (xem docstring `scripts/eval_kis.py`) — cần người xem video thủ công, không tự động
  hoá được phần này.
- Theo dõi tiến độ mở rộng GT trong `docs/15_RESEARCH_AGENDA.md` (mọi ablation ở đó
  phụ thuộc trực tiếp vào cỡ dev set này).

## 5. Biến môi trường theo phase

Đối chiếu `online/config.py::Settings.from_env` và `.env.example` — nhóm theo phase
production (doc 11 §5), không lặp lại toàn bộ bảng env đã có, chỉ nêu nhóm **mới cần
thêm** khi làm tới phase đó:

- **Đã có, dùng ngay** (Phần 2 tài liệu này): `AIC_ENABLE_OCR_FUZZY`,
  `AIC_ENABLE_QUERY_PREP`, `AIC_ENABLE_EXPANSION`, `AIC_ENABLE_RULES` — 4 cờ mới,
  mặc định tắt, xem `docs/14_TECHNICAL_PREPARATION.md` mục "Đã làm".
- **Phase 2** (index): biến kết nối Elasticsearch (`AIC_ES_URL`...), chưa tồn tại —
  cần định nghĩa khi viết adapter ES thay `LexicalRetriever` in-memory.
- **Phase 3** (online): biến cho `llm_planner.py` (endpoint Qwen3-14B), Redis cache/
  circuit breaker — chưa tồn tại, định nghĩa cùng lúc viết module.
- **Phase 4** (ops): Prometheus/Grafana endpoint, alert webhook — chưa tồn tại.

Nguyên tắc giữ nguyên từ `Settings.from_env`: thiếu biến bắt buộc phải `raise` ngay lúc
khởi động (không silent default sai) — theo đúng policy "không silent degradation" của
sơ đồ gốc.
