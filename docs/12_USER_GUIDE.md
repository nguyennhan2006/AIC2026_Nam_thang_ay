# 12. Hướng dẫn sử dụng bản hiện tại (branch `server_implementation`)

Tài liệu này là hướng dẫn **thực hành** — chạy được gì hôm nay, kết nối UI↔backend
đúng cách, test bằng UI hoặc `curl`. Kiến trúc/thiết kế mục tiêu nằm ở
[01_ARCHITECTURE.md](01_ARCHITECTURE.md) (tổng quan) và
[11_SERVER_IMPLEMENTATION.md](11_SERVER_IMPLEMENTATION.md) (profile thi đấu, A100,
chưa code); chạy model thật trên Kaggle xem
[KAGGLE_OFFLINE_GUIDE.md](KAGGLE_OFFLINE_GUIDE.md).

## 1. Cài đặt môi trường local

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows: .venv\Scripts\activate
pip install -e ".[api,faiss,test]"
```

Cần thêm `ffmpeg`/`ffprobe` trong PATH nếu muốn chạy offline pipeline trên video thật
(không cần nếu chỉ test online search trên dữ liệu đã export sẵn). Cài thêm
`.[gpu]` chỉ khi chạy model thật (Qwen2.5-VL, OWLv2, CLIP, Whisper) — máy không có
GPU rời thì xem [KAGGLE_OFFLINE_GUIDE.md](KAGGLE_OFFLINE_GUIDE.md) thay vì cài cục bộ.

**Không có dotenv loader** trong code — `cp .env.example .env` chỉ để tham khảo,
biến môi trường phải `export`/set thật trong shell trước khi chạy lệnh, hoặc set
trực tiếp trong lệnh (`AIC_FOO=bar python -m ...`).

## 2. Có dữ liệu để search (chọn 1 trong 3)

| Cách | Lệnh | Khi nào dùng |
|---|---|---|
| Seed demo (nhanh nhất) | `python -m scripts.seed_demo` | Chỉ để xác nhận plumbing — data giả (`L01_V001`, 3 scene) |
| Offline pipeline (mock) | đặt video vào `storage/raw/videos/<ID>.mp4` (tên dạng `L\d{2}_V\d{3}`) → `python -m offline run` | Metadata thật về scene/keyframe/timing nhưng caption/OCR là placeholder |
| Offline pipeline (model thật) | như trên nhưng `AIC_OFFLINE_PROVIDER=remote` + worker chạy Qwen2.5-VL — xem [KAGGLE_OFFLINE_GUIDE.md](KAGGLE_OFFLINE_GUIDE.md) | Caption/OCR/object/ASR thật |

Sau khi có `storage/exports/scenes.jsonl`, validate trước khi bật online:

```bash
python -m datasection.cli storage/exports
python -m scripts.preflight
```

## 3. Chạy Online backend

```bash
uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

Kiểm tra: `curl http://127.0.0.1:8000/v1/health` →
`{"status":"ok","backend":"local","scene_count":N,"dataset":"..."}`.

Backend **tự phục vụ luôn UI tĩnh** tại `/ui/` (mount `StaticFiles`, xem
[online/api/app.py](../online/api/app.py)) — mở thẳng
`http://127.0.0.1:8000/ui/` hoặc `http://localhost:8000/ui/`, **không cần chạy
thêm server nào khác** cho việc test thông thường.

## 4. Cách kết nối UI ↔ backend chuẩn (đọc kỹ mục này trước khi báo lỗi CORS)

Trình duyệt coi `http://localhost:8000` và `http://127.0.0.1:8000` là **hai origin
khác nhau** dù cùng máy cùng cổng. UI lưu địa chỉ backend trong ô "Backend" (mặc
định `http://localhost:8000`, lưu vào `localStorage`).

**Quy tắc chuẩn: origin bạn gõ trên thanh địa chỉ trình duyệt phải khớp CHỮ Y HỆT
với giá trị trong ô "Backend" của UI** — cùng là `localhost` hoặc cùng là
`127.0.0.1`, không trộn hai kiểu. Có 2 cách dùng, chọn 1:

- **Cách A — tích hợp (khuyến nghị khi test 1 máy)**: mở
  `http://localhost:8000/ui/`, giữ nguyên ô Backend mặc định
  `http://localhost:8000`. UI và API cùng origin → trình duyệt không áp dụng CORS
  luôn, không phụ thuộc `AIC_CORS_ORIGINS` cấu hình gì.
- **Cách B — UI tách rời** (`./scripts/run_local_ui.sh`, cổng 5173 — dùng khi test
  UI trỏ vào backend Vast.ai/máy khác): đây là request **cross-origin thật**, bắt
  buộc backend phải whitelist origin của UI qua `AIC_CORS_ORIGINS` (mặc định đã có
  sẵn `http://localhost:5173,http://127.0.0.1:5173` — khớp origin của
  `run_local_ui.sh`). Nếu UI chạy ở origin khác (domain khác, cổng khác), phải thêm
  origin đó vào `AIC_CORS_ORIGINS` rồi khởi động lại backend.

Nếu vẫn thấy lỗi `Access-Control-Allow-Origin`: 99% là do origin UI đang mở không
khớp một trong hai quy tắc trên — sửa ô Backend (cách A) hoặc sửa
`AIC_CORS_ORIGINS` + restart (cách B), không phải lỗi code.

`AIC_ONLINE_API_KEY` (nếu bật) yêu cầu header `Authorization: Bearer <key>` cho
mọi `/v1/*` trừ `/v1/health` — nhập vào ô "API token" của UI.

## 5. Dùng UI

1. Chọn **Loại nhiệm vụ** (KIS/AVS/Sequence/VQA), gõ **Truy vấn**, bấm Tìm kiếm.
2. Mỗi kết quả là 1 card trong lưới 3–5 cột/hàng:
   - **Checkbox góc trái header** — chọn kết quả vào khay bên phải (card viền
     xanh khi đã chọn).
   - **6 icon** dưới card — bấm để mở panel tương ứng, bấm lại để đóng:
     📊 lý do khớp (breakdown điểm từng nhánh retrieval) · 📝 Caption ·
     🔤 OCR · 🎙 ASR · 🏷 Keyword · 🎬 phát video từ đúng thời điểm.
     4 icon giữa gọi `GET /v1/scenes/{scene_id}` (cache lại, chỉ tải 1 lần/scene).
3. **Khay "Đã chọn"** bên phải: giữ nguyên qua nhiều lần tìm kiếm khác nhau
   (lưu `localStorage`) để gom shortlist dần trong lúc thi.
   - **"Tìm lại chỉ trong các video đã chọn"**: search lại đúng câu truy vấn hiện
     tại nhưng giới hạn `filters.video_ids` theo các video đã chọn.
   - **"Xuất CSV nộp bài"**: tải file `rank,video_id,frame_idx,timestamp_sec,
     scene_id,score` — `frame_idx` lấy trực tiếp từ `best_keyframe_id`
     (dạng `..._F001080` → `1080`), không cần biết `fps`.
   - **"Xoá hết"**: xoá khay, không xoá kết quả tìm kiếm đang hiển thị.

## 6. Test trực tiếp qua API (không cần UI)

```bash
# health
curl http://127.0.0.1:8000/v1/health

# KIS
curl -X POST http://127.0.0.1:8000/v1/search/kis \
  -H "Content-Type: application/json" \
  -d '{"query":"người cào muối trên cánh đồng","top_k":5}'

# AVS
curl -X POST http://127.0.0.1:8000/v1/search/avs \
  -H "Content-Type: application/json" -d '{"query":"cánh đồng muối","top_k":10}'

# Sequence (câu có mốc thời gian: sau đó/tiếp theo/cuối cùng)
curl -X POST http://127.0.0.1:8000/v1/search/sequence \
  -H "Content-Type: application/json" \
  -d '{"query":"Người cào muối, sau đó đoàn người vẫy tay, cuối cùng đứng trước căn nhà","top_k":5}'

# VQA
curl -X POST http://127.0.0.1:8000/v1/vqa \
  -H "Content-Type: application/json" -d '{"question":"Có bao nhiêu người cào muối?","top_k_evidence":5}'

# Chi tiết 1 scene (caption/OCR/ASR/keyword đầy đủ + toàn bộ keyframe)
curl http://127.0.0.1:8000/v1/scenes/L01_V001_S0000

# Ảnh/video gốc (đường dẫn lấy từ best_keyframe_path/video_path trong kết quả search)
curl -o frame.jpg http://127.0.0.1:8000/v1/media/processed/keyframes/L01_V001/frame_000150.svg
```

`filters` truyền được trong mọi request search/VQA:
`{"video_ids":[...],"scene_ids":[...],"has_ocr":true,"has_asr":false,
"start_sec_gte":0,"end_sec_lte":120}` (xem `SearchFilters` trong
[online/domain/models.py](../online/domain/models.py)).

## 7. Đánh giá chất lượng (bắt buộc trước khi kết luận cải tiến nào tốt hơn)

```bash
python -m scripts.eval_kis --metadata storage/exports/scenes.jsonl \
  --groundtruth examples/kis_groundtruth_L16_V001.jsonl --mode all
```

In ra Recall@{1,5,20,50,100}, MRR, video-Recall@100 cho 4 mode
(`metadata_only/vector_only/ocr_only/fusion`). Ablation thêm cờ
`--use-query-prep --use-rules --use-expansion`. Không tự kết luận model/rule nào
tốt hơn nếu chưa chạy lại bảng này.

## 8. Biến môi trường hay dùng

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIC_ONLINE_BACKEND` | `local` | `local` (in-memory hashing, chỉ smoke test) hoặc `qdrant` (cần `AIC_QDRANT_URL`+`AIC_EMBEDDING_URL`) |
| `AIC_METADATA_JSONL` | `storage/exports/scenes.jsonl` | file scenes online đọc — backend chỉ load 1 lần lúc khởi động, đổi file phải **restart** |
| `AIC_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | whitelist origin cross-origin — xem mục 4 |
| `AIC_ONLINE_API_KEY` | rỗng (tắt auth) | bật Bearer token cho mọi `/v1/*` trừ health |
| `AIC_CANDIDATE_LIMIT` | 100 | top-k mỗi nhánh retrieval trước khi fusion |
| `AIC_RRF_K` | 60 | hằng số k trong weighted RRF |
| `AIC_OFFLINE_PROVIDER` | `mock` | `mock` (placeholder) hoặc `remote` (gọi GPU worker thật qua `AIC_GPU_URL`) |
| `AIC_GPU_PROVIDER` | `mock` | provider của chính worker: `mock` hoặc `transformers` (Qwen2.5-VL+OWLv2+CLIP+Whisper) |
| `AIC_CAPTION_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | dùng chung cho caption **và** semantic OCR |
| `AIC_SCENE_SECONDS` | 8 | độ dài scene (uniform-cut) |

## 9. Sự cố thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Access-Control-Allow-Origin` khi bấm Tìm kiếm | origin trang UI ≠ giá trị ô Backend | Xem mục 4 |
| `404 no scene matched the query` | Query không khớp field nào đã index (bình thường nếu caption còn là mock placeholder) | Dùng query khớp `examples/kis_groundtruth*.jsonl`, hoặc chạy pipeline model thật |
| `503 vector backend is not ready` ở `/v1/health` | `AIC_ONLINE_BACKEND=qdrant` nhưng Qdrant chưa chạy/chưa healthy | Kiểm tra `docker compose` Qdrant, hoặc chuyển tạm `AIC_ONLINE_BACKEND=local` |
| `scene_count` không đổi sau khi chạy `offline run` mới | Backend chỉ load `scenes.jsonl` lúc khởi động (lifespan), không tự reload | Restart uvicorn |
| `storage/exports/*` tự nhiên quay lại data demo `L01_V001` | Chạy `python -m unittest discover` sau khi đã có metadata thật — 1 test gọi `scripts.seed_demo` ghi đè | Chạy lại `python -m offline run` (+ `offline index`) sau khi chạy test suite |
| `.env` không có tác dụng dù đã sửa | Không có dotenv loader | Set biến môi trường trực tiếp trong shell chạy lệnh |
