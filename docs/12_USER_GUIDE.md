# 12. Hướng dẫn sử dụng bản hiện tại (branch `server_implementation`)

Tài liệu này là hướng dẫn **thực hành** — chạy được gì hôm nay, kết nối UI↔backend
đúng cách, test bằng UI hoặc `curl`, bật FPT AI Marketplace (rerank + QA LLM), và
những lỗi vặt hay gặp trên máy Windows local. Kiến trúc/thiết kế mục tiêu nằm ở
[01_ARCHITECTURE.md](01_ARCHITECTURE.md) (tổng quan) và
[11_SERVER_IMPLEMENTATION.md](11_SERVER_IMPLEMENTATION.md) (profile thi đấu, A100,
chưa code); test thủ công đầy đủ trên dữ liệu thật L21_V001 xem
[17_MANUAL_TEST_GUIDE_L21_V001.md](17_MANUAL_TEST_GUIDE_L21_V001.md); chạy model
thật trên Kaggle xem [KAGGLE_OFFLINE_GUIDE.md](KAGGLE_OFFLINE_GUIDE.md).

Có **2 bản UI song song**, khác kiến trúc frontend VÀ khác mức độ tính năng:

- `online/ui/` (vanilla JS, không cần Node.js, backend tự phục vụ tại `/ui/`) —
  contract cũ, giữ làm fallback đơn giản, **không cập nhật theo PR-01 trở đi**.
  Chỉ dùng khi cần smoke test không cài Node.js.
- `online/ui-react/` (React/Vite, cần `npm install`) — **"Competition Retrieval
  Studio"**: task chip KIS/QA/TRAKE/AVS, chế độ Simple/Advanced, Weight Panel đọc
  `GET /v1/search/capabilities` (không hard-code branch nào — panel tự ẩn control
  nếu server chưa hỗ trợ), Results Panel (card kết quả + Sequence Viewer/Timeline
  riêng cho TRAKE), Preview & Details panel (Preview video/Evidence/Trace), Submission
  Board build/validate CSV qua backend (`/v1/submissions/*`, không tự build ở
  client), Compare Lab + System/Health đọc session trace. Có SSE thật
  (`/v1/search/stream`) bật/tắt ở Query Studio.

Dùng `online/ui-react/` cho mọi việc thật (mixing console, submission, evidence,
TRAKE/QA workspace); `online/ui/` chỉ còn phù hợp smoke test nhanh.

## 1. Cài đặt môi trường local

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows: .venv\Scripts\activate
pip install -e ".[api,faiss,test]"
```

Cần thêm `ffmpeg`/`ffprobe` trong PATH nếu muốn chạy offline pipeline trên video thật
(không cần nếu chỉ test online search trên dữ liệu đã export sẵn). Cài thêm
`.[gpu]` chỉ khi chạy model thật cục bộ (Qwen2.5-VL, OWLv2, CLIP, Whisper) — máy
không có GPU rời thì dùng FPT AI Marketplace (mục 3b) hoặc xem
[KAGGLE_OFFLINE_GUIDE.md](KAGGLE_OFFLINE_GUIDE.md).

**Không có dotenv loader** trong code — `cp .env.example .env` chỉ để tham khảo,
biến môi trường phải `export`/`$env:` thật trong shell trước khi chạy lệnh, hoặc set
trực tiếp trong lệnh (`AIC_FOO=bar python -m ...`). **Không** `cp .env.fpt.local .env`
nếu file đó có cấu hình dataset path riêng của một thí nghiệm cũ (vd trỏ tới một
export không còn tồn tại) — chỉ export/copy đúng các biến `AIC_FPT_*` bạn cần từ đó.

## 2. Có dữ liệu để search (chọn 1 trong 3)

| Cách | Lệnh | Khi nào dùng |
|---|---|---|
| Seed demo (nhanh nhất) | `python -m scripts.seed_demo` | Chỉ để xác nhận plumbing — data giả (`L01_V001`, 3 scene) |
| Offline pipeline (mock) | đặt video vào `storage/raw/videos/<ID>.mp4` (tên dạng `L\d{2}_V\d{3}`) → `python -m offline run` | Metadata thật về scene/keyframe/timing nhưng caption/OCR là placeholder |
| Offline pipeline (model thật) | như trên nhưng `AIC_OFFLINE_PROVIDER=remote` + worker Qwen2.5-VL, hoặc `python -m scripts.enrich_keyframes_fpt --env-file .env.fpt.local` (FPT VLM, không cần thuê GPU) | Caption/OCR/object/ASR thật — xem [17_MANUAL_TEST_GUIDE_L21_V001.md](17_MANUAL_TEST_GUIDE_L21_V001.md) mục 1 cho quy trình đầy đủ trên L21_V001 |

Sau khi có `storage/exports/scenes.jsonl` (hoặc `storage/exports_l21/scenes.jsonl`
nếu bạn tách riêng dataset thật), validate trước khi bật online:

```bash
python -m datasection.cli storage/exports
python -m scripts.preflight
```

## 3. Chạy Online backend

```bash
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

Nếu bạn dùng dataset thật thay vì `storage/exports/scenes.jsonl` mặc định:

```bash
$env:AIC_METADATA_JSONL = "storage/exports_l21/scenes.jsonl"   # PowerShell
```

Kiểm tra: `curl http://127.0.0.1:8000/v1/health` →
`{"status":"ok","backend":"local","scene_count":N,"dataset":"...","dataset_version":"...",
"video_count":N,"keyframe_count":N,"asr_segment_count":N,...}`.

Backend **tự phục vụ luôn UI tĩnh** tại `/ui/` (mount `StaticFiles`, xem
[online/api/app.py](../online/api/app.py)) — mở thẳng
`http://127.0.0.1:8000/ui/` hoặc `http://localhost:8000/ui/`, **không cần chạy
thêm server nào khác** cho việc test UI vanilla.

**Đổi metadata/env rồi restart không thấy tác dụng?** Backend chỉ đọc
`scenes.jsonl`/env lúc khởi động (FastAPI lifespan), không tự reload. Trên Windows
`uvicorn --reload` có thể để lại **process cũ vẫn giữ cổng 8000 dù lệnh
`Stop-Process` báo thành công** (đặc biệt nếu dùng
`-ErrorAction SilentlyContinue`) — luôn xác nhận cổng đã trống trước khi khởi động
lại:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
# Nếu vẫn còn process cũ:
Stop-Process -Id <PID> -Force
```

### 3b. Bật FPT AI Marketplace (rerank thật + QA answer generation qua LLM)

Không có GPU rời/server A100 thì FPT AI Marketplace (OpenAI-compatible) thay tạm
cho: text rerank (`bge-reranker-v2-m3` qua `/rerank`), QA answer generation qua LLM,
và enrichment caption/OCR (`scripts/enrich_keyframes_fpt.py`, xem mục 2). Set trước
khi khởi động backend:

```powershell
$env:AIC_FPT_ENABLED = "true"
$env:AIC_FPT_API_KEY = "<api key FPT thật>"
$env:AIC_FPT_RERANK_MODEL = "bge-reranker-v2-m3"     # bật rerank.text
$env:AIC_FPT_LLM_MODEL = "Qwen3.6-27B"               # bật QA answer qua LLM
```

Xác nhận: `GET /v1/search/capabilities` phải có `"rerank": {"text": true, ...}`.
Không có cách xác nhận qua endpoint cho QA LLM riêng — theo dõi field `source`
(`"fpt_llm"` vs rule-based, vd `"ocr_exact"`) trong kết quả `qa[]`, hoặc warning
`"FPT QA LLM bỏ qua ..."` nếu model/key sai.

Lưu ý: `.env.example`/`.env.fpt.local` còn khai báo `AIC_FPT_QUERY_LLM_MODEL`/
`AIC_FPT_FAST_LLM_MODEL`/`AIC_FPT_DEEP_LLM_MODEL` (3 tầng LLM dự kiến cho
query-understanding sau này) — **code hiện tại chưa đọc 3 biến này**, chỉ
`AIC_FPT_LLM_MODEL` (dùng cho QA). Đừng tưởng set 3 biến đó là đủ.

`FptQaAnswerer` chỉ nâng cấp `AIC_FPT_QA_TOP_N` (mặc định 5) evidence pack đứng
đầu mỗi câu QA — rule-based `ANSWER_TOOLS` vẫn luôn chạy làm baseline/fallback, vì
luật chấm QA tính bất kỳ dòng nào trong submission đúng cả ba (video/frame/answer),
không chỉ dòng đầu.

## 4. Cách kết nối UI ↔ backend chuẩn (đọc kỹ mục này trước khi báo lỗi CORS)

Trình duyệt coi `http://localhost:8000` và `http://127.0.0.1:8000` là **hai origin
khác nhau** dù cùng máy cùng cổng. UI lưu địa chỉ backend trong ô "Backend" (mặc
định `http://localhost:8000`, lưu vào `localStorage`).

**Quy tắc chuẩn: origin bạn gõ trên thanh địa chỉ trình duyệt phải khớp CHỮ Y HỆT
với giá trị trong ô "Backend" của UI** — cùng là `localhost` hoặc cùng là
`127.0.0.1`, không trộn hai kiểu. Có 2 cách dùng, chọn 1:

- **Cách A — tích hợp (khuyến nghị khi test 1 máy, chỉ áp dụng cho `online/ui/`
  vanilla)**: mở `http://localhost:8000/ui/`, giữ nguyên ô Backend mặc định
  `http://localhost:8000`. UI và API cùng origin → trình duyệt không áp dụng CORS,
  không phụ thuộc `AIC_CORS_ORIGINS`.
- **Cách B — UI tách rời** (`online/ui-react` qua `npm run dev`, cổng 5173 mặc
  định — dùng khi chạy React UI hoặc trỏ vào backend Vast.ai/máy khác): đây là
  request **cross-origin thật**, bắt buộc backend whitelist origin của UI qua
  `AIC_CORS_ORIGINS` (mặc định đã có sẵn
  `http://localhost:5173,http://127.0.0.1:5173`).

Nếu vẫn thấy lỗi `Access-Control-Allow-Origin`: 99% là do origin UI đang mở không
khớp một trong hai quy tắc trên — sửa ô Backend (cách A) hoặc sửa
`AIC_CORS_ORIGINS` + restart (cách B), không phải lỗi code.

`AIC_ONLINE_API_KEY` (nếu bật) yêu cầu header `Authorization: Bearer <key>` cho
mọi `/v1/*` trừ `/v1/health` — nhập vào ô "API token" của UI.

## 5. Backend trên server xa (Vast.ai) + UI chạy trên máy local

Kịch bản thi đấu thật: GPU/backend chạy trên máy thuê (Vast.ai), thao tác viên ngồi
máy local. Khác mục 4 ở chỗ đây là 2 máy vật lý khác nhau — luôn là **cách B (UI tách
rời)**, không có "cách A cùng origin" để chọn.

### Bước 1 — Trên server

1. Thuê máy, làm theo checklist `docs/05_VAST_DEPLOYMENT.md` (driver GPU, disk, port).
2. Set `AIC_CORS_ORIGINS` **trước khi khởi động** khớp đúng origin UI sẽ mở ở máy
   local (đổi sau phải restart, không hot-reload biến env) — mặc định
   `http://localhost:5173,http://127.0.0.1:5173` đã đúng cho cả 2 UI (Vite React
   cũng mặc định cổng 5173, `run_local_ui.sh` cũng vậy — nhưng **KHÔNG chạy đồng
   thời cả 2 UI** vì đụng cổng, chỉ chạy 1 trong 2).
3. Khởi động: `./scripts/run_backend.sh` (nghe `0.0.0.0:8000`) hoặc qua
   `infra/docker-compose.vast.yml`. Set `AIC_FPT_*` (mục 3b) trên server nếu muốn
   rerank/QA LLM thật. Public cổng qua giao diện port-forward của Vast.ai (không
   public thêm cổng nào khác — xem `docs/05` mục Network).
4. Xác nhận từ máy local: `curl https://<host-vast>:<port>/v1/health`.

### Bước 2 — Trên máy local, chọn 1 trong 2 UI

| UI | Lệnh | Khi nào dùng |
|---|---|---|
| Cũ (vanilla JS, `online/ui/`) | `./scripts/run_local_ui.sh` → mở `http://localhost:5173` | Không cần Node.js, smoke test nhanh, task KIS/AVS/Sequence/VQA kiểu cũ |
| Mới (React/Vite, `online/ui-react/`) | `cd online/ui-react && npm install && npm run dev` → mở URL Vite in ra (mặc định `http://localhost:5173`) | Dùng thật: Weight Panel, TRAKE workspace, Submission Board. Bản build ổn định: `npm run build` rồi `npx serve dist -l 5173` |

### Bước 3 — Trỏ UI vào server xa

Ô **"Backend"**/"API base" trong UI: điền đúng URL public của server
(`https://<host-vast>:<port>`, **không phải** `localhost`/`127.0.0.1`). Nếu server
bật `AIC_ONLINE_API_KEY`, điền cùng giá trị vào ô **"API token"**.

### Lưu ý

- TLS bắt buộc nếu public qua internet thật — xem `docs/05` mục Network.
- Lỗi CORS: sửa `AIC_CORS_ORIGINS` **phải restart backend** để có hiệu lực.
- 2 UI dùng chung `localStorage` theo **origin trình duyệt**, không theo mã nguồn.

## 6. Dùng UI React ("Competition Retrieval Studio")

1. **Query Studio** (đầu trang): chọn task bằng chip KIS/QA/TRAKE/AVS, gõ truy vấn,
   chỉnh Top-K (không còn bị giới hạn ngầm ở 5 — xem ghi chú Top-K bên dưới),
   Ctrl+Enter để tìm luôn. Chế độ **Simple** (mặc định) ẩn API base/token/stream;
   **Advanced** hiện đủ. TRAKE với ≥2 sự kiện tách bằng dấu câu sẽ hiện event chips
   ngay dưới ô truy vấn.
2. **Weight Panel** (cột trái workbench): chỉnh trọng số modality, fusion/rerank,
   TRAKE alignment (chỉ hiện khi task=TRAKE), per-step weights (khi có ≥2 sự kiện),
   và preset lưu sẵn. **Kéo slider KHÔNG tự search lại** — chỉ cập nhật "draft",
   phải bấm Tìm kiếm để áp dụng và thật sự gửi trọng số đó lên server. Panel chỉ
   hiện control mà `GET /v1/search/capabilities` báo là có hỗ trợ.
   - **`Max results / video`** (trong Fusion & Ranking): dataset hiện chỉ có 1
     video nên dedup mặc định (`max_per_video=5` cho KIS) khiến Top-K > 5 không có
     tác dụng — đặt số này (vd 20) nếu muốn nhiều hơn 5 kết quả.
3. **Results Panel** (giữa): card kết quả theo task; TRAKE có thêm Sequence
   Viewer (chuỗi ngang các bước) + Timeline riêng theo mốc giây của từng bước.
4. **Preview & Details** (cột phải, sticky): 3 tab — Preview (phát video đúng
   thời điểm), Evidence (caption/OCR/ASR/object đầy đủ), Trace (branch status +
   modality weights + tải JSON debug).
5. **Submission Board**: build/validate CSV qua backend
   (`POST /v1/submissions/build|validate`) — không tự build ở client. Xem trước
   điểm bằng `POST /v1/submissions/evaluate-local` nếu có gold nhỏ.
6. Các trang khác ở top nav: History (session trace/replay), Dataset (thống kê),
   System (health/branch status).

## 7. Test trực tiếp qua API (không cần UI)

`top_k` mặc định 20 nếu bỏ trống (khác giới hạn dedup 5/video của KIS — 2 chuyện
khác nhau, xem mục 6). Convenience endpoint tự điền `task` theo path; endpoint
thống nhất `/v1/search` bắt buộc `task` trong body.

```bash
# health (kèm dataset stats)
curl http://127.0.0.1:8000/v1/health

# capabilities — LUÔN đọc trước khi set search_options, đừng đoán branch/rerank nào có thật
curl http://127.0.0.1:8000/v1/search/capabilities

# KIS
curl -X POST http://127.0.0.1:8000/v1/search/kis \
  -H "Content-Type: application/json" \
  -d '{"query":"biển cảnh báo sạt lở nguy hiểm cạnh khu dân cư ven sông","top_k":20}'

# QA
curl -X POST http://127.0.0.1:8000/v1/search/qa \
  -H "Content-Type: application/json" \
  -d '{"query":"Biển cảnh báo ghi gì?","top_k":10}'

# TRAKE (câu có nhiều bước, tách bằng dấu câu)
curl -X POST http://127.0.0.1:8000/v1/search/trake \
  -H "Content-Type: application/json" \
  -d '{"query":"Người dân đứng nhìn vết nứt; sau đó lực lượng chức năng tới rào chắn khu vực","top_k":10}'

# AVS
curl -X POST http://127.0.0.1:8000/v1/search/avs \
  -H "Content-Type: application/json" -d '{"query":"khu vực ven sông bị sụt lún","top_k":10}'

# Endpoint thống nhất + search_options tuỳ chỉnh (giống Weight Panel gửi lên) + debug trace
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"task":"TEXTUAL_KIS","query":"...","top_k":20,"debug":true,
       "search_options":{"fusion":{"max_results_per_video":20}}}'

# Chi tiết bằng chứng của một candidate (dùng trong Evidence Inspector)
curl http://127.0.0.1:8000/v1/evidence/<candidate_id>

# Chi tiết 1 scene (caption/OCR/ASR/keyword đầy đủ + toàn bộ keyframe)
curl http://127.0.0.1:8000/v1/scenes/<scene_id>

# Build submission CSV theo đúng format BTC
curl -X POST http://127.0.0.1:8000/v1/submissions/build \
  -H "Content-Type: application/json" \
  -d '{"task":"TEXTUAL_KIS","items":[{"video_id":"L21_V001","frame_idx":1080}]}'

# Ảnh/video gốc (đường dẫn lấy từ best_keyframe_path/video_path trong kết quả search)
curl -o frame.jpg http://127.0.0.1:8000/v1/media/processed/keyframes/L21_V001/frame_000150.jpg
```

`filters` truyền được trong mọi request search:
`{"video_ids":[...],"scene_ids":[...],"has_ocr":true,"has_asr":false,
"start_sec_gte":0,"end_sec_lte":120}` (xem `SearchFilters` trong
[online/domain/models.py](../online/domain/models.py)). Endpoint `/v1/vqa` (VQA
theo contract cũ, độc lập với task QA mới) và `/v1/search/sequence` vẫn còn nhưng
đã `deprecated=True` — dùng `/v1/search/trake` thay cho sequence.

## 8. Đánh giá chất lượng (bắt buộc trước khi kết luận cải tiến nào tốt hơn)

```bash
# KIS: Recall@K/MRR/video-Recall trên 4 mode retrieval
python -m scripts.eval_kis --metadata storage/exports/scenes.jsonl \
  --groundtruth examples/kis_groundtruth_L16_V001.jsonl --mode all

# 4 task (KIS/QA/TRAKE/AVS) trên gold set L21_V001
python -m scripts.eval_tasks --metadata storage/exports_l21/scenes.jsonl
```

Ablation thêm cờ vào cả hai script: `--use-query-prep --use-rules --use-expansion
--use-rerank`. `--use-rerank` bật **cả** text rerank thật lẫn QA answer generation
qua FPT (cần `AIC_FPT_ENABLED`+`AIC_FPT_RERANK_MODEL`/`AIC_FPT_LLM_MODEL`, mục 3b)
— không set env thì `--use-rerank` sẽ báo lỗi rõ ràng thay vì âm thầm chạy không
rerank. Không tự kết luận model/rule nào tốt hơn nếu chưa chạy lại bảng này trước
và sau khi đổi.

## 9. Biến môi trường hay dùng

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIC_ONLINE_BACKEND` | `local` | `local` (in-memory) hoặc `qdrant` (cần `AIC_QDRANT_URL`+`AIC_EMBEDDING_URL`) |
| `AIC_METADATA_JSONL` | `storage/exports/scenes.jsonl` | file scenes online đọc — backend chỉ load 1 lần lúc khởi động, đổi file phải **restart** |
| `AIC_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | whitelist origin cross-origin — xem mục 4 |
| `AIC_ONLINE_API_KEY` | rỗng (tắt auth) | bật Bearer token cho mọi `/v1/*` trừ health |
| `AIC_CANDIDATE_LIMIT` | 100 | top-k mỗi nhánh retrieval trước khi fusion |
| `AIC_RRF_K` | 60 | hằng số k trong weighted RRF |
| `AIC_OFFLINE_PROVIDER` | `mock` | `mock` (placeholder) hoặc `remote` (gọi GPU worker thật qua `AIC_GPU_URL`) |
| `AIC_GPU_PROVIDER` | `mock` | provider của chính worker: `mock` hoặc `transformers` (Qwen2.5-VL+OWLv2+CLIP+Whisper) |
| `AIC_CAPTION_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | dùng chung cho caption **và** semantic OCR |
| `AIC_SCENE_SECONDS` | 8 | độ dài scene (uniform-cut) |
| `AIC_FPT_ENABLED` | `false` | bật FPT AI Marketplace — cần `AIC_FPT_API_KEY` |
| `AIC_FPT_RERANK_MODEL` | rỗng | model `/rerank` FPT — không set thì fallback `AIC_RERANK_TEXT_URL` (worker tự host) nếu có |
| `AIC_FPT_LLM_MODEL` | rỗng | model chat dùng cho QA answer generation (`FptQaAnswerer`) — **không** phải 1 trong 3 biến `AIC_FPT_QUERY/FAST/DEEP_LLM_MODEL` (chưa wire) |
| `AIC_FPT_QA_TOP_N` | 5 | số evidence pack đứng đầu mỗi câu QA được nâng cấp bằng LLM |

## 10. Sự cố thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Access-Control-Allow-Origin` khi bấm Tìm kiếm | origin trang UI ≠ giá trị ô Backend | Xem mục 4 (cùng máy) hoặc mục 5 (server xa) |
| Sửa code/env, restart nhưng backend vẫn hành xử như cũ | `uvicorn --reload` để lại process cũ giữ cổng 8000 dù `Stop-Process` báo OK | `Get-NetTCPConnection -LocalPort 8000 -State Listen` — nếu còn PID cũ, `Stop-Process -Id <PID> -Force` rồi kiểm tra lại trước khi start |
| `404 no scene matched the query` | Query không khớp field nào đã index (bình thường nếu caption còn là mock placeholder) | Dùng query khớp `examples/kis_groundtruth*.jsonl`/`examples/AIC2026_L21_V001_queries_4tasks.jsonl`, hoặc chạy pipeline model thật |
| `503 vector backend is not ready` ở `/v1/health` | `AIC_ONLINE_BACKEND=qdrant` nhưng Qdrant chưa chạy/chưa healthy | Kiểm tra Qdrant, hoặc chuyển tạm `AIC_ONLINE_BACKEND=local` |
| `scene_count` không đổi sau khi chạy `offline run`/`assemble` mới | Backend chỉ load `scenes.jsonl` lúc khởi động | Restart uvicorn (và kiểm tra cổng trống — xem dòng trên) |
| `storage/exports/*` tự nhiên quay lại data demo `L01_V001` | Chạy `python -m unittest discover` sau khi đã có metadata thật — 1 test gọi `scripts.seed_demo` ghi đè | Chạy lại `python -m offline run` (+ `offline index`) sau khi chạy test suite |
| `.env`/`.env.fpt.local` không có tác dụng dù đã sửa | Không có dotenv loader | Set biến môi trường trực tiếp trong shell chạy lệnh |
| `422 task is required` ở `POST /v1/search` | Endpoint thống nhất bắt buộc `task` trong body (khác convenience endpoint) | Thêm `"task":"TEXTUAL_KIS"` (hoặc QA/TRAKE/AVS), hoặc gọi `/v1/search/kis` v.v. |
| `422` khi set `search_options` từ UI/curl | Field đó chưa có consumer thật, bị chặn ở `UNSUPPORTED` (`online/services/capabilities.py`) để không "cấu hình giả vờ có tác dụng" | Đọc `GET /v1/search/capabilities` trước khi set, đừng đoán field nào chạy thật |
| `ssl.SSLCertVerificationError: Basic Constraints of CA cert not marked critical` khi tự viết script gọi HTTPS (không qua `FptClient`) | OpenSSL 3.x/Python 3.13+ bật `VERIFY_X509_STRICT`, một số CA/proxy máy Windows không tuân thủ chi tiết RFC 5280 này | Dùng `FptClient`/`FptClient._request_once` (đã có `_build_ssl_context()` vá đúng 1 cờ này), đừng gọi thẳng `urllib.request.urlopen` |
| QA vẫn chỉ trả answer kiểu regex (vd toàn số/1 từ ngắn) dù đã bật FPT | Thiếu `AIC_FPT_LLM_MODEL`, hoặc set nhầm 1 trong 3 biến `QUERY/FAST/DEEP_LLM_MODEL` (chưa wire) | Set đúng `AIC_FPT_LLM_MODEL` (mục 3b), kiểm tra `warnings` trong response |
