# AIC Video Retrieval V1 — Data + Offline + Online

Bộ V1 tích hợp cho KIS, AVS, tìm chuỗi sự kiện và retrieval-grounded VQA.
Ba phần dùng chung một contract, nhưng có thể scale độc lập:

- `datasection/`: nguồn sự thật canonical (`Video → Scene → Keyframe`), manifest, checksum và validator.
- `offline/`: ingest video, scene/keyframe, caption/OCR/object/ASR, embedding và index. Có checkpoint/retry và worker GPU từ xa.
- `online/`: hybrid retrieval, temporal linking, API FastAPI, media serving và UI
  chạy local gọi backend Vast.ai — 2 bản song song cùng tính năng: `online/ui/`
  (vanilla JS, backend tự phục vụ tại `/ui/`) và `online/ui-react/` (React/Vite,
  cần `npm install`), xem `docs/12_USER_GUIDE.md` mục 5.

## Chạy smoke test trong 2 phút

Yêu cầu Python 3.11+.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[api,test]"
python -m scripts.smoke_e2e
uvicorn online.api.app:app --host 0.0.0.0 --port 8000
```

Terminal thứ hai:

```bash
./scripts/run_local_ui.sh
```

Mở `http://localhost:5173`, giữ Backend là `http://localhost:8000`, bấm
“Kiểm tra server”, rồi tìm `Gừng cay muối mặn`.

## Chạy dữ liệu thật

Tên file video là ID canonical, ví dụ `storage/raw/videos/L01_V001.mp4`.

```bash
cp .env.example .env
export AIC_OFFLINE_PROVIDER=remote
export AIC_GPU_URL=https://HOST_VAST_AI:PORT
export AIC_GPU_API_KEY='...'
python -m offline run
python -m offline index --encoder remote --qdrant
```

`mock` chỉ dùng để xác nhận plumbing. Trên Vast.ai hoặc Kaggle (T4x2) đặt
`AIC_GPU_PROVIDER=transformers` để bật Qwen2.5-VL-7B-Instruct (caption + semantic
OCR trong cùng model), OWLv2, CLIP và Whisper; pin revision qua các biến
`AIC_*_MODEL_REVISION` trong `.env.example` và dùng đúng encoder đã tạo index. Xem
`docs/KAGGLE_OFFLINE_GUIDE.md` để chạy full pipeline thật trên Kaggle notebook khi
máy local không đủ VRAM/RAM cho model 7B.

Không có GPU rời/máy thuê thì dùng **FPT AI Marketplace** thay tạm (rerank thật,
QA answer generation qua LLM, enrichment caption/OCR qua
`scripts/enrich_keyframes_fpt.py`) — set `AIC_FPT_ENABLED=true` +
`AIC_FPT_API_KEY`, xem `docs/12_USER_GUIDE.md` mục 3b cho đầy đủ biến cần set và
cách xác nhận đã bật đúng.

## Triển khai Vast.ai

```bash
cp .env.example .env
# sửa token, CORS origin, AIC_ONLINE_BACKEND=qdrant
docker compose -f infra/docker-compose.vast.yml up -d --build
curl http://HOST:PORT/v1/health
python -m scripts.preflight --check-gpu-warmup   # gọi thử caption/ocr/object/embedding trước khi nhận traffic
```

Chỉ public cổng backend. Qdrant và worker ở internal network. UI local nhập URL
backend Vast.ai và token Online nếu bật — chi tiết đầy đủ (CORS, chọn UI, chạy
`npm run dev`/build) xem `docs/12_USER_GUIDE.md` mục 5 và checklist thuê máy ở
`docs/05_VAST_DEPLOYMENT.md`.

## Tài liệu

Đọc theo thứ tự:

1. `docs/01_ARCHITECTURE.md`
2. `docs/02_DATA_CONTRACT.md`
3. `docs/03_OFFLINE_PIPELINE.md`
4. `docs/04_ONLINE_RETRIEVAL.md`
5. `docs/05_VAST_DEPLOYMENT.md`
6. `docs/06_OPERATIONS_SECURITY.md`
7. `docs/07_ACCEPTANCE_AND_LIMITATIONS.md`
8. `docs/08_FILE_GUIDE.md`
9. `docs/09_RESEARCH_ALIGNMENT.md`
10. `docs/10_VERIFICATION_REPORT.md`
11. `docs/11_SERVER_IMPLEMENTATION.md` — thiết kế profile thi đấu (A100, chưa code)
12. `docs/12_USER_GUIDE.md` — setup, kết nối UI↔backend (cùng máy hoặc server xa),
    test API, sự cố thường gặp — **đọc trước khi chạy bản hiện tại**
13. `docs/13_PRODUCTION_READINESS_INFO.md` — checklist thông tin/checkpoint cần
    chuẩn bị trước khi lên production
14. `docs/14_TECHNICAL_PREPARATION.md` — ticket kỹ thuật theo phase, có mục "Đã làm"
    theo dõi việc nào đã wire xong
15. `docs/15_RESEARCH_AGENDA.md` — câu hỏi nghiên cứu mở, đo bằng `scripts/eval_kis.py`

Model thật trên Kaggle: `docs/KAGGLE_OFFLINE_GUIDE.md`. `scripts/caption_qwen3vl.py`
là đường khác để nâng chất lượng caption (Qwen3-VL-32B qua OpenRouter, không cần
GPU thuê) — xem mục "Đã làm" trong `docs/14_TECHNICAL_PREPARATION.md`.

`docs/AIC2026_Competition_Completion_Guide/` là bộ tài liệu tham khảo rộng hơn
(submission ops, UI thi đấu đầy đủ, reliability) cho giai đoạn hoàn thiện hệ thống
thi đấu — không phải kế hoạch đã chốt thực thi, xem README riêng trong thư mục đó.

## Lệnh kiểm tra

```bash
python -m unittest discover -s tests -v
python -m datasection.cli storage/exports
python -m scripts.preflight
python -m compileall -q datasection offline online scripts tests
```

UI React (`online/ui-react/`, riêng ecosystem Node.js):

```bash
cd online/ui-react
npm install
npm run build     # tsc -b + vite build
npx vitest run    # unit test cho exportCsv.ts (format CSV nộp bài)
```
