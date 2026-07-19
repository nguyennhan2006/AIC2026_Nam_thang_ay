# AIC Video Retrieval V1 — Data + Offline + Online

Bộ V1 tích hợp cho KIS, AVS, tìm chuỗi sự kiện và retrieval-grounded VQA.
Ba phần dùng chung một contract, nhưng có thể scale độc lập:

- `datasection/`: nguồn sự thật canonical (`Video → Scene → Keyframe`), manifest, checksum và validator.
- `offline/`: ingest video, scene/keyframe, caption/OCR/object/ASR, embedding và index. Có checkpoint/retry và worker GPU từ xa.
- `online/`: hybrid retrieval, temporal linking, API FastAPI, media serving và UI chạy local gọi backend Vast.ai.

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
OCR trong cùng model), OWLv2, CLIP và Whisper; pin model/revision phù hợp dữ liệu
và dùng đúng encoder đã tạo index. Xem `docs/KAGGLE_OFFLINE_GUIDE.md` để chạy full
pipeline thật trên Kaggle notebook khi máy local không đủ VRAM/RAM cho model 7B.

## Triển khai Vast.ai

```bash
cp .env.example .env
# sửa token, CORS origin, AIC_ONLINE_BACKEND=qdrant
docker compose -f infra/docker-compose.vast.yml up -d --build
curl http://HOST:PORT/v1/health
```

Chỉ public cổng backend. Qdrant và worker ở internal network. UI local nhập URL
backend Vast.ai và token Online nếu bật.

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

## Lệnh kiểm tra

```bash
python -m unittest discover -s tests -v
python -m datasection.cli storage/exports
python -m scripts.preflight
python -m compileall -q datasection offline online scripts tests
```
