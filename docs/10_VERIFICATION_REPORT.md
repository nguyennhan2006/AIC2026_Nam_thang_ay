# 10. Verification report — 2026-07-14

## Đã chạy và đạt

| Kiểm tra | Kết quả |
|---|---|
| Unit/integration | 12/12 pass |
| Canonical Data → Online | pass |
| OCR KIS + best keyframe/timestamp | pass |
| Ordered sequence | pass |
| ASR cross-scene clipping | pass |
| Export checksum/tamper | pass |
| Qdrant UUID/named vector payload | pass (mock HTTP) |
| Local index + manifest publish | pass |
| Preflight local build | ready |
| Python compileall | pass |
| JavaScript syntax | pass |
| Wheel build + isolated import | pass |

Wheel: `aic_video_retrieval_v1-1.0.0-py3-none-any.whl`.

## Chưa thể xác nhận trong môi trường tạo bundle

- FastAPI/Uvicorn runtime không có sẵn nên HTTP API chưa được khởi động tại đây;
  dependencies và Docker image đã khai báo.
- Không có Docker daemon nên compose chưa được start thực tế.
- Không có Qdrant/Vast.ai endpoint thật nên live collection, TLS, CORS qua internet,
  GPU models, VRAM và throughput chưa được đo.
- Model weights và dữ liệu AIC thật không kèm bundle; quality benchmark chưa thể chạy.

Các mục này là acceptance bắt buộc trên hạ tầng đích trước khi coi là production.
Chạy `scripts/preflight.py`, golden queries và `scripts/load_test.py` theo
`docs/06_OPERATIONS_SECURITY.md`.
