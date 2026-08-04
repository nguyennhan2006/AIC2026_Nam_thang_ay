# 08. File guide

| File/package | Mục tiêu | Ưu điểm | Điểm lưu ý |
|---|---|---|---|
| `datasection/schemas/common.py` | primitive/ID/path/vector | một quy ước chung | đổi regex là breaking |
| `video.py` | FPS và aggregate scenes | kiểm tra chéo timeline | probe sai làm reject |
| `scene.py` | aggregate keyframes | evidence nằm cùng scene | JSONL lớn hơn normalized DB |
| `dataset.py` | build/model/index manifest | rollback/audit | phải cập nhật count/version |
| `exporter.py` | atomic JSONL/checksum | tránh đọc file dở | cần disk cho file tạm |
| `offline/pipeline.py` | orchestration | resumable, provider-neutral | uniform scene là baseline |
| `offline/providers.py` | mock/remote inference | retry + idempotency | không log payload base64 |
| `offline/worker.py` | GPU API gateway | auth/concurrency | phải thay engine bằng model thật |
| `offline/indexing.py` | FAISS/local + Qdrant | snapshot + filter server | encoder/dimension phải khớp |
| `online/adapters/*` | BM25/vector/metadata | dễ đổi backend | BM25 in-memory cần RAM |
| `online/services/*` | fusion/temporal/VQA | logic không dính hạ tầng | cần benchmark trọng số |
| `online/api/*` | FastAPI composition/routes | stateless scale-out | giữ CORS/token chặt |
| `online/ui/*` | UI local | không cần build tool | token nằm localStorage |
| `infra/*` | container profiles | tái lập deploy | pin image/version khi release |
| `scripts/seed_demo.py` | canonical demo | test nhanh | không phải dataset thật |

Mỗi file có seam rõ. Team có thể thay model/index/UI mà không sửa contract; chỉ
copy file sau khi chấp nhận trách nhiệm và giới hạn nêu trên.
