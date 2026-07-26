# 14. Kỹ thuật cần chuẩn bị

Doc này biến các mục *planned* của `docs/11_SERVER_IMPLEMENTATION.md` thành ticket
ngắn, có điều kiện nghiệm thu — không lặp lại bảng zone-by-zone đã có ở đó. Thứ tự
theo phase §5 doc 11.

## Đã làm (mẫu cho cách "làm ngay, có cờ, có ablation")

**Wire 4 module retrieval đã code sẵn nhưng chưa vào production**
(`online/adapters/ocr_fuzzy.py`, `online/services/query_prep.py`,
`online/services/query_expansion.py`, `online/services/rules.py`) vào
`online/api/container.py`, đứng sau 4 cờ env mới (`AIC_ENABLE_OCR_FUZZY`,
`AIC_ENABLE_QUERY_PREP`, `AIC_ENABLE_EXPANSION`, `AIC_ENABLE_RULES`), mặc định **tắt**
— hành vi production hôm nay không đổi cho tới khi ai đó chủ động bật + đã chạy
ablation qua `scripts/eval_kis.py` chứng minh tăng Recall@K/MRR. Đây là khuôn mẫu áp
dụng cho mọi mục "planned" dưới đây: code trước, cờ tắt mặc định, đo bằng eval_kis
(hoặc benchmark tương đương) trước khi bật thật.

## Phase 1 — Enrich production (A1–A4, B5–B9, C10–C11 qua queue)

| Ticket | Đụng module | Nghiệm thu |
|---|---|---|
| Ingest quarantine thay vì raise-dừng-cả-run | `offline/pipeline.py` (theo §4.A1 doc 11) | video lỗi vào `storage/quarantine/`, run tiếp video còn lại, ledger ghi lý do |
| Ingest incremental (skip video đã `complete`) | `offline/pipeline.py::run` | chạy lại `python -m offline run` trên corpus cũ không xử lý lại video đã xong |
| TransNetV2 tuned thay uniform-scene | worker endpoint mới `POST /v1/inference/scene` + `offline/pipeline.py` | boundary + confidence trả về, merge scene <1s / split >60s |
| Keyframe: cosine dedup + blur + motion + thumbnail | `offline/keyframe_select.py` (module mới) | `QualitySignals` không còn field null trên dữ liệu enrich mới |
| Silero VAD trước Whisper | `offline/pipeline.py` audio block | giảm hallucination đoạn im lặng — so thủ công 10 clip có/không VAD |
| Job queue Redis+RQ thay vòng for tuần tự | `offline/worker.py`, module mới `offline/queue.py` | resume giữa chừng không hỏng dữ liệu (test bằng kill -9 job đang chạy) |

## Phase 2 — Index + eval (D13–D15, mở rộng ground-truth)

| Ticket | Đụng module | Nghiệm thu |
|---|---|---|
| Qdrant HNSW tuned (m/ef_construct tường minh) + named vectors | `offline/indexing.py::QdrantIndexer` | benchmark p95 search trước/sau tune |
| Alias `aic_scenes_active` cho publish/rollback | `offline/indexing.py` | demo đổi alias không downtime, rollback về version trước |
| Elasticsearch adapter thay 4× `LexicalRetriever` in-memory | adapter mới cùng interface `Retriever` (`online/ports/interfaces.py`), wiring ở `online/api/container.py` | eval_kis số liệu ES ≥ BM25 in-memory trên cùng dev set trước khi cắt hẳn (xem §6 doc 11) |
| Mở rộng ground-truth ≥50 query/task | `examples/*.jsonl` | theo dõi ở `docs/13_PRODUCTION_READINESS_INFO.md` mục 4 |
| Parquet export cho DuckDB | `datasection/exporter.py` thêm hàm export phẳng | `GET /v1/scenes/{id}` đọc được qua DuckDB, không đổi contract JSONL hiện tại |

## Phase 3 — Online (E16–E20, F21–F23)

| Ticket | Đụng module | Nghiệm thu |
|---|---|---|
| Query parser Qwen3-14B + fallback có cờ | module mới `online/services/llm_planner.py` | fail → `parser="rule_fallback"` hiển thị trong response, không silent |
| Per-branch deadline + `branches[]` trong response | `online/services/search.py::_retrieve` | giả lập 1 nhánh treo → response vẫn trả về đúng hạn, `branches[]` báo status timeout |
| Circuit breaker quanh vLLM/ES/Qdrant | module mới, wrap ở `online/api/container.py` | giả lập ES chết → backend trả degraded, không 503 toàn phần |
| Rerank cascade (BGE top-300→50, Qwen3-VL top-20) | endpoint `POST /v1/rerank` ở worker + hook vào `search.py` | ablation rerank on/off qua eval_kis |
| VQA verifier độc lập | `online/services/vqa.py` | test case "answer không có evidence chống lưng" → abstain đúng |

## Phase 4 — UI + ops (F24, G25–G27)

| Ticket | Đụng module | Nghiệm thu |
|---|---|---|
| UI React/Vite (thay vanilla JS) | `online/ui/*` viết lại | rank/evidence/latency/branches hiển thị đúng, export frame_idx đúng format BTC |
| Preflight warmup gate | mở rộng `scripts/preflight.py` | backend chỉ nhận traffic sau khi mọi model trả lời request thử |
| **CI eval gate**: `eval_kis` chạy trước mỗi merge vào `server_implementation` | CI workflow mới (chưa có `.github/workflows/`) | PR fail nếu Recall@K giảm so với baseline lưu trong repo |
| Prometheus/Grafana + alert build_id mismatch | `infra/docker-compose.production.yml` (đã có skeleton, chưa test trên server thật) | dashboard hiển thị latency theo branch, alert bắn khi encoder/manifest lệch |

## Việc hạ tầng đứng ngoài 4 phase

- `infra/docker-compose.production.yml`: đã viết theo doc 11 nhưng **chưa test trên
  server thật** (ghi trong `[[aic2026-server-implementation]]`) — việc đầu tiên khi
  thuê được máy là chạy thử compose này, không phải viết lại.
- `docs/05_VAST_DEPLOYMENT.md`: cần đối chiếu lại với compose production mới (doc đó
  viết cho profile cũ) trước khi dùng làm hướng dẫn thuê máy thật.
