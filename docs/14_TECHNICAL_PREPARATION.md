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

## TECH-DEBT: package `schemas/` ở gốc repo

**Phát hiện lúc nào**: khi mở rộng `ColorFeature` cho Search Mixing Console W1
(commit "feat(offline): add CPU color feature extraction") — `tests/test_keyframe_schema.py`/
`test_scene_schema.py` import từ `schemas.*` (gốc repo), một bản **trùng lặp cũ**
của `datasection/schemas/` đã lệch nhau từ trước (thiếu cả validation UUID cho
Qdrant `vector_id` mà `offline/indexing.py` đã dựa vào) — 2 test đó vô tình test
nhầm bản cũ, che giấu bug cho tới khi regenerate `contracts/*.schema.json` từ bản
canonical.

**Đã làm ngay (an toàn, không phá gì)**:
- Sửa `tests/test_keyframe_schema.py`/`test_scene_schema.py` import đúng
  `datasection.schemas.*` + sửa 1 test dùng `vector_id` không phải UUID (bug có sẵn,
  bị bản cũ che giấu).
- Chuyển `schemas/__init__.py`, `schemas/common.py`, `schemas/keyframe.py`,
  `schemas/scene.py` thành **compatibility façade** (`from datasection.schemas.X
  import *`) — 2 cây schema không còn phát triển độc lập được nữa, mọi thay đổi ở
  `datasection/schemas/` tự động phản ánh qua `schemas/` mà không cần sửa 2 nơi.
  `schemas/README.md` (quy ước ID) giữ nguyên, vẫn đúng.

**Còn lại (chưa làm, cần xác nhận trước khi làm)**: xoá hẳn thư mục `schemas/`.
Tại thời điểm chuyển sang façade, `rg "from schemas\.|import schemas\b" --type py`
đã cho kết quả rỗng (không còn ai import trực tiếp) — nhưng có thể có script/notebook
ngoài version control (vd trên máy Kaggle) vẫn trỏ vào đây. Chỉ xoá khi:
1. Chạy lại `rg "from schemas\.|import schemas\b"` vẫn rỗng sau một khoảng thời gian.
2. Xác nhận không notebook/script rời nào (kể cả trên Kaggle) còn import `schemas.*`.

## TECH-DEBT: chưa có persistent frame-embedding cache

**Phát hiện lúc nào**: khi viết `offline/embedding_reader.py` cho clip pooling
(Search Mixing Console W1). `ProviderEmbeddingReader` gọi lại
`provider.image("embedding", ...)` cho mỗi keyframe cần dùng — không có nơi nào lưu
embedding của một keyframe sau khi tính xong, kể cả `Keyframe.embedding_refs` (field
này tồn tại trong schema nhưng chưa được `offline/pipeline.py` populate cho keyframe
visual embedding).

**Hệ quả hiện tại**: `offline/indexing.py::scene_rows_remote` (dựng vector scene) và
clip pooling (dựng vector clip) đều tự gọi lại encoder cho cùng một keyframe thay vì
tái dùng — với `MemoizedEmbeddingReader` chỉ cache trong phạm vi một lần `process_video`
(không cache giữa scene-indexing và clip-pooling, không cache giữa các lần chạy khác
nhau). Baseline V1 chấp nhận việc này (đúng quyết định của user khi duyệt clip pooling:
"MemoizedEmbeddingReader tránh recompute trong cùng run, persistent frame cache để
thành tech debt riêng") — không phải bug, nhưng sẽ tốn thêm lời gọi encoder không cần
thiết khi event/temporal search (W1 bước sau) cũng cần cùng loại vector.

**Việc cần làm sau (chưa làm)**: lưu embedding keyframe xuống một vị trí ổn định
(file hoặc index) và populate `Keyframe.embedding_refs` ngay lúc enrich keyframe
trong `offline/pipeline.py`, để scene pooling, clip pooling và event aggregation sau
này đọc lại cùng vector đã lưu thay vì tính lại mỗi lần. Nghiệm thu: đo số lần gọi
`provider.image("embedding", ...)` trước/sau trên cùng một video — phải giảm khi
chạy lại toàn bộ pipeline (scene index + clip pooling) trên dữ liệu đã enrich.

## TECH-DEBT: `dense_visual_clip` chưa wire — cần `clip_rows_remote` + CLI mở rộng

**Đính chính**: bản ghi trước đây ở mục này nói "Qdrant indexing pipeline không có
entry point chạy được" — **sai**, đã tự kiểm lại và xin lỗi vì kết luận vội. `python -m
offline index --scenes storage/exports/scenes.jsonl --qdrant --encoder remote` (xem
`offline/cli.py::_main`) là entry point **thật, chạy được**: gọi `scene_rows_remote`
(encoder thật qua `RemoteInferenceProvider`) hoặc `scene_rows` (encoder `local`/hash,
không cần GPU), `QdrantIndexer.provision`/`upsert`, rồi publish `IndexArtifact` vào
`dataset_manifest.json`. Đã xác minh thật bằng cách gọi trực tiếp Qdrant Cloud instance
trong `.env` (`AIC_QDRANT_URL`) lúc audit phiên này: instance sống, credential đúng,
nhưng **0 collection tồn tại** — nghĩa là CLI này có thật nhưng **chưa từng được chạy**
trên dữ liệu thật, không phải "không tồn tại". Grep trước đó bị sót vì chỉ tìm trong
`scripts/`/`online/`, không tìm `offline/cli.py`.

**Việc còn thiếu thật (chỉ áp dụng cho `dense_visual_clip`, không áp dụng cho
`dense_visual_scene` — nhánh đó đã có CLI xong)**:
1. Viết `clip_rows_remote` trong `offline/indexing.py` (đọc `clips.jsonl` + resolve
   `embedding_refs[0].storage_locations[0].vector_uri`), theo đúng mẫu
   `scene_rows_remote`.
2. Thêm `--clip`/collection riêng vào `offline/cli.py`'s `index` subcommand (hoặc
   script riêng), nghiệm thu: `curl` collection Qdrant thấy đúng số điểm = clip_count.
3. Wire `DenseRetriever` thứ hai (đổi tên/modality qua constructor, xem
   `online/adapters/dense_retriever.py`) trỏ vào clip collection, sau `AIC_ENABLE_
   CLIP_DENSE` (cờ mới, mặc định tắt).
4. **Quan trọng**: `dense_visual_clip` chỉ có ý nghĩa ở backend `qdrant` (encoder
   query và encoder ảnh phải cùng không gian embedding thật) — KHÔNG wire ở backend
   `local` (ở đó query được encode bằng `HashingTextEncoder`, một token-hash không
   liên quan gì tới không gian ảnh của `MockInferenceProvider`/`RemoteInferenceProvider`
   — cosine similarity giữa hai không gian không liên quan sẽ là nhiễu, không phải
   kết quả yếu, nên KHÔNG được coi là "baseline chấp nhận được" như cách
   `local_dense` hiện tại chấp nhận được (nó hash TEXT, không giả vờ visual)).

## TECH-DEBT: rerank cascade tầng 1/2 (BGE / Qwen3-VL) vẫn chỉ là tier 0 (rules)

Đã có (Search Mixing Console W5): `online/services/rules.py` (tier 0, bonus/penalty
tường minh) + `fuse_candidates`/`weighted_rrf` hỗ trợ per-branch weight/enable qua
`SearchOptions.branches`, 5 phương thức fusion (`rrf`/`weighted_sum`/`max_score`/
`intersection`/`union` — 4 phương thức sau dùng contribution theo rank, không phải
raw-score-normalized weighted sum thật, xem docstring `online/services/fusion.py`).

Chưa có (đã ghi ở Phase 3 phía trên, nhắc lại vì liên quan trực tiếp
`RerankOptions.text`/`RerankOptions.vlm` đã có contract từ W0): tier 1 (BGE
reranker top-300→50) và tier 2 (Qwen3-VL top-20) cần một model server thật
(`POST /v1/inference/rerank` ở `offline/gpu_engine.py`/`offline/worker.py`, theo
đúng mẫu `RemoteTextEncoder`/`RemoteInferenceProvider` đã có) — chưa tồn tại. Endpoint
`GET /v1/search/capabilities` báo trung thực `rerank.text`/`rerank.vlm` đều `false`
cho tới khi việc này xong, không hard-code `true`.

## TECH-DEBT: React UI mới dừng ở backend capabilities, chưa có Advanced/Expert tier

`GET /v1/search/capabilities` (mới, real, introspect trực tiếp
`search_service.retrievers` — không hard-code danh sách branch) là contract mà UI
cần để dựng branch mixer, nhưng `online/ui-react/` **chưa đọc endpoint này** và chưa
có Simple/Advanced/Expert tier, branch card, evidence inspector theo đúng mục 15 của
plan gốc. Việc UI cần làm trước khi coi W7 xong: gọi `/v1/search/capabilities` lúc
khởi động, dựng danh sách branch động thay vì hard-code, thêm control cho
`SearchOptions.branches[*].weight/top_k/enabled` đã có contract sẵn ở backend.
