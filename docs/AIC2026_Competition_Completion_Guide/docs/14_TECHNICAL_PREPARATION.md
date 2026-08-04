# 14. Kỹ thuật cần chuẩn bị — ticket và nghiệm thu

Nguyên tắc: code sau feature flag, mặc định tắt, có test và ablation trước khi bật.

## Phase 0 — Contracts và UI skeleton

| Ticket | Module | Nghiệm thu |
|---|---|---|
| ID/provenance contracts | `schemas/*` | mọi candidate map ngược được |
| Search/branch/candidate/evidence schemas | API schemas | OpenAPI + TS types pass |
| Session/feedback/submission schemas | services + DB | restore và audit được |
| Release manifest loader | config/preflight | mismatch làm startup fail |
| React competition skeleton | `online/ui/*` | mock KIS flow E2E |
| Video/frame service | API | exact frame + neighbors đúng |
| Feature flag registry | config | profile dump được |

## Phase 1 — Enrich production

| Ticket | Module | Nghiệm thu |
|---|---|---|
| Ingest quarantine | offline pipeline | video lỗi không dừng run |
| Incremental/resume | ledger/queue | kill -9 và resume không corrupt |
| TransNetV2 + fallback | scene worker | confidence + merge/split policy |
| Keyframe quality | keyframe module | blur/motion/dedup/thumbnail đầy đủ |
| VAD + ASR | audio block | giảm hallucination silence |
| Caption multi-frame/region experimental | caption worker | human audit + retrieval delta |
| PaddleOCR vi+en | OCR worker | bbox/confidence/normalization |
| Queue Redis/RQ/Celery | worker | retries/idempotency |

## Phase 2 — Index và eval

| Ticket | Module | Nghiệm thu |
|---|---|---|
| Qdrant named vectors/HNSW | indexer | p95 benchmark |
| Alias publish/rollback | indexer | zero-downtime demo |
| ES field indexes | adapter | ≥ baseline hoặc giữ song song |
| Event index | event pipeline | search + neighbor event |
| Clip index experimental | clip pipeline | action group eval |
| GT expansion | examples/eval | stratified KIS/VQA/AVS |
| Parquet/DuckDB export | exporter | contract không đổi |
| Integrity scanner | scripts | orphan/mismatch = 0 |

## Phase 3 — Online backend

| Ticket | Module | Nghiệm thu |
|---|---|---|
| Raw query + parser fallback | query service | Q0 luôn search |
| Query variants + drift warning | expansion | variant provenance |
| Entity toggle/must-match | API/service | UI rerun đúng |
| Per-branch deadline/status | search | timeout không làm full 503 |
| Circuit breaker | wrappers | failure injection pass |
| Fusion/dedup | search | per-query trace |
| BGE rerank | rerank endpoint | ablation + p95 |
| VLM rerank | GPU worker | only top-N, fallback |
| Event/temporal endpoints | temporal service | neighbor/ordered search |
| Evidence pack | evidence service | token/hit tradeoff |
| VQA router/tools/verifier | VQA service | unsupported→abstain |
| AVS grade/diversity | AVS service | mAP/nDCG/redundancy |
| Submission proxy | submission service | mock BTC server pass |

## Phase 4 — Competition UI

Tách ticket, không gom thành một mục:

- Query composer/progressive clues.
- Parsed entity editor.
- Result grid/list/timeline/event.
- Evidence/video/frame strip.
- KIS exact-frame tray.
- VQA evidence/answer/verifier.
- AVS grade/diversity/basket.
- Team board.
- Submission/history.
- Health drawer.
- Autosave/restore.
- Keyboard shortcuts.
- Playwright suite.

## Phase 5 — Ops và release

- Warmup preflight.
- CI eval gate.
- Prometheus/Grafana.
- Alerts.
- Backup/rollback.
- Immutable image tags.
- Release freeze.
- Full drills.

## Hạ tầng ngoài phase

- Test `docker-compose.production.yml` trên server thật trước khi viết lại.
- Cập nhật deployment guide theo compose mới.
- Tạo mock competition server để E2E và drill.
