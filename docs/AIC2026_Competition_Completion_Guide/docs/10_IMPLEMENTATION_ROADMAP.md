# 10. Roadmap triển khai toàn hệ thống

## Phase 0 — Contracts và skeleton

Mục tiêu: tránh backend/UI/index phát triển lệch nhau.

- Chuẩn ID/schema/provenance.
- Search/branch/candidate/evidence/session/submission contracts.
- React competition skeleton.
- Video/frame endpoints.
- Session persistence.
- Feature flag framework.
- Release manifest.

Exit criteria:

- UI gọi mock API end-to-end.
- Exact-frame selection từ sample video hoạt động.
- Submission formatter mock pass.

## Phase 1 — Offline production enrich

- Ingest quarantine/incremental.
- TransNetV2 + uniform fallback.
- Keyframe quality/dedup/thumbnail.
- VAD + Whisper.
- Caption/OCR/object/action.
- Queue/resume.
- Metadata provenance.

Exit:

- Kill/restart không hỏng ledger.
- Sample human audit pass threshold.
- Không field mapping quan trọng bị null.

## Phase 2 — Index và baseline evaluation

- Qdrant named vectors + alias.
- Elasticsearch field indexes.
- Frame/scene/event/clip indexes.
- Ground truth expansion.
- Baseline dense/sparse/RRF.
- Dataset/index integrity scan.

Exit:

- Publish/rollback không downtime.
- Baseline metrics freeze.
- Search response có branch status/version.

## Phase 3 — Online competition backend

- Raw query + parser/variants.
- Per-branch deadlines.
- Fusion/dedup.
- BGE rerank.
- Event/neighbor/temporal endpoints.
- Evidence pack.
- VQA router/verifier.
- Submission proxy.
- Circuit breakers/cache.

Exit:

- Degraded mode tests pass.
- Submission mock server pass.
- p95 within profile budget.

## Phase 4 — Complete competition UI

- Query editor/progressive clues.
- Result views.
- Video/evidence inspector.
- KIS tray.
- VQA answer/evidence card.
- AVS grading/diversity/basket.
- Team board.
- Keyboard shortcuts.
- Autosave/restore.

Exit:

- Playwright E2E pass.
- Operator drill không có P0.

## Phase 5 — Research upgrades

- SigLIP2/OpenCLIP ensemble.
- Qwen3-VL caption/deep rerank.
- Clip embeddings.
- Conditional temporal search.
- Composed image-text retrieval.
- Learning-to-rank.

Chỉ merge competition profile khi thắng ablation.

## Dependency highlights

- VQA answer quality phụ thuộc evidence retrieval trước.
- AVS diversity phụ thuộc event/segment dedup trước.
- Exact-frame submission phụ thuộc frame service/ID contract trước.
- UI competition không nên đợi Phase 4 để bắt đầu; skeleton ở Phase 0.

## Definition of done cho ticket

- Code.
- Tests.
- Docs/config.
- Metrics/logs.
- Feature flag.
- Acceptance evidence.
- Rollback.
- Owner/reviewer.
