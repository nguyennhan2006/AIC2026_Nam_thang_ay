# 03. Danh mục toàn bộ chức năng hệ thống

Tài liệu này là catalog chức năng. Mỗi mục phải được đánh dấu trong `16_MASTER_CHECKLIST.md`.

## A. Data và Index Management

- Dataset browser theo batch/video.
- Trạng thái enrich từng video/scene.
- Quarantine và retry video lỗi.
- Incremental processing.
- Model/index provenance.
- Dense/sparse index registry.
- Alias publish/rollback.
- Index compatibility check.
- Reprocess một video/scene/module.
- Export JSONL/Parquet.
- Data integrity scan.

## B. Query Input và Understanding

- KIS/VQA/AVS manual task selector.
- Auto task routing có confidence và override.
- Text tiếng Việt/Anh/mixed.
- Progressive clues.
- Image upload/clipboard/result frame.
- Image + modification text.
- Query history và branch tree.
- Structured entity extraction.
- Must-match/nice-to-have/negative.
- OCR/ASR exact phrase marking.
- Temporal ordering.
- Raw query preservation.
- Cross-lingual variants.
- Query expansion và drift warning.
- Entity toggle/edit.
- Search profile selector.

## C. Retrieval

- Dense text-to-frame.
- Dense text-to-scene.
- Dense text-to-clip.
- Caption sparse search.
- OCR exact/partial/fuzzy/numeric.
- ASR exact/semantic/timestamp.
- Object/action/attribute search.
- Scene/environment/time/location filters.
- Image-to-image.
- Search from crop/region.
- Composed image-text retrieval.
- Event retrieval.
- Previous/next event.
- Related events.
- Conditional temporal search.
- Ordered A→B→C sequence search.

## D. Fusion và Reranking

- Score normalization.
- Candidate dedup.
- RRF.
- Weighted fusion.
- Query-adaptive profile.
- Exact-match boosts.
- Soft must-match coverage.
- Lightweight rerank.
- Deep VLM rerank.
- Temporal sequence scorer.
- Contradiction penalty.
- Hard-negative comparison.
- Score/evidence explanation.

## E. KIS Workspace

- Progressive clue mode.
- Rank history per clue.
- Pin/hide candidate.
- More-like-this/less-like-this.
- Search same object/person/scene/action.
- Video-grouped/timeline/event views.
- Neighbor frame/scene/event.
- Exact-frame selection.
- Current frame capture.
- Frame index validation.
- KIS submission tray.
- Early submit và confirm.

## F. VQA Workspace

- Question type router.
- Evidence modality planner.
- Evidence retrieval and table.
- OCR/ASR rule extraction.
- Count tool.
- Tracking tool.
- Temporal multi-evidence reasoning.
- VLM answer generation.
- Independent verifier.
- Supported/partial/contradicted/insufficient states.
- Human evidence replacement.
- Manual answer override.
- VQA submission validation.

## G. AVS Workspace

- Inclusion/exclusion editor.
- Relevance grade 0–3.
- Strict/balanced/broad threshold.
- MMR diversity.
- Visual/event clustering.
- Temporal dedup.
- Maximum per video.
- Minimum temporal distance.
- Coverage summary.
- Bulk select/remove/reorder.
- AVS result basket.
- Bulk export/submit.

## H. Evidence Viewer

- Video segment player.
- Frame-by-frame seek.
- Playback speed and loop.
- Frame strip.
- OCR/object overlays.
- ASR timestamp highlight.
- Caption/OCR/ASR/object/action/event tabs.
- Score waterfall.
- Previous/current/next event.
- Search from current frame/crop.
- Copy frame/timestamp/id.

## I. Interactive Feedback

- Relevant/partial/irrelevant.
- Wrong video/wrong moment/wrong action/wrong OCR.
- Duplicate label.
- Keyword add/remove/AND/OR/exclude.
- Positive/negative visual feedback.
- Session search tree.
- Feedback export.
- Replay with new index.

## J. Submission và Competition Operations

- Official contract config.
- Validate payload.
- Competition server test connection.
- Idempotency.
- Retry/offline queue.
- Submission log.
- Duplicate prevention.
- Team shared board.
- Query claim/reviewer/approval.
- Competition countdown.
- Autosave/restore.
- Practice/replay mode.
- Full drill reports.

## K. Reliability và Administration

- Health dashboard.
- Preflight warmup.
- Per-branch deadlines.
- Circuit breakers.
- Degraded response.
- Cache.
- Build/index mismatch alert.
- Prometheus/Grafana.
- Audit log.
- Secret management.
- Backup/restore.
- Read-only competition index mode.

## L. Priority

### P0 — thi được

Hybrid retrieval, evidence viewer, video/frame navigation, exact-frame selection, session, submission proxy, health/fallback.

### P1 — cạnh tranh tốt

Editable parser, progressive clue, event retrieval, image search, rerank cascade, VQA verifier, AVS diversity, team board.

### P2 — nghiên cứu tăng điểm

Composed retrieval, clip embedding, conditional temporal search, learning-to-rank, region indexing, advanced tracking.

### P3 — sau cuộc thi

Albums, chatbot tổng quát, mobile, public multi-tenant product.
