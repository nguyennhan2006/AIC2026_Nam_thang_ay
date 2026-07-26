# 16. Master checklist hoàn thiện bộ thi đấu

Dùng bảng này làm nguồn theo dõi chính. Không đánh dấu `COMPETITION_READY` nếu chưa có evidence nghiệm thu.

## A. Contracts

- [ ] ID regex và mapping.
- [ ] Search schemas.
- [ ] Branch status.
- [ ] Candidate/evidence.
- [ ] Session/feedback.
- [ ] Submission internal/official.
- [ ] Release manifest.

## B. Offline

- [ ] Quarantine.
- [ ] Incremental/resume.
- [ ] Scene detection + fallback.
- [ ] Keyframe quality.
- [ ] ASR/VAD.
- [ ] Caption.
- [ ] OCR.
- [ ] Object/action/scene.
- [ ] Event segmentation.
- [ ] Provenance/checksum.

## C. Index

- [ ] Frame dense.
- [ ] Scene dense.
- [ ] Caption sparse.
- [ ] OCR sparse/exact.
- [ ] ASR sparse.
- [ ] Event index.
- [ ] Alias rollback.
- [ ] Integrity scan.

## D. Online

- [ ] Q0 raw query.
- [ ] Parser/fallback.
- [ ] Query variants/drift.
- [ ] Per-branch deadlines.
- [ ] Fusion/dedup.
- [ ] Rerank.
- [ ] Event/temporal.
- [ ] Evidence pack.
- [ ] VQA verifier.
- [ ] AVS diversity.
- [ ] Sessions/feedback.

## E. UI

- [ ] Competition mode.
- [ ] Query/progressive clues.
- [ ] Parsed chips.
- [ ] Result views.
- [ ] Video/frame strip.
- [ ] Neighbor events.
- [ ] Exact frame.
- [ ] KIS tray.
- [ ] VQA card.
- [ ] AVS basket.
- [ ] Health/degraded status.
- [ ] Autosave/restore.
- [ ] Keyboard shortcuts.

## F. Submission

- [ ] Official rule recorded.
- [ ] Mock server.
- [ ] Validate/format/send.
- [ ] Idempotency.
- [ ] Retry/offline queue.
- [ ] History/audit.
- [ ] Duplicate prevention.

## G. Evaluation

- [ ] Stratified GT.
- [ ] KIS metrics.
- [ ] VQA answer+evidence.
- [ ] AVS relevance+diversity.
- [ ] UI time-to-correct/time-to-submit.
- [ ] CI regression gate.

## H. Reliability

- [ ] Preflight.
- [ ] Circuit breakers.
- [ ] Cache.
- [ ] Fallback matrix tested.
- [ ] Monitoring/alerts.
- [ ] Backup/rollback.
- [ ] No secrets in frontend.

## I. Operations

- [ ] Roles.
- [ ] Team board.
- [ ] Functional drill.
- [ ] Failure drill.
- [ ] Full simulation.
- [ ] Release freeze.
- [ ] Go/no-go approval.
