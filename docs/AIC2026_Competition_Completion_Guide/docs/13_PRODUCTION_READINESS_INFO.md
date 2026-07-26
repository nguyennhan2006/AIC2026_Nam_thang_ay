# 13. Thông tin cần chuẩn bị trước khi lên production và thi đấu

Tài liệu này mở rộng checklist production thành checklist **server + client + competition operations**.

## 1. Checkpoint và revision cần pin

| Model | Vai trò | Trạng thái hiện tại | Việc cần làm | Blocker |
|---|---|---|---|---|
| Qwen3-VL-32B-Instruct | caption/VQA/deep rerank | chưa benchmark production | pin revision, VRAM/latency/human audit | trước khi bật deep path |
| Qwen3-14B-Instruct AWQ | query parser | module chưa hoàn thiện | chọn source, JSON-mode, fallback | Phase 3 |
| SigLIP2 | dense chính | chưa thay CLIP | chọn checkpoint/dim/named vector | trước index mới |
| OpenCLIP ViT-L/14 | ensemble/đối chứng | chưa pin tag | pin pretrained tag | không blocker |
| TransNetV2 | scene | uniform fallback hiện có | verify source/license/tune | trước scene migration |
| PaddleOCR vi+en | exact OCR | Qwen OCR semantic hiện có | process isolation/CUDA test | trước OCR index |
| Whisper large-v3 + alignment | ASR | turbo chunk-level | test vi alignment | nếu cần word timestamp |
| BGE reranker-v2-m3 | rerank tầng 1 | endpoint chưa hoàn chỉnh | benchmark top-k/latency | trước cascade |
| Elasticsearch analyzer vi | sparse | BM25 local | verify ICU/analyzer/image | trước corpus lớn |

Mọi checkpoint phải xuất hiện trong release manifest và response provenance.

## 2. License/token gate

- Hugging Face token và gated model acceptance.
- pyannote optional.
- Kiểm tra license sử dụng trong cuộc thi.
- Mirror/download test trước ngày thi.
- Không phát hiện thiếu token khi job đã chạy.

## 3. Hạ tầng

Cần chốt bằng benchmark, không chỉ ước tính:

- GPU topology.
- RAM.
- NVMe cho corpus/index/cache.
- Network local↔server.
- HTTP range video throughput.
- Backup instance/replica.
- Chi phí theo giờ và thời gian warmup.

## 4. Ground truth

Không dùng `SEQUENCE` như task thứ tư. Mục tiêu:

- ≥50 KIS, có nhóm temporal/sequence.
- ≥50 VQA, có OCR/ASR/count/tracking/temporal.
- ≥50 AVS, có multi-positive/diversity.

Mỗi item có interval, hard negative và ambiguity flag khi cần.

## 5. Environment variables

Nhóm:

- Feature flags retrieval.
- ES/Qdrant/Postgres/Redis.
- LLM/VLM endpoints.
- Branch deadlines.
- Cache/circuit breaker.
- Video/object storage.
- Submission server/token.
- Monitoring/alerts.
- Release/build IDs.

Thiếu biến bắt buộc phải fail-fast.

## 6. Competition rule và submission contract

Trạng thái phải được điền khi BTC công bố:

- Official tasks.
- Input/output schema.
- Max results.
- Resubmission rule.
- Early submission rule.
- Rate limit.
- Authentication.
- External API policy.
- Scoring/time policy.

Mọi phần chưa biết ghi `pending_official_rule`, không tự suy đoán trong code.

## 7. Dataset/index freeze

Freeze:

- dataset/scene/keyframe/caption/OCR/ASR versions;
- model revisions;
- dense/sparse index build IDs;
- schema/backend/frontend build IDs.

Dùng `templates/competition_release_manifest.yaml`.

## 8. Client readiness

- React app build immutable.
- Session autosave/restore.
- Exact-frame selection.
- Progressive clue.
- KIS/VQA/AVS baskets.
- Keyboard shortcuts.
- Health/degraded status.
- Submission history.

## 9. Submission readiness

- Test endpoint/token.
- Validate/format/send/retry.
- Idempotency.
- Offline queue.
- Log server response.
- Duplicate prevention.

## 10. Recovery readiness

- ES fallback.
- Dense fallback.
- VLM skip.
- Cached results.
- Submission queue.
- Index alias rollback.
- UI session recovery.

## 11. Team readiness

- Role assignment.
- Shared board.
- Query claim.
- Reviewer.
- Incident escalation.
- Timekeeper.

## 12. Drill gate

Trước competition release phải có:

- Functional drill.
- Failure-injection drill.
- Full simulation.

Không có P0 unresolved sau drill cuối.
