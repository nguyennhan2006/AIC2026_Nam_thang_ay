# 12. Runbook cho đội thi

## 1. Vai trò

### Technical lead

- Freeze release.
- Quyết định rollback.
- Theo dõi health/incident.

### Backend/GPU operator

- Preflight.
- Worker/index health.
- Restart/fallback.

### KIS operator

- Progressive clues.
- Candidate refinement.
- Exact-frame selection.

### VQA/AVS operator

- Evidence/answer review.
- Relevance grading/bulk selection.

### Submission reviewer

- Validate.
- Confirm payload.
- Theo dõi server response.

Có thể gộp role nếu đội ít người, nhưng trách nhiệm phải rõ.

## 2. Trước cuộc thi

- Freeze release manifest.
- Run preflight.
- Test sample search mỗi task.
- Test exact frame.
- Test submission endpoint.
- Check clocks/timezone.
- Check token expiry.
- Check backup internet/power.
- Open health dashboard.
- Confirm operator assignments.

## 3. Khi có query

1. Claim query.
2. Chọn đúng task/profile.
3. Search raw query.
4. Chỉ bật parser/expansion cần thiết.
5. Pin candidates.
6. Verify evidence/video.
7. Select exact output.
8. Reviewer validate.
9. Submit và confirm response.

## 4. Incident playbooks

### ES down

- UI báo degraded.
- Chuyển sparse fallback profile.
- Không restart toàn backend.

### Qdrant down

- Dùng replica/FAISS fallback.
- Giảm search profile nếu cần.

### VLM OOM

- Open circuit.
- Bỏ deep rerank.
- Dùng BGE/fusion.

### UI mất kết nối

- Reconnect.
- Restore session.
- Dùng cached results.

### Submission timeout

- Kiểm tra idempotency/history.
- Không gửi lại mù.
- Queue/retry theo policy.

## 5. Sau mỗi query

- Ghi result/submission status.
- Gắn lỗi đáng chú ý.
- Không chỉnh production config tùy tiện giữa vòng nếu chưa có lead approval.

## 6. Sau cuộc thi/drill

- Export sessions.
- Export feedback.
- Incident timeline.
- Metric report.
- Root-cause và action items.
