# 07. Submission và vận hành cuộc thi

## 1. Nguyên tắc

- Submission là module riêng, không phải nút export đơn giản.
- Token BTC chỉ ở backend.
- Mọi request/response phải lưu log.
- Dùng idempotency để tránh gửi trùng.
- Format chính thức phải cấu hình được vì rule có thể thay đổi.

## 2. Luồng KIS

```text
Candidate scene đúng
→ mở video
→ chọn frame chính xác
→ verify frame_idx/timestamp
→ add KIS tray
→ validate contract
→ confirm
→ send
→ record response
```

Validation:

- Video tồn tại.
- Frame nằm trong `[0, total_frames-1]`.
- Frame map đúng video.
- Không dùng timestamp khi rule yêu cầu frame.
- Không duplicate active answer.

## 3. Luồng VQA

- Answer không rỗng.
- Answer type hợp lệ.
- Evidence có thể mở được.
- Verifier status không phải `contradicted`.
- Nếu `insufficient_evidence`, UI phải yêu cầu operator xác nhận trước khi gửi.

## 4. Luồng AVS

- Không vượt số lượng tối đa.
- Không duplicate segment/event theo rule.
- Sort order đúng.
- Mỗi result có video/segment/frame theo contract.
- Basket cho phép bulk remove/reorder.

## 5. Submission Proxy

Chức năng:

- `validate()`.
- `format_official()`.
- `send()`.
- `retry()`.
- `get_status()`.
- `history()`.

Retry policy:

- Retry network timeout/5xx theo exponential backoff ngắn.
- Không retry validation error/4xx non-retryable.
- Không retry vô hạn.
- Idempotency key giữ nguyên qua retry.

## 6. Offline queue

Khi endpoint BTC tạm mất:

- Queue payload đã validate.
- Hiển thị `queued_offline`.
- Cho operator cancel trước khi retry.
- Retry tự động khi kết nối trở lại nếu rule cho phép.

## 7. Team coordination

Vai trò đề xuất:

- Search operator KIS.
- Search operator VQA/AVS.
- Reviewer/submission operator.
- Backend/GPU operator.
- Team lead/timekeeper.

Shared board:

- Query claimed by.
- Candidate pins.
- Review status.
- Approved result.
- Submitted by.
- Server response.

## 8. Progressive clue operations

- Mỗi clue có timestamp.
- Lưu query text tích lũy và query từng clue.
- Rank history.
- Pinned candidates không mất.
- Cho phép early submit với confirm hai bước.

## 9. Competition drill

### Drill 1 — Functional

- Mỗi task ít nhất 10 query.
- Search, evidence, exact frame, submit.
- Không fault injection.

### Drill 2 — Failure injection

- Tắt ES.
- Tắt VLM.
- Tăng latency Qdrant.
- Mất mạng submission.
- Refresh UI giữa query.

### Drill 3 — Full simulation

- Thời lượng như vòng thi.
- Chia role thật.
- Dùng release candidate freeze.
- Ghi metric người dùng và hệ thống.

## 10. Báo cáo sau drill

- Query success rate.
- Time-to-first-result.
- Time-to-correct.
- Time-to-submit.
- Number of refinements.
- Submission errors.
- Incidents.
- Root cause.
- Action items/P0/P1/P2.
