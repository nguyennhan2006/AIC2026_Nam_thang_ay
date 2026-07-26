# 01. Phạm vi hệ thống và tiêu chí thành công

## 1. Mục tiêu

Xây dựng một hệ thống tìm kiếm và hỏi đáp video đa phương thức phục vụ ba nhiệm vụ:

- **KIS — Known-Item Search:** tìm đúng moment/segment/frame cụ thể.
- **VQA — Video Question Answering:** tìm evidence và trả lời có căn cứ.
- **AVS — Ad-hoc Video Search:** tìm nhiều segment liên quan và xếp hạng tốt.

Hệ thống cuối không chỉ là API search. Nó phải là một nền tảng thi đấu gồm:

```text
Offline enrichment/indexing
→ Online retrieval/reasoning
→ Human verification
→ Exact result selection
→ Submission
→ Audit/replay
```

## 2. Input và output theo task

### 2.1. KIS

Input: mô tả tự nhiên, có thể được cung cấp dần.

Output nội bộ tối thiểu:

```json
{
  "task": "KIS",
  "query_id": "Q001",
  "results": [
    {
      "video_id": "L01_V003",
      "segment_id": "L01_V003_S0012",
      "frame_idx": 3610,
      "start_sec": 120.2,
      "end_sec": 128.6,
      "score": 0.87,
      "evidence": ["visual", "caption", "ocr"]
    }
  ]
}
```

### 2.2. VQA

Input: câu hỏi tự nhiên, kho video hoặc phạm vi video.

Output:

```json
{
  "task": "VQA",
  "answer": "6",
  "answer_type": "count",
  "status": "supported",
  "confidence": 0.84,
  "evidence": [
    {"video_id": "L02_V014", "frame_idx": 8420, "reason": "Bánh có 6 cây nến."}
  ]
}
```

### 2.3. AVS

Input: mô tả tổng quát và tiêu chí bao gồm/loại trừ.

Output: danh sách nhiều segment, có relevance grade, cluster và dedup information.

## 3. Năng lực lõi

1. **Retrieval:** tìm đúng video/scene/frame/event.
2. **Grounded understanding:** caption, OCR, ASR, object/action, scene, event.
3. **Temporal reasoning:** trước/sau, chuỗi A→B→C, khoảng thời gian, neighbor event.
4. **Human-in-the-loop:** sửa parser, pin/hide, refine, exact-frame selection.
5. **Submission operations:** validate, send, retry, log, tránh trùng.
6. **Reliability:** degraded mode, fallback, restore session và audit.

## 4. Ngoài phạm vi ưu tiên trước cuộc thi

- Chatbot tổng quát trên toàn archive.
- Album cá nhân.
- Mobile-first UI.
- Full graph database nếu relational/event table đã đủ.
- Fine-tune model lớn khi chưa có ablation chứng minh lợi ích.
- Tự động gọi mọi model nặng cho mọi query.

## 5. Metric thành công

### KIS

- Recall@1/5/20/50/100.
- MRR.
- Video-Recall@K.
- Hit-in-interval.
- Time-to-first-correct-result.
- Time-to-submit.

### VQA

- Exact Match/F1.
- Numeric/count accuracy.
- Evidence hit@K.
- Grounded answer rate.
- Hallucination rate.
- Correct abstention rate.

### AVS

- mAP.
- nDCG@K.
- Precision/Recall@K.
- Unique relevant events.
- Redundancy rate.
- Diversity-aware relevance.

### System

- p50/p95/p99 latency theo branch.
- Error/degraded rate.
- Search session restore rate.
- Submission success rate.
- Index/model mismatch rate = 0.

## 6. Tiêu chí release

Một feature chỉ được bật trong competition profile khi:

1. Có feature flag và default rõ ràng.
2. Có unit/integration/E2E test.
3. Có ablation trên dev set đủ lớn.
4. Không làm p95 vượt ngân sách đã chốt.
5. Có log/provenance.
6. Có rollback path.
