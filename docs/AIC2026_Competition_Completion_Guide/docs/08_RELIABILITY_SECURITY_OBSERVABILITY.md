# 08. Reliability, security và observability

## 1. No silent degradation

Mọi nhánh phải có status. Response tổng có thể `completed_with_warning`, nhưng không được giả vờ full-quality.

## 2. Deadlines đề xuất

Mỗi profile có ngân sách riêng, ví dụ:

```yaml
fast:
  total_ms: 1500
  dense_ms: 500
  sparse_ms: 500
  rerank_ms: 400
balanced:
  total_ms: 4000
quality:
  total_ms: 12000
```

Số cuối phải benchmark thật.

## 3. Circuit breakers

Bao quanh:

- vLLM/Qwen.
- Elasticsearch.
- Qdrant/Milvus.
- Competition server.
- Object storage/video stream.

State: closed/open/half-open; có metric và UI status.

## 4. Fallback matrix

| Thành phần lỗi | Fallback |
|---|---|
| LLM parser | Rule parser + raw query |
| ES | Local BM25/read-only backup |
| Qdrant | Secondary FAISS/read replica nếu có |
| VLM rerank | BGE + fusion result |
| OCR online | Dùng OCR offline cached |
| ASR branch | Trả degraded, không bịa transcript |
| Video stream | Thumbnail/evidence metadata |
| Submission server | Validated offline queue |

## 5. Cache

- Query embeddings.
- Parsed query.
- Branch results.
- Candidate evidence.
- Thumbnails.
- Video segment ranges.
- VQA result với version key.

Cache key phải chứa release/index/model version.

## 6. Preflight

- Check GPU/CUDA.
- Load model.
- Dummy inference.
- Verify index alias.
- Verify index dimension.
- Verify manifest checksum.
- Open sample video/frame.
- Run smoke queries.
- Test submission endpoint.

Backend chỉ ready khi gate pass hoặc explicit degraded profile được operator chấp nhận.

## 7. Observability

Metrics:

- Requests/task/profile.
- Latency per stage/branch.
- Branch success/timeout/error.
- Candidate counts.
- Cache hit.
- GPU memory/utilization.
- Queue depth.
- Submission success/failure.
- Version mismatch.

Logs:

- Structured JSON.
- Request ID/session ID/query ID.
- Không log secret.
- Không lưu chain-of-thought.

Traces:

```text
parse → branch retrieval → fusion → rerank → evidence → answer/submission
```

## 8. Alerts

P0:

- Active index missing.
- Query/index dimension mismatch.
- Submission endpoint unavailable gần thời điểm thi.
- GPU worker all down.
- Data corruption/checksum fail.

P1:

- p95 tăng vượt budget.
- Branch timeout rate cao.
- Cache hit giảm mạnh.
- Build ID mismatch.

## 9. Security

- Secrets qua environment/secret store.
- Competition token chỉ backend.
- TLS.
- Role-based access.
- Input size/type validation.
- Rate limit.
- Audit log.
- Read-only indexes khi thi.
- Không expose internal stack trace ra UI.

## 10. Backup và rollback

- PostgreSQL snapshot.
- Index alias rollback.
- Release manifest lịch sử.
- Frontend/backend image tags immutable.
- Session/submission log durable.
