# 07. Tiêu chí nghiệm thu và giới hạn

## P0 — bắt buộc

- Contract Video/Scene/Keyframe và ID validation pass.
- Export atomic, manifest/count/checksum verify pass.
- Data → Online integration test pass.
- KIS exact OCR trả đúng scene/best keyframe/timestamp.
- Sequence chỉ link cùng video và tăng theo thời gian.
- UI local gọi được backend remote, health và thumbnail hoạt động.
- Worker có auth, timeout/retry, concurrency bound.
- Qdrant collection/payload indexes/upsert idempotent.
- Có local portable index để phục hồi.

## P1 — chất lượng thi đấu

- Thay mock bằng caption/OCR/object/ASR/embedding thật và ghi provenance.
- Benchmark Recall@K/mAP/nDCG/latency trên validation set.
- Scene detector/keyframe selection theo shot/motion/OCR, không chỉ uniform.
- Cross-encoder/MLLM reranker và grounded VQA generator.

## Giới hạn công khai của bundle

Không thể khẳng định model quality hay độ ổn định của một instance Vast.ai chưa
được cấp quyền chạy thử. Bundle đã kiểm tra code path local, contract, export,
retrieval và packaging. GPU model weights, TLS/domain, data AIC thật và live
Qdrant phải được team cấu hình rồi chạy acceptance/load test trên hạ tầng thật.

Mock provider và hashing vector là baseline kiểm thử, không dùng để đánh giá
retrieval. VQA hiện evidence-first để tránh hallucination. FAISS native chỉ tạo
khi cài optional dependency; portable JSON luôn có.
