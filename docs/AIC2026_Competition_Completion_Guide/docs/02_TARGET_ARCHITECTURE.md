# 02. Kiến trúc mục tiêu của hệ thống thi đấu

## 1. Kiến trúc tổng thể

```text
RAW VIDEOS
  │
  ▼
OFFLINE PIPELINE
  ├── ingest + quarantine + ledger
  ├── audio extraction + VAD + ASR
  ├── shot/scene/event segmentation
  ├── keyframe/clip sampling
  ├── visual/clip embeddings
  ├── caption + region/crop metadata
  ├── OCR
  ├── object/action/scene metadata
  ├── normalized metadata + provenance
  └── index publication
          │
          ├── Qdrant/Milvus/FAISS dense indexes
          ├── Elasticsearch/OpenSearch sparse indexes
          ├── PostgreSQL metadata/session/submission
          ├── Redis cache/queue/circuit state
          └── Object storage/video/frame/thumbnail

LOCAL COMPETITION CLIENT
  │ HTTPS + SSE/WebSocket
  ▼
REMOTE FASTAPI BACKEND
  ├── Query Service
  ├── Retrieval Orchestrator
  ├── Dense Search
  ├── Sparse/OCR/ASR Search
  ├── Fusion/Rerank
  ├── Temporal/Event Service
  ├── Evidence/VQA Service
  ├── Video/Frame Service
  ├── Session/Feedback Service
  ├── Submission Proxy
  └── Health/Observability
```

## 2. Ranh giới offline và online

### Offline

- Chạy model nặng trên toàn corpus.
- Tạo scene, keyframe, clip, caption, OCR, ASR, embeddings.
- Chuẩn hóa metadata.
- Build/version/publish index.
- Human audit và eval dataset.

### Online

- Parse query và giữ raw query.
- Sinh query variants có kiểm soát.
- Search nhiều branch song song.
- Fusion/rerank top candidates.
- Chạy deep VLM chỉ trên tập nhỏ.
- Xây evidence pack.
- Video navigation và submission.

Không chạy caption/OCR toàn corpus trong đường online. Online chỉ được chạy bổ sung trên top candidate khi cần.

## 3. Các lớp dịch vụ

### 3.1. Query Service

- Task routing.
- Structured parsing.
- Raw/normalized/bilingual variants.
- Must-match/nice-to-have.
- Temporal constraints.
- Drift warnings.

### 3.2. Retrieval Orchestrator

- Per-branch top-k.
- Deadlines.
- Branch status.
- Candidate normalization.
- Dedup.
- Fusion.

### 3.3. Retrieval adapters

- Dense frame.
- Dense scene.
- Dense clip.
- Caption sparse.
- OCR exact/fuzzy.
- ASR exact/semantic.
- Object/action.
- Event search.
- Image-to-image.
- Region/crop.

### 3.4. Rerank Service

- Lightweight rerank top 100–300.
- Deep multimodal rerank top 5–30.
- Must-match coverage.
- Contradiction detection.
- Temporal scorer.

### 3.5. Evidence/VQA

- Evidence selection.
- Evidence compression.
- Answer router.
- Rule extraction.
- Count/tracking tools.
- Answer generation.
- Independent verifier.
- Abstention.

### 3.6. Submission Proxy

- Validate official format.
- Attach idempotency key.
- Send/retry.
- Store request/response.
- Prevent accidental duplicate.

## 4. Data stores

| Store | Vai trò | Yêu cầu |
|---|---|---|
| PostgreSQL | Metadata, sessions, feedback, submissions | Transactional, backup |
| Qdrant/Milvus | Dense vectors, named vectors | Alias publish/rollback |
| Elasticsearch | Caption/OCR/ASR/object sparse | Field weighting, vi analyzer |
| Redis | Queue, cache, circuit state, session hot state | TTL và persistence policy |
| Object storage/NVMe | Videos, frames, thumbnails, manifests | HTTP range, checksum |

## 5. Deployment profile

### Local client

- React/Vite/TypeScript.
- Không chứa competition secret.
- Có local cache/session draft.
- Có reconnect và cached results.

### Remote backend

- FastAPI.
- GPU worker vLLM/inference endpoints.
- Read-only indexes trong giờ thi.
- Submission proxy giữ token.
- SSE/WebSocket cho progress.

## 6. Tính bất biến bắt buộc

1. Không có vector/document mồ côi.
2. Không query encoder lệch index encoder.
3. Không publish index thiếu manifest.
4. Không nhận traffic trước warmup.
5. Không trả `ok` nếu branch quan trọng đã fail mà không báo.
6. Không submit nếu payload chưa validate.
