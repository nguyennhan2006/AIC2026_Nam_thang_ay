# 06. API và data contracts

## 1. Quy ước ID

Khuyến nghị:

```text
video_id   = L{batch:02d}_V{video:03d}
scene_id   = {video_id}_S{scene:05d}
frame_id   = {video_id}_F{frame_idx:09d}
segment_id = {video_id}_G{segment:05d}
event_id   = {video_id}_E{event:05d}
```

Mọi contract phải có `schema_version`.

## 2. Search endpoints

```text
POST /v1/search/kis
POST /v1/search/avs
POST /v1/vqa
POST /v1/search/image
POST /v1/search/composed
POST /v1/query/parse
POST /v1/query/expand
POST /v1/rerank
```

## 3. Evidence/video endpoints

```text
GET /v1/candidates/{candidate_id}
GET /v1/evidence/{candidate_id}
GET /v1/videos/{video_id}/stream
GET /v1/videos/{video_id}/frames/{frame_idx}
GET /v1/videos/{video_id}/neighbors/{frame_idx}
GET /v1/scenes/{scene_id}/neighbors
GET /v1/events/{event_id}/neighbors
```

## 4. Session/feedback/submission

```text
POST /v1/sessions
GET  /v1/sessions/{session_id}
PATCH /v1/sessions/{session_id}
POST /v1/feedback
POST /v1/submissions/validate
POST /v1/submissions/send
GET  /v1/submissions/history
```

## 5. Search request

```json
{
  "schema_version": "SearchRequestV2",
  "task": "KIS",
  "query": "Một người áo vàng đang bơi.",
  "session_id": "S_001",
  "clue_id": "C_003",
  "top_k": 100,
  "profile": "competition_balanced",
  "filters": {
    "video_ids": [],
    "time_range": null,
    "collections": []
  },
  "options": {
    "use_visual": true,
    "use_caption": true,
    "use_ocr": false,
    "use_asr": false,
    "use_temporal": false,
    "deep_verify_top_k": 10
  }
}
```

## 6. Branch status

```json
{
  "branch": "asr_sparse",
  "status": "timeout",
  "duration_ms": 4000,
  "candidate_count": 0,
  "error_code": "BRANCH_DEADLINE_EXCEEDED",
  "fallback_used": false
}
```

Status enum:

- `ok`
- `disabled`
- `empty`
- `timeout`
- `error`
- `circuit_open`
- `fallback`

## 7. Candidate response

```json
{
  "candidate_id": "L01_V003_S00012",
  "video_id": "L01_V003",
  "scene_id": "L01_V003_S00012",
  "event_id": "L01_V003_E00007",
  "start_sec": 120.2,
  "end_sec": 128.6,
  "representative_frame_idx": 3610,
  "supporting_frame_indices": [3590, 3610, 3630],
  "score": 0.87,
  "scores": {
    "visual": 0.71,
    "caption": 0.82,
    "ocr": 0.93,
    "asr": 0.18,
    "fusion": 0.64,
    "rerank": 0.88
  },
  "matched_sources": ["visual", "caption", "ocr"],
  "contradictions": [],
  "versions": {
    "dataset": "btc_v2",
    "dense_index": "siglip2_v4",
    "sparse_index": "es_v7"
  }
}
```

## 8. Search response

```json
{
  "request_id": "R_0018",
  "session_id": "S_001",
  "status": "completed_with_warning",
  "query_plan": {},
  "branches": [],
  "results": [],
  "timing_ms": {
    "parse": 80,
    "retrieve": 420,
    "fusion": 20,
    "rerank": 850,
    "total": 1370
  },
  "warnings": ["ASR branch timed out"],
  "release_id": "aic2026_rc3"
}
```

## 9. Evidence pack

```json
{
  "candidate_id": "...",
  "keyframes": [
    {
      "frame_idx": 3610,
      "caption": "...",
      "ocr": ["..."],
      "objects": ["..."],
      "actions": ["..."]
    }
  ],
  "asr_window": {
    "start_sec": 118.0,
    "end_sec": 130.0,
    "text": "..."
  },
  "previous_event": {},
  "next_event": {},
  "retrieval_scores": {}
}
```

## 10. Submission contract nội bộ

```json
{
  "submission_id": "SUB_001",
  "query_id": "Q001",
  "task": "KIS",
  "payload_internal": {},
  "payload_official": {},
  "validation_status": "valid",
  "idempotency_key": "...",
  "server_status": "accepted"
}
```

Official formatter phải tách khỏi internal schema để thích ứng khi BTC thay format.

## 11. Error envelope

```json
{
  "error": {
    "code": "INDEX_VERSION_MISMATCH",
    "message": "Query encoder and active index are incompatible.",
    "retryable": false,
    "request_id": "R_0018"
  }
}
```
