# AIC 2026 — Master Synchronization Guide

Báo cáo quét codebase cho thấy nhận định **khoảng 30% đồng bộ** là hợp lý: lõi `retrieval → fusion → rules` đã có nền tốt, nhưng contract hiện vẫn xoay quanh `scene`, trong khi đầu ra thi đấu cần `frame_idx`; đồng thời evidence pack, rerank cascade, submission, session, Q&A hoàn chỉnh, TRAKE hai giai đoạn và AVS relevance/diversity gần như chưa tồn tại. 

Điểm quan trọng nhất là không nên tiếp tục bổ sung từng tính năng rời rạc. Cần khóa một **kiến trúc đích duy nhất**, sau đó di chuyển code hiện tại theo thứ tự phụ thuộc.

---

# 1. Nguồn sự thật thống nhất

Tạo ngay:

```text
docs/00_SOURCE_OF_TRUTH.md
```

Thứ tự ưu tiên khi các tài liệu mâu thuẫn:

1. Luật và format chính thức của BTC.
2. Gold benchmark và schema đã version hóa.
3. Code, config, migration và test hiện tại.
4. Notebook đã chạy có output manifest và metric.
5. Tài liệu nghiên cứu, paper và roadmap.
6. Tài liệu cũ hoặc kiến trúc suy luận từ chat.

Audit trước cũng xác định chính xác thứ tự này và yêu cầu đánh dấu nội dung cũ là `superseded`, thay vì âm thầm sửa để mất lịch sử. 

## 1.1. Taxonomy chuẩn

Dùng một enum chính thức:

```python
class TaskType(StrEnum):
    TEXTUAL_KIS = "TEXTUAL_KIS"
    QA = "QA"
    TRAKE = "TRAKE"
    AVS = "AVS"  # task nội bộ mở rộng
```

Alias chỉ tồn tại tại API boundary:

```text
KIS        → TEXTUAL_KIS
VQA        → QA
Q&A        → QA
SEQUENCE   → TRAKE
```

Ba task hướng thi đấu là `TEXTUAL_KIS`, `QA`, `TRAKE`; `AVS` được giữ làm task nội bộ để đánh giá truy xuất rộng. 

Không tiếp tục sử dụng lẫn lộn:

```text
kis / textual_kis
vqa / qa
sequence / temporal
```

trong domain core.

---

# 2. Kiến trúc đích duy nhất

```text
OFFLINE
Video
→ scene
→ keyframe/frame manifest
→ color/OCR/object/caption
→ ASR
→ frame embeddings
→ clip pooling
→ action tags
→ event grouping
→ event aggregation
→ index build
→ versioned artifacts

ONLINE
SearchRequest
→ Query Understanding
→ Task-aware Planner
→ Retriever Registry
→ Concurrent Retrieval
→ Score Normalization
→ Threshold Processing
→ Fusion
→ Aggregation/Dedup
→ Rerank Cascade
→ Evidence Hydration
→ Task Processor
→ Submission-aware Ranking
→ Validation/Export

TASK PROCESSORS
TEXTUAL_KIS → safe-frame result
QA          → frame-answer joint result
TRAKE       → video-first ordered sequence
AVS         → relevant diverse segments
```

Online không được trực tiếp phụ thuộc vào chi tiết Qdrant, Elasticsearch, model hoặc file path. Mỗi tầng phải đi qua port/adapter.

---

# 3. Quyết định nền tảng: retrieval entity và submission entity

Hiện online đang dùng `scene` làm đơn vị chính và đã làm mất `frame_idx`. Đây là blocker lớn nhất vì kết quả hiện tại không thể chuyển thẳng thành submission. 

Không cần loại bỏ scene retrieval. Cần phân biệt:

```text
Retrieval entity:
frame | scene | clip | event | video

Submission entity:
frame_idx
frame-answer tuple
ordered frame list
```

Scene vẫn có thể là candidate retrieval, nhưng mọi scene bắt buộc phải có danh sách frame evidence đầy đủ.

## 3.1. Thay `keyframe_ids`, `paths`, `timestamps` song song

Không tiếp tục lưu ba list tách rời:

```python
keyframe_ids: list[str]
keyframe_paths: list[str]
keyframe_timestamps: list[float]
```

Thay bằng:

```python
class FrameEvidence(StrictModel):
    keyframe_id: str
    video_id: str
    scene_id: str

    frame_idx: int
    timestamp_sec: float
    image_path: str

    quality_score: float | None = None
    blur_score: float | None = None
    boundary_distance_frames: int | None = None

    caption: str | None = None
    ocr_texts: list[str] = Field(default_factory=list)
    object_labels: list[str] = Field(default_factory=list)
    action_labels: list[str] = Field(default_factory=list)
    dominant_colors: list[str] = Field(default_factory=list)

    embedding_refs: list[EmbeddingReference] = Field(
        default_factory=list
    )
```

```python
class SceneDocument(StrictModel):
    scene_id: str
    video_id: str

    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float

    keyframes: list[FrameEvidence]

    caption: str
    ocr_text: str
    asr_text: str
    keywords: list[str]
    objects: list[str]
    actions: list[str]

    event_id: str | None = None
    artifact_version: str
```

## 3.2. Candidate contract

```python
class Candidate(StrictModel):
    candidate_id: str
    entity_type: Literal[
        "frame", "scene", "clip", "event", "video"
    ]

    video_id: str
    scene_id: str | None = None
    clip_id: str | None = None
    event_id: str | None = None

    frame_idx: int | None = None
    timestamp_sec: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None

    source: str
    modality: Modality

    raw_score: float
    normalized_score: float | None = None
    percentile_score: float | None = None
    rank: int

    model_id: str | None = None
    index_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    payload: dict[str, JsonValue] = Field(default_factory=dict)
```

## 3.3. SearchHit contract

```python
class SearchHit(StrictModel):
    rank: int

    candidate_id: str
    video_id: str
    scene_id: str | None
    event_id: str | None

    best_frame_idx: int
    best_keyframe_id: str | None
    best_timestamp_sec: float

    start_frame: int | None
    end_frame: int | None

    final_score: float
    safe_frame_score: float | None

    matched_branches: list[str]
    branch_scores: dict[str, BranchScore]
    branch_contributions: dict[str, float]

    evidence_summary: str | None
    warnings: list[str]
```

`best_frame_idx` phải là bắt buộc cho KIS/QA output.

---

# 4. Source tree mục tiêu

```text
online/
├── domain/
│   ├── tasks.py
│   ├── query.py
│   ├── search_config.py
│   ├── candidate.py
│   ├── scores.py
│   ├── evidence.py
│   ├── execution.py
│   ├── task_results.py
│   └── submission.py
│
├── ports/
│   ├── retriever.py
│   ├── reranker.py
│   ├── query_planner.py
│   ├── evidence_store.py
│   ├── media_store.py
│   └── submission_gateway.py
│
├── adapters/
│   ├── retrieval/
│   │   ├── dense_frame.py
│   │   ├── dense_scene.py
│   │   ├── dense_clip.py
│   │   ├── lexical.py
│   │   ├── ocr.py
│   │   ├── asr.py
│   │   ├── object_action.py
│   │   ├── color.py
│   │   ├── event.py
│   │   ├── image_similarity.py
│   │   └── composed.py
│   ├── rerank/
│   │   ├── rules.py
│   │   ├── bge.py
│   │   ├── qwen_vl.py
│   │   └── temporal.py
│   └── stores/
│       ├── qdrant.py
│       ├── elasticsearch.py
│       ├── metadata.py
│       ├── artifact_store.py
│       └── session_store.py
│
├── services/
│   ├── query_understanding.py
│   ├── retrieval_planner.py
│   ├── retrieval_orchestrator.py
│   ├── score_normalization.py
│   ├── thresholding.py
│   ├── fusion.py
│   ├── aggregation.py
│   ├── deduplication.py
│   ├── rerank_pipeline.py
│   ├── evidence_builder.py
│   ├── safe_frame.py
│   ├── kis.py
│   ├── qa/
│   │   ├── parser.py
│   │   ├── evidence_selector.py
│   │   ├── tools.py
│   │   ├── answer_generator.py
│   │   ├── normalizer.py
│   │   └── verifier.py
│   ├── trake/
│   │   ├── parser.py
│   │   ├── video_retriever.py
│   │   ├── event_retriever.py
│   │   ├── sequence_search.py
│   │   ├── frame_refinement.py
│   │   └── scorer.py
│   └── avs/
│       ├── criteria.py
│       ├── grader.py
│       ├── clustering.py
│       └── diversity.py
│
├── competition/
│   ├── rules.py
│   ├── scorer.py
│   ├── ranking_planner.py
│   ├── submission_builder.py
│   └── submission_validator.py
│
└── api/
    ├── routes/
    ├── dependencies.py
    ├── container.py
    └── error_handlers.py
```

Không cần di chuyển toàn bộ file ngay lập tức. Có thể giữ adapter hiện tại và migrate dần vào cấu trúc này.

---

# 5. Workstream W0 — khóa contract và migration

Đây là bước đầu tiên, trước rerank, VQA hoặc TRAKE.

## 5.1. Task contract

Sửa:

```text
online/domain/models.py
online/domain/search_config.py
online/api/routes.py
```

Yêu cầu:

* Canonical task enum.
* Alias normalization.
* Path task và body task mâu thuẫn → `409` hoặc `422`.
* Không ghi đè im lặng.

Ví dụ:

```python
if request.task is not None and request.task != path_task:
    raise TaskConflictError(
        body_task=request.task,
        path_task=path_task,
    )
```

## 5.2. Frame contract

Đưa `frame_idx` xuyên suốt:

```text
datasection
→ SceneDocument
→ metadata adapter
→ Candidate
→ SearchHit
→ EvidencePack
→ task result
→ submission
```

## 5.3. Migration script

Tạo:

```text
scripts/migrate_scene_documents_v2.py
```

Nhiệm vụ:

* Đọc scene document cũ.
* Join keyframe metadata canonical.
* Tạo `FrameEvidence`.
* Validate `frame_idx`.
* Xuất scene document V2.
* Báo record không map được.
* Không overwrite dữ liệu cũ.

## 5.4. Schema version

```python
schema_version: Literal["scene_document_v2"]
```

Index mới dùng alias riêng:

```text
aic_scenes_v2
```

Không cập nhật collection cũ tại chỗ.

## DoD W0

* Search result luôn có `best_frame_idx`.
* Mapping frame round-trip pass.
* Alias task pass.
* Endpoint cũ vẫn hoạt động qua compatibility layer.
* Không candidate nào thiếu `video_id`.
* Không publish reference đến artifact chưa tồn tại.

---

# 6. Workstream W1 — branch registry và identity

Log đã phát hiện `QueryExpansionRetriever` đổi tên thành `bm25_caption_expanded`, nhưng candidate bên trong vẫn mang `source="bm25_caption"`, khiến capability và branch weight không khớp. 

Cần tách:

```text
branch_id      = adapter ổn định
execution_id   = branch + query variant
```

Ví dụ:

```text
branch_id: caption_bm25
execution_id: caption_bm25.raw

branch_id: caption_bm25
execution_id: caption_bm25.expanded
```

Candidate sử dụng `execution_id` làm `source`.

```python
class RetrieverExecution(StrictModel):
    execution_id: str
    branch_id: str
    query_variant: str
    options: BranchRuntimeOptions
```

Registry:

```python
class RetrieverRegistry:
    def register(self, retriever: Retriever) -> None: ...
    def resolve(self, branch_id: str) -> Retriever: ...
    def capabilities(self) -> list[BranchCapabilities]: ...
```

## 6.1. Sửa dense retriever local

Log cho thấy `dense_visual` local hiện thực chất là BM25/hash trên caption và keyword, không phải dense visual thật. 

Không quảng cáo nó là dense retrieval.

Hai lựa chọn:

```text
A. Đổi tên thành lexical_hash_fallback.
B. Giữ dense_visual nhưng capabilities báo:
   backend_kind = "lexical_fallback"
   degraded = true
```

Khuyến nghị A để không làm sai số liệu ablation.

## DoD W1

* Mọi branch có `branch_id`, `modality`, `backend_kind`.
* Capability ID khớp candidate source.
* Base và expanded có thể chỉnh weight độc lập.
* Dense fallback không được report như vector search.

---

# 7. Workstream W2 — Retrieval Orchestrator có khả năng chịu lỗi

Hiện `asyncio.gather()` không có timeout và `return_exceptions=True`; một branch lỗi có thể làm toàn request trả 500. 

## 7.1. Branch execution

```python
async def execute_branch(
    execution: RetrieverExecution,
    query: PreparedQuery,
) -> BranchExecutionResult:
    try:
        result = await asyncio.wait_for(
            execution.retriever.retrieve(...),
            timeout=execution.options.timeout_ms / 1000,
        )
        return BranchExecutionResult.success(result)
    except asyncio.TimeoutError:
        return BranchExecutionResult.timeout(...)
    except Exception as exc:
        return BranchExecutionResult.failed(
            error_code=type(exc).__name__,
        )
```

## 7.2. Không để một branch làm chết search

```python
branch_results = await asyncio.gather(
    *tasks,
    return_exceptions=False,
)
```

Mỗi task đã tự bắt exception và chuyển thành typed status.

## 7.3. Execution state

```python
class BranchStatus(StrictModel):
    execution_id: str
    status: Literal[
        "success",
        "disabled",
        "unavailable",
        "timeout",
        "failed",
        "empty"
    ]
    latency_ms: int
    candidate_count: int
    warning: str | None
```

Response phải có:

```json
{
  "status": "COMPLETED_WITH_WARNINGS",
  "branch_status": {}
}
```

Không silent fallback.

## DoD W2

* Event/Qdrant timeout không làm toàn request 500.
* UI thấy branch nào lỗi.
* Có p50/p95 theo branch.
* Có timeout riêng trong `SearchOptions`.
* Search vẫn hoàn tất nếu còn ít nhất một branch thành công.

---

# 8. Workstream W3 — đấu nối toàn bộ SearchOptions

Audit ghi nhận khoảng 20 field hiện được schema accept nhưng không có consumer. Điều này nguy hiểm vì UI nhận `200 OK` dù backend bỏ qua cấu hình. 

Mỗi field phải có đúng một owner:

| Field                     | Consumer                      |
| ------------------------- | ----------------------------- |
| `enabled`                 | Retrieval planner             |
| `weight`                  | Fusion                        |
| `top_k`                   | Retriever invocation          |
| `min_score`               | Threshold service             |
| `threshold_space`         | Score normalization/threshold |
| `threshold_policy`        | Threshold service             |
| `query_variant`           | Query service                 |
| `timeout_ms`              | Orchestrator                  |
| `field_weights`           | Lexical retriever             |
| `model_id`                | Model registry                |
| `index_id`                | Index registry                |
| `fusion_top_k`            | Fusion                        |
| `rrf_k`                   | Fusion                        |
| `normalized_score_method` | Normalizer                    |
| `dedup_scope`             | Dedup service                 |
| `dedup_similarity`        | Dedup service                 |
| `max_results_per_video`   | Result planner                |
| `display_top_k`           | Response formatter            |
| `group_by`                | Result aggregation/UI         |
| `sort_by`                 | Response formatter            |
| `enable_hyde`             | Query understanding           |
| `enable_temporal_parsing` | Query parser                  |

## 8.1. Strict validation

Trong giai đoạn migration:

```text
Implemented option → chạy thật.
Unsupported option → 422.
Unavailable capability → 409 hoặc warning rõ.
```

Không chấp nhận rồi bỏ qua.

## 8.2. Capabilities-driven UI

```text
GET /v1/search/capabilities
```

Trả cho từng branch:

```json
{
  "id": "caption_bm25",
  "available": true,
  "supported_controls": [
    "enabled",
    "weight",
    "top_k",
    "min_score",
    "threshold_policy",
    "query_variant",
    "timeout_ms",
    "field_weights"
  ]
}
```

---

# 9. Workstream W4 — score normalization, threshold và fusion

Fusion hiện là tầng mạnh nhất của codebase; có nhiều method, branch weight và deterministic tie-break. Nên giữ lõi này và bổ sung các tầng còn thiếu thay vì viết lại. 

## 9.1. BranchScore

```python
class BranchScore(StrictModel):
    raw_score: float
    normalized_score: float
    percentile_score: float | None

    score_kind: Literal[
        "cosine",
        "inner_product",
        "bm25",
        "fuzzy_ratio",
        "confidence",
        "histogram_similarity",
        "reranker"
    ]

    normalization_method: str
    calibration_version: str | None
```

## 9.2. Baseline normalization

| Score               | Baseline                       |
| ------------------- | ------------------------------ |
| Cosine              | `(score + 1) / 2`              |
| BM25                | percentile trong result branch |
| OCR fuzzy           | giữ nguyên 0–1                 |
| Detector confidence | giữ nguyên                     |
| HSV similarity      | giữ nguyên                     |
| Reranker logit      | sigmoid hoặc percentile        |

## 9.3. Threshold

```python
if policy == "hard":
    remove candidate below threshold
else:
    apply soft penalty
```

Phải áp threshold **trước fusion** đối với branch-level threshold.

## 9.4. Fusion output

Ngoài `component_scores`, cần:

```python
branch_contributions: dict[str, float]
```

Với RRF:

```python
contribution = weight / (rrf_k + rank)
```

`fusion_top_k` phải dùng đúng, không thay bằng `candidate_limit` mơ hồ.

---

# 10. Workstream W5 — aggregation và dedup

Hiện fusion vô tình gom candidate theo `scene_id`, nhưng chưa phải dedup service thật. 

Tạo:

```text
online/services/aggregation.py
online/services/deduplication.py
```

Các scope:

```text
frame
scene
event
video-time-window
visual-near-duplicate
```

Output aggregated:

```python
class AggregatedCandidate(StrictModel):
    candidate_id: str
    video_id: str

    scene_id: str | None
    event_id: str | None

    best_frame: FrameEvidence
    supporting_frames: list[FrameEvidence]

    matched_branches: list[str]
    branch_scores: dict[str, BranchScore]
    branch_contributions: dict[str, float]
```

Task policy:

```text
KIS   → dedup scene/event nhưng giữ vài video alternative.
QA    → giữ nhiều frame evidence cùng event khi cần.
TRAKE → không dedup các step khác nhau.
AVS   → dedup event mạnh và thêm diversity.
```

---

# 11. Workstream W6 — Evidence Pack

Evidence không thể chỉ là `{modality, text, score}`.

```python
class EvidencePack(StrictModel):
    candidate_id: str
    video_id: str
    scene_id: str | None
    event_id: str | None

    start_frame: int
    end_frame: int

    keyframes: list[FrameEvidence]
    asr_window: str | None

    previous_context: NeighborContext | None
    next_context: NeighborContext | None

    branch_scores: dict[str, BranchScore]
    branch_contributions: dict[str, float]
    rule_adjustments: list[RuleAdjustment]

    model_versions: dict[str, str]
    index_versions: dict[str, str]
```

Endpoint:

```text
GET /v1/evidence/{candidate_id}
```

Evidence được build lazy cho top candidate, không hydrate toàn bộ hàng nghìn candidate.

---

# 12. Workstream W7 — Rerank cascade

## Stage 0 — Rules

Giữ `rules.py`, nhưng chuyển must-match từ suy luận dấu ngoặc kép sang query signature thật.

Rules:

* exact OCR/ASR boost;
* must-match coverage;
* rare cue;
* negative constraint;
* temporal consistency;
* cross-branch agreement;
* duplicate teaser penalty.

## Stage 1 — Text reranker

Input text:

```text
caption
OCR
ASR
objects
actions
scene/event context
```

Không đưa raw JSON khổng lồ.

Baseline:

```text
BGE reranker
top 300 → 80
```

## Stage 2 — VLM reranker

```text
top 20
1–5 representative frames
query + compact evidence
```

Model failure phải fallback về Stage 1, có warning.

## Stage 3 — Temporal verifier

Chỉ bật khi query có:

```text
before/after
ordered steps
tracking
transition
TRAKE
```

---

# 13. Workstream W8 — bốn task processors

## 13.1. Textual KIS

```text
Query
→ known-item signature
→ broad retrieval
→ fusion/rerank
→ dedup
→ evidence
→ safe-frame selection
→ ranked frame results
```

Known-item signature:

```python
class KisSignature(StrictModel):
    must_match: list[Constraint]
    nice_to_have: list[Constraint]
    rare_cues: list[Constraint]
    negative_constraints: list[Constraint]
```

Safe-frame:

```text
semantic score
+ quality score
+ interval centrality
+ OCR/action evidence
- scene-boundary penalty
- blur penalty
```

Output bắt buộc có `best_frame_idx`.

## 13.2. QA

Current `EvidenceOnlyAnswerGenerator` chỉ nối caption/OCR/ASR thành text, chưa phải answerer. 

Flow:

```text
Question
→ event query + answer target
→ answer-type router
→ evidence retrieval
→ task tool
→ answer candidates
→ normalization
→ verifier
→ joint frame-answer ranking
```

Tools baseline:

| Answer type    | Tool                          |
| -------------- | ----------------------------- |
| OCR            | OCR exact/fuzzy               |
| ASR            | transcript lookup             |
| Count          | object boxes + temporal dedup |
| Color          | color metadata + VLM          |
| Entity/object  | caption/object/VLM            |
| Yes/no         | evidence verifier             |
| Temporal       | neighboring scenes/events     |
| Multi-evidence | evidence aggregator           |

Result:

```python
class QaResultItem(StrictModel):
    rank: int
    video_id: str
    frame_idx: int
    answer: str
    canonical_answer: str
    joint_score: float
    verifier_status: str
    evidence_ids: list[str]
```

## 13.3. TRAKE

Không giữ TRAKE như 41 dòng trong `temporal.py`. Tách thành first-class service như audit yêu cầu. 

### Stage A — video retrieval

* Retrieve từng step.
* Aggregate theo video.
* Step coverage.
* Ordered-pair coverage.
* Global context.
* Rank top videos.

### Stage B — sequence search

* Filter toàn bộ branch theo video.
* Retrieve scene/clip/event cho từng step.
* Beam search.
* Strictly increasing frame/time.
* Gap constraint.

### Stage C — frame refinement

* Decode frame window.
* Stride 1.
* Score từng frame.
* Chọn semantic moment.
* Giữ top alternatives.

Output:

```python
class TrakeResultItem(StrictModel):
    rank: int
    video_id: str
    frame_ids: list[int]
    sequence_score: float
    step_scores: list[float]
    uncertainty: list[float]
```

## 13.4. AVS

Current `_diversify_avs` chỉ giới hạn kết quả mỗi video, chưa đủ. 

Flow:

```text
inclusion/exclusion
→ high-recall retrieval
→ relevance grade 0–3
→ threshold
→ event dedup
→ clustering
→ MMR
→ ranked segment list
```

AVS result:

```python
class AvsResultItem(StrictModel):
    rank: int
    video_id: str
    segment_id: str
    start_frame: int
    end_frame: int

    relevance_grade: int
    score: float
    cluster_id: str | None
```

---

# 14. Workstream W9 — competition/submission layer

Tạo riêng:

```text
online/competition/
```

Không nhét submission logic vào `SearchService`.

## 14.1. Contracts

```python
class KisSubmission(StrictModel):
    video_id: str
    frame_idx: int

class QaSubmission(StrictModel):
    video_id: str
    frame_idx: int
    answer: str

class TrakeSubmission(StrictModel):
    video_id: str
    frame_ids: list[int]
```

## 14.2. Validator

Kiểm tra:

* Tối đa 100 items.
* Frame không âm.
* Frame thuộc video.
* QA answer không rỗng.
* TRAKE đủ số step.
* True frame index, không phải keyframe ordinal.
* Dataset/index version đúng.
* Duplicate warning.
* Rank liên tục.

## 14.3. Local scorer

```text
KIS:
video đúng + frame trong interval

QA:
video đúng + frame trong interval + answer đúng

TRAKE:
sai video = 0
đúng video = mean(step frame hit)
```

Gold schema hiện tại cũng đã yêu cầu lưu interval, answer, ordered moments và hard negatives để đánh giá retrieval/rerank đúng cách. 

---

# 15. Workstream W10 — API thống nhất

## 15.1. Unified search

```text
POST /v1/search
POST /v1/search/stream
```

Convenience endpoints chỉ là wrapper:

```text
POST /v1/search/kis
POST /v1/search/qa
POST /v1/search/trake
POST /v1/search/avs
```

Wrapper không chứa logic riêng.

## 15.2. Supporting endpoints

```text
POST /v1/query/prepare
GET  /v1/search/capabilities
GET  /v1/search/presets
POST /v1/search/config/validate

GET  /v1/evidence/{candidate_id}
GET  /v1/videos/{video_id}/frame-window
GET  /v1/scenes/{scene_id}
GET  /v1/events/{event_id}

POST /v1/rerank
POST /v1/trake/retrieve-videos
POST /v1/trake/align

POST /v1/submissions/build
POST /v1/submissions/validate
POST /v1/submissions/evaluate-local
POST /v1/submissions/send

POST /v1/search-sessions
GET  /v1/search-sessions/{session_id}
POST /v1/search-sessions/{session_id}/replay
```

## 15.3. Streaming events

```text
search_started
query_prepared
branch_started
branch_completed
branch_failed
fusion_completed
rerank_completed
evidence_ready
alignment_completed
search_completed
```

---

# 16. Workstream W11 — UI đồng bộ theo backend thật

Chỉ React UI cần parity đầy đủ. Vanilla giữ làm basic fallback.

```text
Competition Retrieval Studio
├── Query Studio
├── Search Mixing Console
├── Results Explorer
├── Evidence Inspector
├── KIS Safe Frame Workspace
├── QA Evidence Studio
├── TRAKE Alignment Studio
├── AVS Relevance/Diversity Workspace
├── Submission Board
├── Compare Lab
└── Health Drawer
```

## 16.1. UI đọc capabilities

Không hard-code branch.

Mỗi card render từ:

```text
branch_id
available
supported_controls
default_config
score_kind
threshold_space
model/index version
```

## 16.2. Search Mixing Console

Mỗi branch:

* Enable.
* Weight.
* Top-k.
* Threshold.
* Hard/soft.
* Query variant.
* Timeout.
* Model/index ở Expert mode.

## 16.3. Result card

Bắt buộc hiển thị:

* `video_id`.
* `frame_idx`.
* `timestamp`.
* scene/event.
* branch contribution.
* safe-frame score.
* evidence preview.
* branch/model/index provenance.

## 16.4. Submission board

Các vùng:

```text
#1
#2–5
#6–20
#21–50
#51–100
```

Cho phép:

* drag reorder;
* replace frame;
* QA answer edit;
* TRAKE step edit;
* validation;
* export;
* submission history.

---

# 17. Workstream W12 — sessions và reproducibility

Mỗi search phải lưu:

```python
class SearchExecutionTrace(StrictModel):
    session_id: str
    request_id: str

    raw_request: SearchRequest
    resolved_config: ResolvedSearchConfiguration
    prepared_query: PreparedQuery

    branch_status: dict[str, BranchStatus]
    branch_result_refs: dict[str, str]

    fusion_config: FusionOptions
    rerank_config: RerankOptions

    model_versions: dict[str, str]
    index_versions: dict[str, str]
    dataset_version: str
    backend_build_id: str

    timings_ms: dict[str, int]
    warnings: list[str]
```

Cần hỗ trợ:

* replay;
* compare;
* export config;
* notebook error analysis;
* reproduction sau khi đổi index.

---

# 18. Offline phải đồng bộ theo online contract

Online không thể hoàn chỉnh nếu offline không xuất đủ dữ liệu.

## 18.1. Artifact bắt buộc

```text
video_manifest
scene_manifest
frame_manifest
keyframe metadata
clip manifest
event manifest
caption table
OCR table
ASR table
object/action table
color table
embedding artifacts
index manifests
```

Project docs đã xác định `frame_idx` là chỉ số chuẩn để submission và metadata phải truy ngược được về video, scene, timestamp, frame path, evidence và model sinh evidence. 

## 18.2. Frame embedding store

Không tiếp tục encode frame rồi vứt vector.

Tạo:

```text
offline/artifacts/embedding_store.py
```

Quy trình:

```text
encode frame
→ atomic write
→ checksum
→ EmbeddingReference
→ populate Keyframe.embedding_refs
→ scene/clip/event pooling dùng chung
```

## 18.3. Clip/event

W1 phải hoàn tất:

```text
color             ✓
clip pooling
action tags
event grouping
event aggregation
```

Scene, clip và event phải dùng ID/version khác nhau; không dùng ba khái niệm như nhau.

---

# 19. Model và index registry

Không hard-code model/index trong adapter.

```yaml
models:
  visual_frame:
    official_clip_b32:
      enabled: true
      dimension: 512

    openclip_l14:
      enabled: true
      dimension: 768

  reranker_text:
    bge_v2_m3:
      endpoint: ...

indexes:
  official_clip_b32_v1:
    backend: qdrant
    collection: aic_frames_v2
    vector_name: official_clip_b32

  clip_pool_v1:
    backend: qdrant
    collection: aic_clips_v1
    vector_name: clip_pool_v1
```

Capabilities đọc registry này.

---

# 20. Evaluation gate thống nhất

Dùng gold mini benchmark version hóa. Audit hiện tại ghi nhận benchmark `L21_V001` gồm 40 query: 12 KIS, 12 QA, 8 AVS và 8 TRAKE; bộ toy bốn dòng chỉ nên được giữ làm smoke data. 

## 20.1. Test pyramid

### Unit

* Models/schema.
* Normalization.
* Threshold.
* Fusion.
* Dedup.
* Answer normalization.
* TRAKE scorer.
* Submission validator.

### Integration

* Query → branch → fusion → frame hit.
* Partial branch failure.
* Evidence hydration.
* QA tuple.
* TRAKE video-lock + sequence.
* AVS MMR.
* Session replay.

### E2E

* KIS search and submit.
* QA answer and submit.
* TRAKE frame alignment.
* AVS bulk selection.
* Advanced SearchOptions payload.
* Restore session after refresh.

## 20.2. Test bắt buộc thêm

```text
test_frame_mapping_roundtrip.py
test_task_alias_and_conflict.py
test_branch_execution_timeout.py
test_search_options_all_consumed.py
test_branch_source_identity.py
test_normalized_score_contract.py
test_threshold_hard_soft.py
test_fusion_contributions.py
test_scene_event_dedup.py
test_evidence_pack.py
test_kis_safe_frame.py
test_qa_joint_result.py
test_trake_video_first.py
test_trake_frame_refinement.py
test_avs_mmr.py
test_submission_max_100.py
test_official_task_scores.py
test_search_session_replay.py
```

---

# 21. Commit/PR sequence an toàn

Không thực hiện tất cả trong một commit.

## PR-01 — Canonical contracts

* Task enum.
* FrameEvidence.
* Candidate V2.
* SearchHit V2.
* Compatibility aliases.
* Schema regeneration.

## PR-02 — Data projection

* SceneDocument V2.
* Frame mapping.
* Migration script.
* Index alias V2.

## PR-03 — Branch runtime hardening

* Registry.
* Stable source/execution ID.
* Dense fallback rename.
* Timeout.
* Branch status.
* Partial failure.

## PR-04 — SearchOptions execution

* Đấu nối toàn bộ fields.
* Strict unsupported validation.
* Capabilities.

## PR-05 — Normalization/fusion/dedup

* BranchScore.
* Threshold.
* Fusion contribution.
* Aggregation.

## PR-06 — Evidence/rerank

* EvidencePack.
* BGE.
* VLM adapter.
* Temporal verifier.

## PR-07 — Task processors

* KIS.
* QA.
* TRAKE.
* AVS.

## PR-08 — Competition layer

* Submission builder.
* Validator.
* Local scorer.
* Ranking zones.

## PR-09 — API/session/stream

* Unified endpoint.
* SSE.
* Session persistence.
* Replay.

## PR-10 — React UI

* Capability-driven mixer.
* Evidence.
* KIS/QA/TRAKE/AVS workspaces.
* Submission board.

## PR-11 — Production gate

* E2E.
* Load smoke.
* Monitoring.
* Deployment docs.
* Release manifest.


---

# 22. Definition of Done toàn hệ thống V1

## Contracts

* Một taxonomy duy nhất.
* `frame_idx` xuyên suốt.
* Không task/body conflict im lặng.
* Mọi artifact/version truy vết được.

## Retrieval

* Branch controls đều chạy thật.
* Per-branch timeout.
* Partial failure.
* Score normalization.
* Hard/soft threshold.
* Per-branch fusion.
* Dedup rõ ràng.

## Tasks

* KIS trả frame.
* QA trả frame-answer tuple.
* TRAKE trả ordered frame list.
* AVS trả segment có relevance/diversity.

## Evidence

* Evidence pack đầy đủ.
* BGE/VLM rerank hoạt động.
* QA verifier.
* TRAKE frame refinement.

## Submission

* Format theo task.
* Max 100.
* Local scorer.
* True frame validation.
* History/retry.

## UI

* Simple/Advanced/Expert.
* Không control giả.
* Branch status.
* Evidence inspector.
* Task workspace.
* Submission board.
* Session restore.

## Evaluation

* Unit/integration/E2E pass.
* Default regression không giảm ngoài tolerance đã chốt.
* Metric theo task.
* Config, model, index và dataset version được lưu.

---

# 23. Việc nên bắt đầu ngay

Đợt đầu tiên nên giới hạn vào một vertical foundation:

```text
Task taxonomy
+ frame_idx propagation
+ SceneDocument V2
+ Candidate/SearchHit V2
+ unified /v1/search
+ submission contracts
+ compatibility tests
```

Không nên bắt đầu bằng BGE, Qwen rerank hay UI nâng cao khi online còn chưa trả được `frame_idx`.

Sau vertical foundation đó, thứ tự đúng là:

```text
runtime reliability
→ SearchOptions thật
→ normalization/dedup
→ evidence/rerank
→ bốn task processors
→ submission
→ UI hoàn chỉnh
```

Đây là con đường ngắn nhất để biến codebase từ khoảng 30% thành một hệ thống thống nhất, thay vì tiếp tục tăng số module nhưng vẫn không tạo được output đúng luật thi đấu.
