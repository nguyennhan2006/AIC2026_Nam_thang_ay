# 05. Query understanding, retrieval, fusion và reranking

## 1. Nguyên tắc query

Luôn giữ:

```text
Q0 = raw query
Q1 = normalized query
Q2 = bilingual/cross-lingual query
Q3...Qn = decomposed/expanded subqueries
```

Không được dùng `main topic` để thay thế Q0. Parser có thể hỗ trợ, không được độc quyền quyết định retrieval.

## 2. ParsedQuery tối thiểu

- Task.
- Language.
- Granularity.
- Objects/people/attributes/actions.
- Scene/spatial/camera/colors.
- OCR/ASR/named entities/numbers/exact phrases.
- Before-after/ordered steps/duration/tracking target.
- Negative constraints.
- Must-match/nice-to-have.
- Branch routing.
- Task-specific fields.

## 3. Routing baseline

Rule-based trước:

- Có quoted text/named entity/number → OCR/ASR boost.
- Có “nói/đọc/nghe” → ASR.
- Có “biển/bảng/chữ/số” → OCR.
- Có before/after/then → temporal.
- Có “tất cả/các cảnh” → AVS broad profile.
- Count/how many → VQA count route.

LLM planner chỉ bổ sung khi query phức tạp hoặc rule confidence thấp.

## 4. Retrieval branches

### Dense

- Frame embedding.
- Scene embedding.
- Clip/video embedding.
- Image-to-image.
- Region/crop.

### Sparse

Field riêng:

```text
short_caption
dense_caption
event_caption
object_tags
action_tags
relation_tags
ocr_text
asr_text
named_entities
```

Không nối tất cả field thành một document duy nhất nếu cần weight/debug.

### Exact/fuzzy

- OCR exact phrase.
- OCR fuzzy normalized.
- ASR exact/fuzzy.
- Numeric exact.

### Event/Temporal

- Event caption search.
- Neighbor event.
- Ordered sequence join.
- Conditional B-after-A.

## 5. Candidate contract

Mỗi branch trả:

```json
{
  "candidate_id": "...",
  "rank": 1,
  "raw_score": 0.71,
  "normalized_score": 0.83,
  "branch": "dense_frame_siglip2",
  "query_variant_id": "Q0",
  "payload": {"video_id": "...", "segment_id": "...", "frame_idx": 123}
}
```

## 6. Fusion

### Baseline

RRF với `rrf_k` đo trên dev set.

### Sau baseline

- Field/branch weight.
- Query-adaptive profile.
- Learning-to-rank.

### Quy tắc

- Exact OCR/ASR có thể boost nhưng không hard-filter khi confidence thấp.
- Object/attribute là soft constraints.
- Metadata thiếu không đồng nghĩa candidate sai.

## 7. Dedup

Gộp khi:

- Cùng scene.
- Timestamp gần nhau.
- Cùng event.
- Visual similarity vượt threshold.

Giữ:

- Best representative frame.
- Supporting frames.
- Branch evidence tổng hợp.

## 8. Rerank cascade

```text
Top 1000 retrieval
→ dedup 300–500
→ BGE/text metadata rerank top 300→50
→ VLM/multi-frame rerank top 20
→ temporal/evidence verifier
```

Các trần phải được tune theo Recall và p95 latency.

## 9. Rerank rubric

- Must-match coverage.
- Nice-to-have coverage.
- Visual semantic match.
- OCR/ASR exactness.
- Action/interaction match.
- Scene relation.
- Temporal match.
- Contradictions.
- Evidence sufficiency.

Output phải có lý do ngắn và score components.

## 10. Interactive feedback

- Positive selected candidate.
- Negative candidate.
- Keyword add/remove.
- Same object/scene/action.
- Search from crop.
- Image + modification text.

Feedback chỉ tác động session hiện tại trừ khi người dùng chủ động export thành training/replay data.

## 11. Hướng từ các solution/paper

- Hệ thống thắng thực tế cho thấy vector retrieval và giao diện entity toggle có giá trị cao.
- Sparse lexical metadata hữu ích với keyword, tên riêng, OCR/ASR và refinement nhiều vòng.
- Crop caption có thể tăng lexical coverage nhưng phải ablation; không mặc định 17 crop cho mọi frame.
- Composed image-text retrieval là nhánh P2, phù hợp refine từ candidate gần đúng, không chặn baseline.
