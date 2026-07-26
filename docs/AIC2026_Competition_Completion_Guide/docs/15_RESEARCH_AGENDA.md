# 15. Research agenda đảm bảo chất lượng service

## 1. Nguyên tắc

- Không kết luận khi chưa đo.
- Tập 14 query chỉ smoke test.
- Mỗi experiment có hypothesis, method, metric, acceptance, result và date.
- Báo cáo per-query regressions.
- Không chỉ nhìn metric tổng.

## A. Ground truth và validity

1. Frame vs scene vs event tolerance ảnh hưởng metric thế nào?
2. Có nhiều positive hợp lệ không?
3. Hard-negative review có thay đổi model ranking không?
4. Reviewer agreement và ambiguity rate?

## B. Query understanding

1. Raw Q0 vs normalized Q1 vs LLM main topic vs Q0+subqueries.
2. PreparedQueryPlanner có lợi với query dài không?
3. OCR term có nên đưa vào dense query?
4. VI→EN expansion có gây drift không?
5. Must-match hard filter vs soft penalty vs rerank-only.
6. Progressive clue: full rerun vs incremental reweight.
7. Rule parser vs LLM planner vs hybrid.

## C. Sparse retrieval

1. OCR fuzzy song song hay thay BM25 OCR?
2. Tune `min_score` và token ratio.
3. ES analyzer vi vs BM25 local.
4. Field weights caption/OCR/ASR/object/action.
5. Exact phrase/numeric boost.
6. Crop/region caption có tăng lexical recall không?

## D. Dense retrieval

1. SigLIP2 vs CLIP-L/14 vs OpenCLIP vs ensemble.
2. Frame vs scene vs event embeddings.
3. Clip embedding cho action/temporal query.
4. Image-to-image model và crop query.
5. Query language VI/EN/bilingual theo encoder.

## E. Fusion/rerank

1. RRF k ∈ {20,40,60,100}.
2. Branch top-k allocation.
3. Weighted fusion vs RRF.
4. Rule bonus scale.
5. BGE top-N cascade.
6. Qwen3-VL rerank top-N.
7. Rerank rubric components.
8. Hard-negative pairwise comparison.
9. p95 latency vs Recall/MRR.

## F. Temporal/event

1. Frame-only vs scene vs event fusion.
2. Neighbor event có giảm time-to-correct không?
3. Ordered A→B→C scorer.
4. Temporal gap function.
5. Conditional B-after-A retrieval.
6. Event segmentation granularity.

## G. KIS interaction

1. Entity toggle có tăng operator success rate?
2. More-like-this bằng image vs image+text.
3. Grid/list/timeline/event view.
4. Neighbor frame count tối ưu.
5. Progressive clue rank improvement.
6. Exact-frame UI và wrong-frame rate.

Metric thêm: time-to-correct, interactions, opened videos, time-to-submit.

## H. VQA

1. Evidence granularity: frame/scene/clip/event.
2. Single VLM vs routed OCR/ASR/count/tool/VLM.
3. Count: direct VLM vs detector vs multi-frame.
4. Tracking utility.
5. Evidence compression.
6. Same-model self-check vs independent verifier.
7. Abstention threshold.
8. Human-assisted correction.

## I. AVS

1. Raw rank vs per-video cap vs temporal dedup vs MMR vs clustering.
2. Relevance grade model/human agreement.
3. Fixed vs adaptive threshold.
4. Scene vs event result granularity.
5. Diversity strength vs mAP/nDCG.
6. Basket review UI.

## J. UI/HCI

1. Grid size.
2. Keyboard shortcuts.
3. Score/evidence visibility.
4. Query editor complexity.
5. Session restore.
6. Cognitive load.
7. Time-to-submit.
8. Team board coordination benefit.

## K. Reliability/cost

1. Branch deadline.
2. Warm/cold latency.
3. Cache hit.
4. Quantization.
5. GPU allocation.
6. Failure injection.
7. Submission retry policy.
8. Degraded mode quality.

## L. Composed image-text retrieval — P2

1. Score interpolation vs caption+text vs LLM composed embedding.
2. Refine từ candidate gần đúng.
3. Search same object with changed action/scene.
4. Synthetic triplet generation có đáng không?

## Ghi kết quả

Dùng `templates/experiment_record.yaml`. Khi có kết quả, cập nhật ngay hypothesis thành `accepted/rejected/inconclusive` kèm metric và ngày.
