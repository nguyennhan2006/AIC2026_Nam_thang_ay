# 09. Ground truth, evaluation và ablation

## 1. Ground truth hiện trạng và mục tiêu

Tập nhỏ chỉ dùng smoke test. Để chốt production default cần dev set đủ lớn và phân tầng.

Mục tiêu tối thiểu ban đầu:

- ≥50 query KIS.
- ≥50 query VQA.
- ≥50 query AVS.

Nhưng quan trọng hơn là coverage theo nhóm khó.

## 2. Stratification

### KIS

- Visual-only.
- Object + attribute.
- Action.
- OCR-heavy.
- ASR-heavy.
- Temporal before/after.
- Ordered sequence.
- Negative constraints.
- Progressive clue.

### VQA

- Visual lookup.
- OCR.
- ASR.
- Count.
- Tracking.
- Temporal.
- Multi-evidence.
- Insufficient evidence.

### AVS

- Broad concept.
- Activity/action.
- Interaction.
- Inclusion/exclusion.
- Many valid results.
- Diversity-sensitive.

## 3. Ground truth schema

Mỗi item nên có:

- Primary positives.
- Secondary positives.
- Valid intervals.
- Relevant events.
- Hard negatives.
- Ambiguity flag.
- Reviewer IDs/agreement.

## 4. Metric

### KIS

- R@1/5/20/50/100.
- MRR.
- Video-R@K.
- Hit at frame/scene/event tolerance.

### VQA

- EM/F1.
- Numeric/count accuracy.
- Evidence hit@K.
- Groundedness.
- Hallucination.
- Abstention precision/recall.

### AVS

- mAP.
- nDCG.
- P/R@K.
- Unique relevant event count.
- Redundancy.

### Interactive

- Time-to-first-correct.
- Time-to-submit.
- Interactions/query.
- Opened videos/query.
- Wrong-frame submission rate.

## 5. Experiment contract

Mỗi experiment lưu:

- hypothesis;
- dataset split/version;
- model/index/prompt versions;
- flags;
- metrics;
- per-query outputs;
- latency;
- cost/VRAM;
- conclusion;
- accepted/rejected;
- date/owner.

Dùng template `templates/experiment_record.yaml`.

## 6. Acceptance gates

Feature được bật khi:

- Metric chính tăng có ý nghĩa thực tế.
- Không làm nhóm query quan trọng tụt nghiêm trọng.
- p95 trong budget.
- Không tăng failure rate quá ngưỡng.
- Có rollback.

Không dùng nguyên tắc “overall tăng nên chấp nhận” nếu OCR/ASR/temporal group tụt mạnh.

## 7. CI eval gate

- Smoke tests trên mỗi PR.
- Full dev eval cho branch release.
- Baseline JSON trong repo/artifact store.
- Fail nếu metric tụt ngoài tolerance.
- Report per-query regressions.

## 8. Benchmark ambiguity

Quy trình:

1. Lấy top hard negatives.
2. Human review.
3. Nếu nhiều candidate hợp lệ, thêm secondary positive hoặc đánh dấu ambiguous.
4. Không ép một frame duy nhất nếu toàn interval hợp lệ.
5. Báo cáo metric trên clean và full set.
