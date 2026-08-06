# 07 — Ma trận thí nghiệm và đánh giá

## 1. Ma trận tối thiểu

| ID | Retriever | Reranker | Train | Mục tiêu |
|---|---|---|---|---|
| E0 | Current | None | No | Baseline |
| E1 | Current | BGE v2-m3 | No | Zero-shot text baseline |
| E2 | Current | Qwen3 0.6B | No | Instruction-aware text baseline |
| E3 | Current | BGE/custom CrossEncoder | Yes | Fine-tune bằng pointwise + hard negative |
| E4 | Current | Text reranker | Distillation | Học soft/graded label |
| E5 | Current | Qwen3-VL 2B | No | Multimodal zero-shot |
| E6 | Current | Text top-100 → VLM top-20 | Optional | Cascade |
| E7 | Current | VLM teacher → text student | Yes | Distill visual teacher |

## 2. Không thay nhiều thứ cùng lúc

Ví dụ E1 và E2 phải dùng cùng:

- candidate pool;
- query set;
- candidate text format;
- top-k;
- split;
- metric code.

## 3. Metric theo tầng

### Candidate recall gate

Trước rerank:

```text
candidate_recall@20
candidate_recall@100
```

Nếu candidate đúng không có trong pool, không đổ lỗi cho reranker.

### Rerank metric

```text
top1_pairwise_accuracy
MRR
nDCG@K
R@1/R@5/R@20
```

### QA

```text
evidence_recall
evidence_rank
answer_accuracy
joint_answer_evidence
```

### TRAKE

```text
event_candidate_recall
frame_oracle_coverage
frame_selection_accuracy
ordered_chain_score
```

## 4. Ablation bắt buộc

- no instruction vs AIC instruction;
- original query vs bilingual/normalized query;
- caption only vs caption+OCR+ASR;
- random negatives vs mixed hard negatives;
- label 0/1 vs grade 0–3;
- text reranker vs VLM;
- 1/3 keyframes vs short clip;
- raw score vs calibrated score;
- text-only cascade vs text→VLM.

## 5. Significance và độ ổn định

Với tập query nhỏ:

- báo cáo từng query;
- bootstrap theo query nếu đủ mẫu;
- báo cáo mean/min/max/std;
- chạy nhiều seed nếu fine-tune;
- không chỉ báo cáo một con số mean.

## 6. Latency

Báo cáo:

```text
candidate count
batch size
input token length
number of frames
latency median
latency p95
peak VRAM
```

## 7. Run manifest

Dùng schema `schemas/run_manifest.schema.json`.

## 8. Điều kiện chốt model

Giữ một variant khi:

1. Candidate recall không giảm.
2. KIS MRR/R@1 hoặc metric task chính tăng.
3. Không chỉ tăng trên một nhóm query nhỏ.
4. Latency/VRAM trong budget.
5. Kết quả tái lập được từ manifest.
6. Không có leakage.
