# 04 — Pipeline chưng cất nhãn

## 1. Candidate generation

Với mỗi query, lấy union từ nhiều retriever:

```text
visual dense
caption dense/sparse
OCR
ASR
object/action
temporal/event
```

Không chỉ lấy top-k từ một retriever.

## 2. Nguồn nhãn

Độ ưu tiên:

```text
gold
> human-corrected
> multi-teacher consensus
> single-teacher high confidence
> heuristic weak label
```

## 3. Pointwise label

Khuyến nghị grade 0–3:

| Grade | Ý nghĩa |
|---:|---|
| 3 | Đúng video/moment và đủ must-match |
| 2 | Gần đúng hoặc thiếu một phần evidence |
| 1 | Cùng chủ đề nhưng sai moment/điều kiện |
| 0 | Không liên quan |

Schema nằm tại `schemas/pointwise_label.schema.json`.

## 4. Pairwise label

Tạo cặp có ý nghĩa:

- positive > hard negative;
- exact moment > near miss;
- complete evidence > partial evidence;
- correct order > wrong order;
- correct OCR/visual combination > OCR-only false positive.

Schema nằm tại `schemas/pairwise_label.schema.json`.

## 5. Hard-negative mixture khởi đầu

```yaml
hard_negative: 0.50
medium_negative: 0.25
random_negative: 0.15
near_gt_negative: 0.10
```

Đây là cấu hình tham khảo, phải ablation.

## 6. Teacher prompt contract

```text
System:
You are assigning training labels for an AIC video retrieval reranker.
Use only the supplied query and candidate evidence.
Do not infer hidden facts.
Do not use the candidate's original retrieval rank as evidence.

Return:
- relevance_grade: 0, 1, 2, or 3
- matched_constraints
- missing_constraints
- contradicted_constraints
- evidence_ids
- confidence
- ambiguous
```

## 7. Teacher context

Cho teacher:

- query;
- task;
- 1–5 keyframes hoặc clip ngắn;
- caption;
- OCR;
- ASR;
- object/action;
- previous/next context.

Không cho:

- gold answer nếu teacher đang tạo nhãn độc lập;
- raw rank;
- fused score;
- tên variant retrieval;
- lời giải dài từ model khác.

## 8. Quality gates

Chỉ nhận pseudo-label khi:

- JSON/schema hợp lệ;
- evidence ID tồn tại;
- không bịa OCR/ASR/object;
- confidence vượt threshold;
- teacher không mâu thuẫn gold;
- nhiều lần shuffle candidate không đảo nhãn mạnh;
- không có train/test leakage.

## 9. Sample weight

```text
gold/human:                 1.00
multi-teacher agreement:   0.85
single teacher confident:  0.60–0.70
weak heuristic:            0.20–0.30
```

## 10. Split

Bắt buộc split theo `video_id`, không split ngẫu nhiên theo pair.

```text
train videos
dev videos
test videos
```

Nếu số video quá ít, dùng leave-one-video-out hoặc k-fold theo video.

## 11. Version dataset

```text
labels/
└── reranker_labels_v001/
    ├── manifest.json
    ├── pointwise_train.jsonl
    ├── pointwise_dev.jsonl
    ├── pairwise_train.jsonl
    ├── pairwise_dev.jsonl
    ├── rejected.jsonl
    └── statistics.json
```
