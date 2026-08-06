# 05 — Thí nghiệm text reranker

## 1. Thứ tự ưu tiên

### Baseline A — BGE zero-shot

Lợi ích:

- Dễ inference.
- Có hệ sinh thái FlagEmbedding.
- Có code fine-tune và hard-negative mining.

### Baseline B — Qwen3-Reranker-0.6B zero-shot

Lợi ích:

- Multilingual.
- Instruction-aware.
- Có thể dùng custom instruction cho AIC.

### Fine-tune đầu tiên

Nên bắt đầu với BGE/FlagEmbedding hoặc CrossEncoder được Sentence Transformers hỗ trợ rõ ràng. Model card Qwen3-Reranker chủ yếu cung cấp inference; việc fine-tune Qwen cần được xem là một bước engineering riêng, không nên giả định có script chính thức hoàn chỉnh.

## 2. Tạo text document từ candidate

```text
[SCENE]
{dense_caption}

[OBJECTS]
{objects}

[ACTIONS]
{actions}

[OCR]
{ocr_text}

[ASR]
{asr_text}

[PREVIOUS]
{previous_caption}

[NEXT]
{next_caption}
```

Quy tắc:

- Giới hạn chiều dài từng field.
- Không lặp caption nhiều lần.
- Không nhét raw JSON quá dài.
- Giữ field marker ổn định giữa train và inference.
- Không đưa retrieval rank vào text.

## 3. Baseline inference

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "/workspace/models/Qwen3-Reranker-0.6B",
    prompts={
        "aic_kis": (
            "Given an AIC Known-Item Search query, judge whether the candidate "
            "metadata describes the exact requested video moment."
        )
    },
    default_prompt_name="aic_kis",
)

pairs = [(row["query"], row["candidate_text"]) for row in rows]
scores = model.predict(pairs, batch_size=8)
```

## 4. CrossEncoder training skeleton

```python
from datasets import load_dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

dataset = load_dataset(
    "json",
    data_files={
        "train": "pointwise_train.jsonl",
        "validation": "pointwise_dev.jsonl",
    },
)

model = CrossEncoder(
    "<sequence-classification-compatible-checkpoint>",
    num_labels=1,
)

loss = BinaryCrossEntropyLoss(model=model)

args = CrossEncoderTrainingArguments(
    output_dir="runs/text_reranker/checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=200,
    logging_steps=20,
    load_best_model_at_end=True,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    loss=loss,
)

trainer.train()
```

Tên field dataset phải được map đúng theo API của Sentence Transformers revision đang dùng. Skeleton này là contract tham khảo, không thay thế example chính thức.

## 5. Chuyển grade 0–3 sang target

### Phương án nhị phân

```text
grade 3 -> 1.0
grade 2 -> 0.67
grade 1 -> 0.33
grade 0 -> 0.0
```

### Phương án pairwise

Tạo cặp khi chênh grade đủ lớn:

```text
grade(A) - grade(B) >= 1
```

Không tạo quá nhiều cặp dễ `3 > 0`; ưu tiên `3 > 2`, `2 > 1`, `1 > 0` và hard-negative pairs.

## 6. Hyperparameter grid nhỏ

```yaml
max_length: [512, 1024]
learning_rate: [1.0e-5, 2.0e-5]
epochs: [1, 2, 3]
hard_negative_ratio: [0.3, 0.5, 0.7]
batch_size: [4, 8, 16]
gradient_accumulation_steps: [1, 2, 4]
```

Không grid-search tất cả tổ hợp. Dùng staged ablation:

1. Chốt data format.
2. Chốt max length.
3. Chốt negative mixture.
4. Chốt learning rate/epoch.
5. Chốt instruction.

## 7. Metric model-level

- Pairwise accuracy.
- AUC hoặc average precision cho positive/negative.
- Calibration chỉ khi cần threshold.
- Score distribution theo label.
- Accuracy theo hard-negative subtype.

## 8. Metric bắt buộc downstream

- KIS R@1/R@5/R@20/MRR.
- QA evidence hit.
- TRAKE event selection/frame selection.
- Zero-result rate.
- Latency/query.
- Peak VRAM.
