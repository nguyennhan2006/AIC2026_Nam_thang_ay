# 03 — Smoke test model

Mỗi model phải vượt qua bốn gate:

1. Load thành công.
2. Chạy được một batch nhỏ.
3. Score positive cao hơn negative.
4. Ghi được thời gian và VRAM.

## 1. Qwen3-Reranker-0.6B bằng Sentence Transformers

```python
import time
import torch
from sentence_transformers import CrossEncoder

model_path = "/workspace/models/Qwen3-Reranker-0.6B"

model = CrossEncoder(
    model_path,
    prompts={
        "aic": (
            "Judge whether the candidate video metadata matches the AIC video "
            "retrieval query. Focus on the described moment, visible evidence, "
            "OCR, ASR, actions, and temporal constraints."
        )
    },
    default_prompt_name="aic",
)

query = "Một người đang cào muối trên đồng."
documents = [
    "Cảnh nhiều người dùng dụng cụ cào muối trắng trên ruộng muối.",
    "Một đám cháy rừng với khói dày trên sườn núi.",
]

start = time.perf_counter()
scores = model.predict([(query, d) for d in documents])
elapsed = time.perf_counter() - start

print("scores:", scores)
print("elapsed_sec:", elapsed)
assert scores[0] > scores[1], "Smoke test ranking failed"
```

Model card cho biết score mặc định có thể là raw logit difference. Không áp sigmoid nếu chỉ cần ranking; áp sigmoid nếu cần score 0–1 và phải ghi rõ activation trong manifest.

## 2. BGE reranker

```python
from FlagEmbedding import FlagReranker

model_path = "/workspace/models/bge-reranker-v2-m3"
reranker = FlagReranker(model_path, use_fp16=True)

pairs = [
    ["Một người đang cào muối.", "Nhiều người đang cào muối trên cánh đồng trắng."],
    ["Một người đang cào muối.", "Khói bốc lên từ một đám cháy rừng."],
]

scores = reranker.compute_score(pairs, normalize=True)
print(scores)
assert scores[0] > scores[1]
```

Nếu API thay đổi theo phiên bản FlagEmbedding, ưu tiên example trong repository/model card của revision đang dùng.

## 3. Qwen3-VL-Reranker-2B

Ưu tiên dùng wrapper trong repository chính thức:

```python
from src.models.qwen3_vl_reranker import Qwen3VLReranker

model = Qwen3VLReranker(
    model_name_or_path="/workspace/models/Qwen3-VL-Reranker-2B",
)

inputs = {
    "instruction": (
        "Rank a candidate frame by how precisely it matches the described "
        "AIC video moment."
    ),
    "query": {"text": "A person raking salt in a salt field."},
    "documents": [
        {"image": "/workspace/smoke/positive.jpg"},
        {"image": "/workspace/smoke/negative.jpg"},
    ],
    "fps": 1.0,
    "max_frames": 8,
}

scores = model.process(inputs)
print(scores)
assert scores[0] > scores[1]
```

Bắt đầu với image đơn. Chỉ chuyển sang video/clip sau khi image smoke test đạt.

## 4. Ghi VRAM

```python
import torch

if torch.cuda.is_available():
    print("allocated_GB:", torch.cuda.max_memory_allocated() / 1024**3)
    print("reserved_GB:", torch.cuda.max_memory_reserved() / 1024**3)
```

Trước mỗi phép đo độc lập:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

## 5. Smoke-test report

```json
{
  "model_id": "Qwen/Qwen3-Reranker-0.6B",
  "revision": "<commit_sha>",
  "status": "pass",
  "positive_score": 7.1,
  "negative_score": -8.3,
  "batch_size": 2,
  "max_length": 1024,
  "latency_sec": 0.42,
  "peak_vram_gb": 2.1
}
```

Không chạy thí nghiệm full nếu positive không cao hơn negative trên smoke case được kiểm tra bằng tay.
