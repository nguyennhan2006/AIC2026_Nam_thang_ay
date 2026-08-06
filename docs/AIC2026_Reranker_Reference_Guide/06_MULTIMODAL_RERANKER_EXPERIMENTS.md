# 06 — Thí nghiệm multimodal reranker

## 1. Vai trò

Qwen3-VL-Reranker-2B nhận query-document pair, trong đó query hoặc document có thể gồm text, image, video hoặc mixed input.

Nên dùng ở một trong hai vai trò:

1. Teacher tạo pseudo-label.
2. Stage-2 reranker cho top 10–30 candidate sau text reranker.

Không nên bắt đầu bằng cách chạy VLM trên top 300 candidate.

## 2. Input ladder

Chạy từ đơn giản đến phức tạp:

```text
L0: query + 1 keyframe
L1: query + 3 keyframes
L2: query + 3 keyframes + compressed metadata
L3: query + short clip
L4: query + short clip + compressed metadata
```

Mỗi level phải có ablation độc lập.

## 3. Keyframe selection

Không gửi tất cả frame của scene. Khởi đầu:

```text
start representative
center representative
end representative
```

Hoặc ba frame có visual diversity cao nhất.

## 4. Video limits

Repository chính thức có input `fps` và `max_frames`. Bắt đầu thấp:

```yaml
fps: 1.0
max_frames: 8
```

Sau đó thử:

```yaml
max_frames: [8, 16, 32]
```

Không tăng frame trước khi chứng minh retrieval gain.

## 5. Instruction AIC

```text
Given an AIC video retrieval query and a candidate frame or clip,
judge whether the candidate contains the exact requested moment.
Use visible actions, objects, spatial relations, OCR, ASR metadata,
and temporal order. A candidate that only matches the broad topic
should receive a lower relevance score.
```

## 6. Inference pattern

```python
from src.models.qwen3_vl_reranker import Qwen3VLReranker

model = Qwen3VLReranker(
    model_name_or_path="/workspace/models/Qwen3-VL-Reranker-2B",
)

inputs = {
    "instruction": "<AIC instruction>",
    "query": {"text": query_text},
    "documents": [
        {
            "text": compressed_metadata,
            "image": frame_path,
        }
    ],
    "fps": 1.0,
    "max_frames": 8,
}

score = model.process(inputs)
```

## 7. LoRA reference

Repository Qwen3-VL-Embedding công bố cấu hình khởi đầu:

```yaml
rank: 32
alpha: 32
target_modules:
  - q_proj
  - v_proj
  - k_proj
  - up_proj
  - down_proj
  - gate_proj
```

Đây là reference từ repository, không phải cấu hình đã tối ưu cho AIC.

## 8. Quantization

- Chỉ thêm 8-bit/4-bit sau khi fp16/bf16 inference đúng.
- Dùng PEFT cho adapter training.
- `device_map="auto"` phù hợp hơn cho inference; không dùng như mặc định cho training.
- Ghi rõ quantization config vào manifest.
- So sánh score/ranking giữa full precision và quantized trên smoke/dev set.

## 9. Teacher labeling

Teacher VLM nên trả:

```json
{
  "relevance_grade": 0,
  "matched_constraints": [],
  "missing_constraints": [],
  "evidence_frame_ids": [],
  "confidence": 0.0,
  "ambiguous": false
}
```

Chạy lại với candidate order hoặc frame order đã shuffle để đo độ ổn định.

## 10. Khi nào giữ VLM reranker

Chỉ giữ nếu:

- text reranker không phân biệt được hai candidate metadata gần giống;
- caption bỏ sót chi tiết quyết định;
- lỗi chính là visual grounding;
- downstream metric tăng đủ lớn so với latency;
- không làm true positive rơi khỏi top-k.

## 11. Khi nào loại

- chỉ tăng score model-level nhưng không tăng MRR/R@K;
- quá nhạy với frame sampling;
- score thay đổi lớn khi đảo thứ tự frame;
- OOM thường xuyên;
- latency không phù hợp;
- teacher hallucinate evidence.
