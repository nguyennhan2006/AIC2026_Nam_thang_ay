# 08 — Troubleshooting

## 1. `KeyError: qwen3`

Nguyên nhân thường gặp: Transformers quá cũ.

Kiểm tra:

```bash
python -c "import transformers; print(transformers.__version__)"
```

Model card Qwen3-Reranker yêu cầu `transformers>=4.51.0`.

## 2. Model tải lại mỗi lần

Kiểm tra:

- `HF_HOME`;
- `HF_HUB_CACHE`;
- local model path;
- container volume có bền vững không;
- notebook có thay cache dir không.

## 3. OOM

Giảm theo thứ tự:

1. batch size;
2. max length;
3. số frame;
4. resolution;
5. candidate count mỗi batch;
6. bật fp16/bf16;
7. gradient accumulation;
8. quantization/LoRA sau cùng.

## 4. Score giống nhau

Kiểm tra:

- candidate text có thật sự khác nhau;
- truncation có cắt phần quan trọng;
- instruction có được áp dụng;
- activation có làm score nén quá mạnh;
- model có đang ở eval mode;
- document formatter có lặp cùng nội dung.

## 5. Score dương/âm khó hiểu

Qwen3-Reranker có thể trả raw logit difference. Với ranking, raw score vẫn dùng được. Không so sánh threshold giữa hai model nếu activation/calibration khác nhau.

## 6. BGE API thay đổi

Ưu tiên code example trong model card/repository đúng revision. Không copy code từ blog cũ rồi nâng package tùy ý.

## 7. JSONL load lỗi

Hugging Face Datasets khuyến nghị mỗi dòng là một JSON object. Mọi dòng phải có type nhất quán.

Sai:

```json
{"label": 1}
{"label": "positive"}
```

Đúng:

```json
{"label": 1.0}
{"label": 0.0}
```

## 8. Train/dev metric cao bất thường

Kiểm tra leakage:

- cùng `video_id` ở train và dev;
- cùng scene với paraphrase query ở hai split;
- hard negative được sinh từ gold dev;
- teacher prompt chứa gold answer;
- candidate text chứa query ID/difficulty tag.

## 9. Fine-tune làm retrieval kém

Khả năng:

- negative quá khó;
- chỉ có một domain/video;
- label teacher nhiễu;
- overfit;
- document format train khác inference;
- loss tốt nhưng ranking objective không phù hợp;
- positive bị thiếu do candidate recall thấp.

## 10. Multimodal score không ổn định

Kiểm tra:

- frame order;
- số frame;
- video decoding;
- ảnh bị resize sai;
- metadata và image không cùng candidate;
- candidate path trỏ nhầm;
- sampling không deterministic.

## 11. `device_map="auto"` trong training

Tài liệu Transformers lưu ý `device_map="auto"` nên được xem là lựa chọn inference. Với training, dùng Accelerate/DeepSpeed/FSDP hoặc cấu hình device rõ ràng.

## 12. Bitsandbytes lỗi CUDA

- kiểm tra GPU/OS được hỗ trợ;
- kiểm tra PyTorch CUDA;
- không cài nhiều bản bitsandbytes chồng nhau;
- thử full precision baseline trước;
- lưu exact error và `pip freeze`.

## 13. Kaggle kernel hỏng sau `pip install`

Restart session. Không tiếp tục dùng class đã import từ phiên bản cũ.
