# Source register

Ngày tổng hợp: 2026-08-05

## Hugging Face Hub

**Tài liệu:** `Download files from the Hub`

Dùng để tham khảo:

- `hf download`;
- `snapshot_download`;
- cache;
- `revision`;
- `allow_patterns`/`ignore_patterns`;
- `local_dir`;
- `HF_HOME`.

## Qwen text reranker

**Model ID:** `Qwen/Qwen3-Reranker-0.6B`

Dùng để tham khảo:

- model type, parameter count, languages, context length;
- Sentence Transformers usage;
- Transformers requirement;
- raw logit difference;
- custom instruction;
- vLLM example.

## BGE reranker

**Model ID:** `BAAI/bge-reranker-v2-m3`

**Repository:** `FlagOpen/FlagEmbedding`

Dùng để tham khảo:

- inference;
- reranker fine-tune;
- hard-negative mining;
- package installation.

## Sentence Transformers

**Tài liệu:** `Cross Encoder Training Overview`

Dùng để tham khảo:

- reranker là stage-2;
- model/dataset/loss/training arguments/evaluator/trainer;
- `CrossEncoder`;
- `CrossEncoderTrainer`.

**Tài liệu:** `Rerankers`

Dùng để tham khảo:

- single-score cross-encoder;
- pairwise input;
- reranking evaluation.

## Qwen multimodal reranker

**Model ID:** `Qwen/Qwen3-VL-Reranker-2B`

**Repository:** `QwenLM/Qwen3-VL-Embedding`

Dùng để tham khảo:

- text/image/video/mixed-modal input;
- pointwise relevance score;
- environment setup;
- download model;
- wrapper inference;
- `fps` và `max_frames`;
- LoRA rank/alpha/target modules;
- vLLM requirement của repository.

## PEFT

**Tài liệu:** `PEFT Quicktour` và `LoRA`

Dùng để tham khảo:

- adapter training;
- `LoraConfig`;
- `get_peft_model`;
- target modules;
- giảm số trainable parameters.

## Transformers quantization

**Tài liệu:** `Bitsandbytes`

Dùng để tham khảo:

- 8-bit/4-bit;
- giảm memory;
- QLoRA;
- lưu ý device map cho inference/training.

## Hugging Face Datasets

**Tài liệu:** `Load`

Dùng để tham khảo:

- load JSON/JSONL;
- mỗi dòng là một JSON object;
- local/remote files;
- Arrow cache.

## Accelerate

**Tài liệu:** `Accelerate`

Dùng để tham khảo:

- launcher;
- distributed training;
- mixed precision;
- FSDP/DeepSpeed integration.

## Cách dùng source register

Khi code khác với tài liệu này:

1. Kiểm tra đúng revision model/repository.
2. Kiểm tra release note của thư viện.
3. Ưu tiên example chính thức mới nhất.
4. Cập nhật manifest và lock file.
5. Không sửa im lặng rồi tiếp tục so sánh với run cũ.
