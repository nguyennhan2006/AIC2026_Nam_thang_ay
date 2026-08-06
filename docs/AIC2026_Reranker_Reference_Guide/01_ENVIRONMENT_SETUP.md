# 01 — Chuẩn bị môi trường

## 1. Tách hai môi trường

Không nên dùng một environment duy nhất cho cả text reranker và Qwen3-VL.

```text
aic-rerank-text
aic-rerank-vl
```

Lý do:

- VLM thường yêu cầu processor, video/image dependency và phiên bản Transformers mới hơn.
- Quantization/FlashAttention có thể xung đột với môi trường text ổn định.
- Tách môi trường giúp rollback và tái lập dễ hơn.

## 2. Environment text reranker

### Conda

```bash
conda create -n aic-rerank-text python=3.11 -y
conda activate aic-rerank-text

python -m pip install --upgrade pip
python -m pip install \
  "torch" \
  "transformers>=4.51.0" \
  "sentence-transformers>=5.4,<6" \
  "datasets" \
  "accelerate" \
  "huggingface-hub" \
  "scikit-learn" \
  "pandas" \
  "pyyaml" \
  "jsonschema"
```

`transformers>=4.51.0` được giữ vì model card Qwen3-Reranker cảnh báo phiên bản cũ hơn có thể không nhận kiến trúc `qwen3`.

### Tùy chọn FlagEmbedding

```bash
python -m pip install "FlagEmbedding[finetune]"
```

Nên cài trong environment text riêng. Sau khi chạy ổn, xuất lock:

```bash
python -m pip freeze > requirements_text_locked.txt
```

## 3. Environment multimodal reranker

Cách ưu tiên là dùng repository chính thức vì repository có script setup và lock riêng.

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
cd Qwen3-VL-Embedding
bash scripts/setup_environment.sh
source .venv/bin/activate
```

Trên Windows, nên chạy phần này trong WSL2 hoặc máy Linux GPU. Không trộn package của repository này vào environment text đang ổn.

## 4. Kiểm tra GPU trước khi cài phần tăng tốc

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Không cài FlashAttention chỉ vì có CUDA. Trước tiên chạy baseline bằng attention mặc định. Chỉ thêm FlashAttention sau khi model inference đã đúng.

## 5. Kiểm tra package

```bash
python - <<'PY'
import torch
import transformers
import sentence_transformers
import datasets
import accelerate
import huggingface_hub

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("huggingface_hub:", huggingface_hub.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 6. Kaggle

- Cài dependency ở cell đầu tiên.
- Restart kernel sau khi cài hoặc nâng phiên bản.
- Không import `transformers`, sau đó mới nâng cấp `transformers`.
- Lưu model/cache dưới `/kaggle/working` nếu muốn đóng gói output.
- Nếu model đã có trong Kaggle Dataset, ưu tiên mount read-only thay vì tải lại.

## 7. Server/VastAI

Khuyến nghị:

```text
/workspace/
├── envs/
├── hf_cache/
├── models/
├── data/
├── runs/
└── code/
```

Đặt cache ở volume bền vững để không tải lại model sau khi restart container.
