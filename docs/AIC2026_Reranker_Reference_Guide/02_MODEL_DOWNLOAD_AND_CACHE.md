# 02 — Tải model và quản lý cache

## 1. Cấu hình cache

### Bash/Linux

```bash
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" /workspace/models
```

### PowerShell

```powershell
$env:HF_HOME="D:\aic\hf_cache"
$env:HF_HUB_CACHE="D:\aic\hf_cache\hub"
New-Item -ItemType Directory -Force $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force "D:\aic\models" | Out-Null
```

## 2. Đăng nhập

Chỉ cần cho model gated/private:

```bash
hf auth login
```

Không lưu token vào notebook hoặc Git.

## 3. Tải bằng CLI

### Qwen3 text reranker

```bash
hf download Qwen/Qwen3-Reranker-0.6B \
  --local-dir /workspace/models/Qwen3-Reranker-0.6B
```

### BGE reranker

```bash
hf download BAAI/bge-reranker-v2-m3 \
  --local-dir /workspace/models/bge-reranker-v2-m3
```

### Qwen3-VL multimodal reranker

```bash
hf download Qwen/Qwen3-VL-Reranker-2B \
  --local-dir /workspace/models/Qwen3-VL-Reranker-2B
```

## 4. Pin revision

Sau khi xác nhận model chạy đúng:

```bash
hf download Qwen/Qwen3-Reranker-0.6B \
  --revision <commit_sha> \
  --local-dir /workspace/models/Qwen3-Reranker-0.6B_<short_sha>
```

Không tự điền một commit giả. Lấy commit hash từ trang Files and versions hoặc API Hub.

## 5. Tải bằng Python

```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-Reranker-0.6B",
    revision="<commit_sha_or_main_for_smoke_only>",
    local_dir="/workspace/models/Qwen3-Reranker-0.6B",
)
print(path)
```

Hugging Face Hub hỗ trợ `allow_patterns` và `ignore_patterns`, nhưng không nên lọc file tùy tiện khi chưa hiểu đầy đủ cấu trúc model.

## 6. Kiểm tra model đã tải đủ

```bash
find /workspace/models/Qwen3-Reranker-0.6B -maxdepth 2 -type f | sort
du -sh /workspace/models/Qwen3-Reranker-0.6B
```

Kiểm tra tối thiểu:

- `config.json`
- tokenizer/processor files
- weight `.safetensors`
- index file nếu model được shard
- model card hoặc README
- generation/config file nếu model yêu cầu

## 7. Offline mode

Sau khi tải xong và smoke test thành công:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Load model bằng local path thay vì model ID để tránh vô tình lấy revision mới.

## 8. Manifest model

Tạo file:

```yaml
model_id: Qwen/Qwen3-Reranker-0.6B
local_path: /workspace/models/Qwen3-Reranker-0.6B
revision: "<commit_sha>"
downloaded_at: "<ISO-8601>"
source: huggingface_hub
smoke_test_status: pass
notes: ""
```

## 9. Không nên làm

- Không dùng `git clone` cho repo model lớn nếu không hiểu Git LFS/Xet.
- Không tải lại model vào nhiều thư mục cho từng experiment.
- Không để mỗi notebook tự tải model.
- Không thay đổi cache path giữa các lần chạy mà không ghi manifest.
- Không xóa cache khi chưa xác nhận local model hoàn chỉnh.
