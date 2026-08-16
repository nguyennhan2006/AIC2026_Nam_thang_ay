# Chạy full offline pipeline (Qwen2.5-VL-7B) trên Kaggle T4x2

Máy local không đủ RAM/VRAM cho Qwen2.5-VL-7B (đã xác nhận: cần ~12.5GB RAM chỉ để
load qua Ollama, máy chỉ còn ~9.7GB rảnh). Kaggle notebook với accelerator **GPU T4x2**
cho GPU thật (2 x T4, 16GB VRAM/GPU) để chạy caption + semantic OCR bằng
Qwen2.5-VL-7B-Instruct thật, cùng OWLv2 (object), CLIP (embedding), Whisper (ASR) —
đúng nghĩa "full phần offline", không còn placeholder nào.

Script mẫu: [`notebooks/kaggle_offline_qwen.py`](../notebooks/kaggle_offline_qwen.py)
— copy từng khối `# %% [n]` thành 1 cell trong notebook Kaggle, chạy tuần tự.

## Notebook rời theo từng stage

Ngoài luồng full ở trên, mỗi stage nặng có một notebook độc lập, code nằm TRỌN
trong notebook (không import repo) nên upload thẳng lên Kaggle là chạy được:

| Notebook | Stage | Model | Ghi chú |
|---|---|---|---|
| [`caption-l21-a.ipynb`](../notebooks/caption-l21-a.ipynb) | caption | Qwen2.5-VL-3B | trả 2 dòng `CAPTION:`/`OBJECTS:` |
| [`ocr-l21-a.ipynb`](../notebooks/ocr-l21-a.ipynb) | OCR | Qwen2.5-VL | |
| [`asr-l21-a.ipynb`](../notebooks/asr-l21-a.ipynb) | ASR | Whisper | |
| [`scene-l21-a.ipynb`](../notebooks/scene-l21-a.ipynb) | scene split | — | |
| [`embed-jina-clip-v2.ipynb`](../notebooks/embed-jina-clip-v2.ipynb) | **embedding ảnh** | **jina-clip-v2** | xem mục dưới |

### `embed-jina-clip-v2.ipynb` — vector ảnh đa ngữ

Sinh vector ảnh 1024 chiều cho `dense_visual`. Khác luồng CLIP ở bốn điểm, đều
là hệ quả của việc chạy FULL DATA chứ không phải một video:

- **Một `.npy` cho mỗi video**, không phải một JSON cho mỗi frame. Layout
  file-mỗi-vector ổn với 855 keyframe; hàng trăm nghìn file nhỏ làm chậm cả zip
  lẫn giải nén và đụng giới hạn output của Kaggle. Cờ `EXPLODE_PER_FRAME=True`
  bung về layout cũ khi cần cắm vào bản chạy local.
- **Lưu float16** — vector đã L2-normalize nên nằm gọn trong [-1, 1]; fp16 đủ
  cho cosine và giảm nửa dung lượng. Cast lại float32 trước khi nhân.
- **Phải tải sẵn `jinaai/jina-clip-implementation`** (cell 3). jina-clip nạp
  code từ một repo RỜI qua `auto_map`; worker chạy `HF_HUB_OFFLINE=1` sẽ chết
  lúc `from_pretrained` với thông báo không liên quan gì tới mạng.
- **Cần `timm` + `einops`**, không cần `flash_attn` (code tự tắt khi không có
  CUDA, và trên T4 thì dùng attention native).

⚠️ `vector_uri` ghi dạng `vectors/<video>.npy#<row>` — trỏ vào MỘT HÀNG của ma
trận. `_read_vector_file` ([frame_vector_store.py:24](../online/adapters/frame_vector_store.py#L24))
hiện chỉ đọc được file-một-vector, CHƯA hiểu cú pháp `#row`. Cắm vào bản local
thì bật `EXPLODE_PER_FRAME`, hoặc sửa reader trước.

⚠️ jina-clip-v2 là **CC-BY-NC-4.0** (phi thương mại), khác CLIP (MIT-ish).

Cell smoke test và cell cuối đều in `cosine cặp ngẫu nhiên`. Nếu trung bình gần
1.0 thì mọi ảnh ra cùng một vector và index vô dụng — trong khi shape, norm và
mọi thứ khác vẫn trông hoàn toàn bình thường. Đừng bỏ qua hai cell đó.

## Lưu ý quan trọng trước khi chạy

- **T4x2 = 2 GPU rời 16GB, không phải 1 pool 32GB.** Code dùng
  `device_map="auto"` (thư viện `accelerate`) để tự động chia Qwen2.5-VL-7B ra cả
  2 GPU — nếu không có `accelerate` hoặc chỉ 1 GPU, model 7B ở fp16 (~15-16GB) có
  thể sát hoặc vượt giới hạn 1 GPU 16GB. Nếu gặp OOM, xem mục "Nếu vẫn OOM" bên dưới.
- Dùng **float16**, không dùng bfloat16 — T4 (kiến trúc Turing) không có tensor core
  bf16 nhanh, fp16 chạy ổn định và nhanh hơn trên T4.
- Lần chạy đầu tiên tải khoảng **~16GB trọng số model** (Qwen2.5-VL-7B + OWLv2 +
  CLIP + Whisper) — có thể mất 10-20+ phút tùy tốc độ mạng Kaggle.
- Bật **Internet = On** và **Accelerator = GPU T4 x2** trong Notebook Settings trước
  khi chạy cell đầu tiên.

## Bước 1 — Tạo Kaggle Dataset cho video

Trên máy local, video test đã chuẩn bị sẵn ở `storage/raw/videos/L16_V001.mp4`
(clip 90s cắt từ `input/K16_V001.mp4`, xem [[aic2026-local-offline-run]] nếu bạn
cần lại bối cảnh). Vào kaggle.com → **Create → New Dataset** → upload file này →
đặt tên dataset (vd `aic2026-l16-v001`) → Create.

Nếu muốn chạy nguyên video gốc 15 phút thay vì clip 90s, upload
`input/K16_V001.mp4`, đổi tên file thành dạng canonical trước khi upload (ví dụ
`L16_V001.mp4`) vì pipeline bắt buộc tên file khớp `^L\d{2}_V\d{3}$`.

## Bước 2 — Tạo notebook, gắn dataset

- New Notebook → Add Data → chọn dataset vừa tạo ở Bước 1.
- Settings (góc phải) → Accelerator: **GPU T4 x2**; Internet: **On**.
- Copy nội dung [`notebooks/kaggle_offline_qwen.py`](../notebooks/kaggle_offline_qwen.py)
  vào các cell theo đúng thứ tự `# %% [1]` → `# %% [9]`.

## Bước 3 — Chạy từng cell theo thứ tự

1. `nvidia-smi` — xác nhận thấy 2 GPU Tesla T4.
2. Clone repo. Repo public thì chạy thẳng; **nếu repo private**, làm theo hướng dẫn
   trong comment của cell (thêm secret `GITHUB_TOKEN` qua Add-ons → Secrets, dùng
   Personal Access Token có quyền `repo` từ GitHub Settings → Developer settings).
3. Cài dependency — **không cài lại `torch`**, Kaggle đã có sẵn bản build CUDA sẵn;
   cài lại `torch` từ PyPI mặc định có thể ghi đè bằng bản CPU-only và làm hỏng GPU.
4. Sửa `DATASET_SLUG` cho khớp tên dataset bạn đặt ở Bước 1, rồi copy video vào
   `storage/raw/videos/`.
5. Khởi động GPU worker nội bộ (cùng kiến trúc với triển khai Vast.ai thật —
   xem `docs/05_VAST_DEPLOYMENT.md` — chỉ khác là chạy chung 1 máy Kaggle thay vì
   tách backend/worker).
6. Chạy `python -m offline run` — đây là bước tải model + suy luận thật, chờ lâu
   nhất. Theo dõi log để biết đang ở giai đoạn nào (probe/scene/keyframe/enrich).
7. `python -m offline index --encoder remote` — build FAISS artifact từ embedding
   CLIP thật (bỏ qua `--qdrant` vì Kaggle không có Qdrant server).
8. Validate (`datasection.cli`, `scripts.preflight`) rồi nén toàn bộ `storage/`
   thành `/kaggle/working/offline_output.zip`.
9. Dừng worker để giải phóng GPU (dọn dẹp, không bắt buộc).

## Bước 4 — Tải kết quả về máy local và merge

Tải `offline_output.zip` từ panel **Output** của notebook, giải nén đè lên
`storage/` trong repo local (thay cho dữ liệu mock hiện có). Sau đó ở máy local:

```bash
python -m datasection.cli storage/exports
python -m scripts.preflight
python -m offline index          # rebuild index cục bộ khớp encoder local nếu cần test backend=local
python -m scripts.eval_kis --metadata storage/exports/scenes.jsonl \
  --groundtruth examples/kis_groundtruth_L16_V001.jsonl --mode all
```

So sánh bảng Recall@K/MRR này với bảng chạy trên metadata mock trước đó (R@1=0 mọi
mode) để biết Qwen2.5-VL có thật sự cải thiện retrieval hay không — đúng nguyên tắc
evaluation-first đã thống nhất trước đó (xem [[aic2026-search-roadmap]]).

Sau đó khởi động lại API local (`uvicorn online.api.app:app`) để test query search
qua `/v1/search/kis` như trước — nhớ restart vì container chỉ load `scenes.jsonl`
một lần lúc khởi động.

## Nếu vẫn OOM trên GPU

- Giảm `AIC_CAPTION_MAX_TOKENS` (mặc định 120) để giảm bộ nhớ sinh token.
- Đổi sang bản nhỏ hơn: `AIC_CAPTION_MODEL=Qwen/Qwen2.5-VL-3B-Instruct` (chất lượng
  thấp hơn 7B nhưng chắc chắn vừa 1 GPU 16GB).
- Kiểm tra `accelerate` đã cài đúng phiên bản (`pip show accelerate`) — thiếu
  `accelerate` thì `device_map="auto"` không chia được sang GPU thứ 2.

## Giới hạn đã biết

- Object detection (OWLv2), embedding (CLIP), ASR (Whisper) vẫn pin cứng vào GPU 0
  (`AIC_GPU_DEVICE=0`) — chỉ Qwen2.5-VL mới tự chia sang cả 2 GPU. Với 1 video ngắn
  việc này không phải nút thắt cổ chai.
- Semantic OCR bbox lấy trực tiếp từ output `bbox_2d` của Qwen2.5-VL (theo đúng
  format model được huấn luyện) — chính xác tương đối, không kỳ vọng bằng OCR
  chuyên dụng (PaddleOCR/EasyOCR) cho bbox pixel-perfect.
