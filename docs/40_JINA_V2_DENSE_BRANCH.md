# 40 — Bật nhánh jina v2 dense (caption dense + jina-clip-v2)

Ngày: 2026-08-25. Bổ sung vào [docs/36](36_CHAY_HE_THONG.md).

---

## Tóm tắt

Nhánh `caption_dense` dùng **jinaai/jina-clip-v2** để encode văn bản caption thành vector 1024 chiều, rồi tìm kiếm bằng cosine similarity. Khác với `dense_visual` (encode ảnh), `caption_dense` khớp query text ↔ caption text.

**Kết quả đo trên 25 câu P1:**

| Method | Video@1 | Video@10 | Frame@10 |
|---|---|---|---|
| BM25 | 24,0% | 48,0% | 8,0% |
| **jina v2** | **76,0%** | **76,0%** | **88,0%** |

+52pp recall video@1 so với BM25.

---

## Kiến trúc

```
Query (VI text)
    ↓ encode với jina-clip-v2 task=retrieval.query
Vector 1024d (L2-normalized)
    ↓ dot product (cosine)
FAISS IndexFlatIP / numpy matrix
    ↓ top-K
Scene candidates
```

**Hai phần cần có:**

1. **Backend format** (`embeddings.npy` + `scene_ids.json` + `manifest.json`) — đọc bởi `CaptionDenseRetriever`
2. **JinaClipV2Encoder** (`online/adapters/dense_text.py`) — encode query bằng `model.encode_text(task="retrieval.query")`

---

## Bước 1 — Tạo backend artifacts từ Kaggle output

Chạy **MỘT LẦN** trên máy đã có Kaggle output:

```bash
python scripts/convert_jina2_to_backend.py
```

Script tạo 3 file trong `storage/caption_embedding_jina_v2/`:

| File | Dung lượng | Ý nghĩa |
|---|---|---|
| `embeddings.npy` | ~592 MB | Ma trận vector float32, 151.459 hàng × 1024 cột |
| `scene_ids.json` | ~2,6 MB | Danh sách scene_id tương ứng |
| `manifest.json` | ~380 B | Ghi `encoder_kind: jina_clip_v2`, query_prefix, model_id |

**Nếu chạy trên VastAI** (không có mapping files):
- Upload toàn bộ thư mục `caption_embedding_jina_v2/` từ máy local qua SCP
- Hoặc chạy script trên local rồi upload chỉ 3 file backend (`embeddings.npy`, `scene_ids.json`, `manifest.json`)

---

## Bước 2 — Cấu hình `.env.fpt.local`

Thêm vào `.env.fpt.local`:

```bash
# Đường dẫn TƯƠNG ĐỐI từ gốc repo (không dùng Unicode path trên Windows)
AIC_CAPTION_DENSE_INDEX=storage/caption_embedding_jina_v2
AIC_CAPTION_DENSE_ENCODER=jina_clip_v2
AIC_CAPTION_DENSE_MODEL=jinaai/jina-clip-v2
```

**Lưu ý về coverage:**
- Jina v2 artifacts chứa **823/873 video** (168.414 keyframe → 151.459 vector)
- 50 video mới (L26_V300-V349) **CHƯA** có trong artifacts
- Coverage ~94% video / ~15% scene (do deduplication keyframe→scene)
- Server **khởi động THÀNH CÔNG** với coverage warning thay vì crash
- Nhánh vẫn chạy, chỉ là không tìm được scene từ 50 video mới

**Để đạt 100% coverage:** embed lại 50 video trên Kaggle, rồi chạy lại `convert_jina2_to_backend.py`.

---

## Bước 3 — Khởi động server

### Trên Windows (PowerShell)

```powershell
$env:AIC_ENV_FILE = ".env.fpt.local"
$env:PYTHONIOENCODING = "utf-8"
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8002
```

> Dùng port khác (8002) để không conflict với server đang chạy. Server nạp jina-clip-v2 (~5,4 GB) trên CPU — mất **5-10 phút** khởi động.

### Trên VastAI (Linux)

```bash
cd /workspace/AIC2026_Nam_thang_ay

# Kill server cũ
pkill -f "uvicorn online.api.app" || true
sleep 2

# Restart với jina v2 config
AIC_ENV_FILE=.env.fpt.local \
AIC_METADATA_JSONL=storage/exports_competition/scenes.jsonl \
AIC_CAPTION_DENSE_INDEX=storage/caption_embedding_jina_v2 \
AIC_CAPTION_DENSE_ENCODER=jina_clip_v2 \
AIC_CAPTION_DENSE_MODEL=jinaai/jina-clip-v2 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONIOENCODING=utf-8 \
nohup python -m uvicorn online.api.app:app \
  --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
```

---

## Bước 4 — Verify nhánh đã đăng ký

```bash
# Chờ startup (~5 phút trên CPU)
sleep 90

# Kiểm tra branch
curl -s http://127.0.0.1:8000/v1/search/capabilities \
  | python -c "import sys,json; [print(b['branch_id']) for b in json.load(sys.stdin).get('branches',[])]"

# Phải thấy: caption_dense
```

Nếu **không** thấy `caption_dense` trong danh sách → kiểm tra:

```bash
# Kiểm tra lỗi trong log
grep -E "caption_dense|ValueError|ERROR" /workspace/server.log | tail -20
```

---

## Troubleshooting

### `ValueError: encoder kind='jina_clip_v2' không hợp lệ`

**Nguyên nhân:** Code chưa có `JinaClipV2Encoder` (commit cũ).

**Fix:**
```bash
cd /workspace/AIC2026_Nam_thang_ay
git fetch origin
git reset --hard origin/full-runnable
# Verify
grep "ENCODER_KINDS" online/adapters/dense_text.py
# Phải thấy: ('e5', 'jina_v3', 'jina_clip_v2')
```

### `UnicodeEncodeError: 'charmap' codec can't encode character`

**Nguyên nhân:** Console Windows không hỗ trợ Unicode. Chỉ xảy ra khi output ra terminal.

**Fix:** Redirect output sang file:
```powershell
python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8002 > server.log 2>&1
```

Server vẫn chạy bình thường, lỗi chỉ ở console output.

### Coverage warning

```
[WARN] caption_dense coverage 14.9% < 98% (151459 index / 87742 corpus)
```

**Đây là bình thường.** 50 video mới chưa embed → coverage thấp. Server vẫn hoạt động đúng với 823 video đã embed.

---

## Thứ tự git commit

```
c0b3dd3 Add JinaClipV2Encoder + jina v2 dense branch support
a44df30 Add caption v2 prompts + retrieval eval (BM25 + jina v2 dense)
```

Pull bằng:
```bash
git fetch origin
git reset --hard origin/full-runnable
```
