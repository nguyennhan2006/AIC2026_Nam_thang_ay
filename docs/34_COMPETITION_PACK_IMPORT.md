# 34 — Nạp pack thi đấu 873 video vào hệ

Từ `AIC2026_competition_clean_v3.zip` (pack v3, `build_mode=COMPETITION_PARTIAL`)
đến một export mà `uvicorn online.api.app:app` nạp thẳng được.

Đã chạy hết và đo ngày **2026-08-17**. Mọi con số dưới đây là đo trên máy local
(15,6 GB RAM, CPU, không GPU), không phải ước lượng.

---

## 1. Lệnh

```powershell
cd "d:\Sinh viên CNhan\AIC\Data\AIC2026_Nam_thang_ay"
.\.venv\Scripts\Activate.ps1

python -m scripts.import_competition_pack `
    --pack "D:\Sinh viên CNhan\download\AIC2026_competition_clean_v3.zip" `
    --out storage/exports_competition `
    --merge-embeddings-from storage/exports_multivideo
```

Đọc thẳng trong `.zip`, không cần giải nén (pack nở 570 MB → 2,44 GB, mà 636 MB
trong đó là ba cặp file trùng byte-for-byte). Khoảng **6 phút**, ghi ra 620 MB
JSONL + 346 MB vector.

Thử nhanh trước khi chạy thật:

```powershell
python -m scripts.import_competition_pack --pack ... --limit-videos 5 --dry-run
python -m scripts.import_competition_pack --pack ... --batch L23 --out storage/exports_pack_L23
```

`--merge-embeddings-from` **không phải tuỳ chọn cho lượt chạy đầy đủ** — xem §3.

## 2. Pack có gì, thiếu gì

873 video · 101.461 scene · 176.707 keyframe canonical.

| Modality | Coverage | |
|---|---:|---|
| Scene boundary (transnetv2) | 100% | |
| keyframe↔scene mapping | 100% | kèm `fps`, `pts_time` |
| Caption tiếng Việt | 95,31% | 168.414 frame |
| ASR | — | 135.997 đoạn, faster-whisper large-v3 |
| HSV | 100% | |
| Dense vector | 95,62% | 168.960 × `jina_clip_v2` 1024 chiều float16 |
| **OCR** | **0,00%** | 0/176.707 |
| **Ảnh keyframe** | **0** | pack không kèm một file ảnh nào |
| Events / objects / actions | 0 | không có |
| width/height | — | không có ở bất kỳ file nào |

## 3. Lỗ hổng L21 — điều khiến `--merge-embeddings-from` là bắt buộc

| Batch | Video | Frame | Vector |
|---|---:|---:|---:|
| **L21** | 29 | 7.790 | **43 (0,6%)** |
| L22–L30 | 844 | 168.917 | 100% |

Mọi batch khác đủ 100%; riêng L21 gần như trống. Mà **toàn bộ 120 truy vấn gold
(`examples/gold_all3.jsonl`) nằm trên L21_V001/V002/V003** — nhập pack không thôi
thì corpus đầy đủ có đúng một lỗ hổng dense ngay chỗ duy nhất đo được, và không
có gì báo lỗi: nhánh `dense_visual` vẫn `success`, chỉ là ba video ấy không bao
giờ xuất hiện qua đường dense.

Máy này đã có sẵn 855 vector (CLIP + jina) cho đúng ba video đó từ trước.
`--merge-embeddings-from storage/exports_multivideo` bù chúng vào, **chỉ cho
`embedding_name` mà pack không có** — không đảo thứ tự ưu tiên giữa hai nguồn
cùng tên.

## 4. Script quyết định những gì

Đọc `scripts/import_competition_pack.py` — docstring đầu file ghi đủ. Tóm tắt
phần có hệ quả đo được:

| Quyết định | Vì sao | Giá phải trả |
|---|---|---|
| Đúc lại mọi id | `SCENE_ID_PATTERN` đòi `S0000` (4 số), pack dùng `S00000` (5) | id gốc giữ ở `extensions.pack` |
| Bỏ scene không có keyframe | `Scene.keyframes` khai `min_length=1` | 13.719 scene; đo được phần mất thật là **1.948/135.970 đoạn ASR (1,4%)** — phần còn lại vẫn nằm trong scene khác vì một đoạn ASR thường phủ nhiều scene |
| Không ghi caption mức scene | trong pack nó đúng bằng caption của keyframe ĐẦU TIÊN, mà `project_scene()` đã cộng caption mọi keyframe lên scene rồi | không có — tránh được việc đếm hai lần trong BM25 |
| Cắt đoạn ASR theo biên scene | `Scene` từ chối đoạn tràn ra ngoài scene | chỉ mốc thời gian bị cắt, `text` giữ nguyên cả câu |
| Tính lại `end_sec` từ frame | pack ghi `end_sec` của frame CUỐI (bao gồm), contract đòi `ts < end_sec` | 3.713/87.742 scene phải nới biên, **tối đa 0,04 s** (~1,2 frame) |
| `color` để trống | tên màu cần phân bố ĐỒNG THỜI của (hue,sat,val); pack chỉ có ba histogram BIÊN | `color_search` rỗng — muốn có thì phải có ảnh rồi chạy `scripts/backfill_color_quality.py` |
| Vector giữ nguyên bố cục 1 file/video | 168.960 file rời = 168.960 lần mở file mỗi lần khởi động trên NTFS | cần cú pháp `vector_uri` mới, xem §5 |

## 5. Thay đổi trong code online

`online/adapters/frame_vector_store.py::_read_vector_file` giờ hiểu
`<đường dẫn>.npy#<hàng>`, mở bằng `mmap` và cache 4 ma trận gần nhất.

Hai hệ quả đáng ghi:

- **Trả `numpy.ndarray` thay vì `list[float]`.** Cắt RAM trung gian 6 lần:
  24,1 KB → 4,1 KB mỗi vector, tức 4,1 GB → 0,7 GB ở quy mô 168.960 vector.
  Không có thay đổi này thì lượt nạp đầy đủ không chạy nổi trên máy 16 GB.
- **mmap được thả ngay khi dựng xong rows.** Trên Windows, giữ mmap là giữ KHOÁ
  file: server đang chạy sẽ chặn chính việc ghi đè export mà nó đang phục vụ.

Test: `tests/test_frame_vector_store.py` — chọn đúng hàng (kiểm giá trị, không
chỉ kiểm số chiều) và hàng vượt biên phải **ném lỗi**, không im lặng bỏ qua.

## 6. Kết quả lượt chạy đầy đủ

```
  video da ghi               873
  scene da ghi               87742
  scene bo (khong keyframe)  13719
  keyframe da ghi            176707
    co caption               168414
    co OCR                   0
    co vector                169810
      trong do bu tu export  855
    co ANH tren dia          855
  doan ASR da ghi            201456
  scene phai noi bien        3713 (toi da 0.0400s)
  file vector da chep        873 (346.1 MB)
  nguon kich thuoc frame     {'measured': 3, 'assumed': 870}
```

`doan ASR da ghi` (201.456) lớn hơn số đoạn gốc (135.997) là ĐÚNG: một câu nói
vắt qua nhiều scene được chiếu vào từng scene, để scene nào cũng tìm được bằng
câu đó.

`assumed: 870` — 870 video không có ảnh nên `width/height` lấy theo
`--assume-frame-size` (mặc định 1280×720), ghi rõ trong
`extensions.frame_size_source`. Ba video có ảnh thì đo thật.

### Nạp và tìm kiếm

| Bước | Thời gian | RSS |
|---|---:|---:|
| Nạp + validate 87.742 scene | 51,7 s | 0,95 GB |
| Đọc 169.810 vector | 41,4 s | 1,78 GB |
| Dựng vector store (numpy) | 4,6 s | 1,76 GB |
| **Dựng container đầy đủ** | **224 s** | **2,66 GB (đỉnh 5,07)** |
| Sau vài truy vấn | — | 3,94 GB |

Nhánh chạy được: `dense_visual`, `bm25_caption`, `bm25_asr` (+ `bm25_ocr`,
`bm25_keyword` đăng ký nhưng rỗng vì OCR 0%).

Độ trễ đo trên 3 truy vấn KIS:

| Nhánh | thấp nhất | cao nhất |
|---|---:|---:|
| `dense_visual` | 5.164 ms | **11.785 ms** |
| `bm25_caption` | 742 ms | 8.156 ms |
| `bm25_asr` | 286 ms | 1.498 ms |

### ⚠️ `AIC_BRANCH_TIMEOUT_MS=8000` sẽ GIẾT nhánh dense ở corpus này

`.env.fpt.local` đang để 8000 ms — con số chọn cho 765 scene. Ở 87.742 scene,
`dense_visual` mất 5,2–11,8 s, tức **vượt hạn ở một phần truy vấn**. Nhánh vượt
hạn thì `branch_status` báo `timeout` và biến mất khỏi kết quả **trong im lặng**;
API vẫn 200, UI vẫn có kết quả, chỉ là tầng ngữ nghĩa đã tắt ở đúng những truy
vấn nó chậm nhất — thường là truy vấn khó.

Đặt **`AIC_BRANCH_TIMEOUT_MS=30000`** cho corpus đầy đủ.

Phần lớn 5–12 s đó là text tower của jina trên CPU (đo riêng: p50 4,5 s/truy vấn
so với 294 ms của CLIP), không phải phép nhân ma trận. Pack chỉ có vector jina
nên không đổi sang CLIP được nếu chưa embed lại.

## 7. Chạy server trên corpus đầy đủ

```powershell
cd "d:\Sinh viên CNhan\AIC\Data\AIC2026_Nam_thang_ay"
.\.venv\Scripts\Activate.ps1

$env:AIC_ENV_FILE               = ".env.fpt.local"
$env:AIC_METADATA_JSONL         = "storage/exports_competition/scenes.jsonl"
$env:AIC_VISUAL_EMBEDDING_NAME  = "jina_clip_v2"
$env:AIC_VISUAL_EMBEDDING_MODEL = "storage/models/jina-clip-v2"
$env:AIC_ENABLE_QUERY_TRANSLATION = "false"
$env:AIC_BRANCH_TIMEOUT_MS      = "30000"
$env:HF_HUB_OFFLINE             = "1"
$env:TRANSFORMERS_OFFLINE       = "1"

python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

`ENABLE_QUERY_TRANSLATION=false` vì jina có text tower đa ngữ — dịch cho nó là
mất thông tin, xem `docs/20` § VISUAL-01. Cũng nhờ vậy đường truy vấn không cần
mạng và không tốn một lời gọi FPT nào.

`HF_HUB_OFFLINE=1` là bắt buộc trên máy này: `storage/models/jina-clip-v2` vẫn
cố hỏi HuggingFace về tokenizer của `jina-embeddings-v3`, mà máy này chặn SSL tới
`huggingface.co` — không đặt thì container chết ngay lúc dựng.

Khởi động mất **~4 phút**. Đợi `Application startup complete.`

Kiểm ngay:

```powershell
curl.exe -s localhost:8000/health | python -m json.tool
curl.exe -s localhost:8000/search/capabilities | python -m json.tool
```

Kỳ vọng `scene_count` **87742**, `video_count` **873**, `keyframe_count`
**176707**, và `dense_visual` với `backend_kind: "vector"`.

## 8. Còn thiếu gì — theo thứ tự thiệt hại

1. **Ảnh keyframe (176.707 file).** Chưa có thì UI không hiện gì, `/media` 404,
   VLM rerank không dùng được, và **không sinh lại được embedding hay OCR** —
   mọi cách vá hai mục dưới đều bắt đầu từ ảnh. Đặt vào
   `storage/processed/keyframes/<video_id>/frame_%06d.jpg`; `image_path` trong
   export đã trỏ sẵn tới đó nên không phải convert lại.
2. **OCR — 0% trên cả 873 video.** Cấu hình hiện tại để `bm25_ocr` weight **1.0**
   vì đo được đó là nhánh duy nhất tìm ra tên người trên chyron, và **75/120 truy
   vấn gold khai `required_modalities` có ocr**. Đo trên corpus này mà không có
   OCR thì số thu được không so được với bất kỳ số nào trong `docs/20`.
3. **`AIC2026_missing_v3.zip`** — README của pack trỏ tới nó cho danh sách 21.275
   orphan + 1.403 conflict. Chưa tải.
4. **Events** — `event_search` chết, và dedup theo `event_id` của AVS không chạy
   (mà `AIC_AVS_MAX_RESULTS_PER_VIDEO=40` được tune dựa trên chính việc dedup ấy
   hoạt động).
5. **Objects / action tags** — `object_search`, `action_search` rỗng.
6. **Video gốc** — cần cho xem lại và TRAKE frame refinement.

## 9. Việc chưa làm

**Chưa đo chất lượng trên corpus đầy đủ.** §6 chứng minh hệ CHẠY, không chứng
minh nó tìm ĐÚNG. Bộ gold vẫn nằm trên L21_V001..V003, giờ có thêm 870 video
nhiễu — đó là phép đo đáng giá nhất còn lại và chưa ai chạy:

```powershell
python -m scripts.eval_tasks --pipeline container `
    --gold examples/gold_all3.jsonl `
    --metadata storage/exports_competition/scenes.jsonl
```

Nhớ `--pipeline container`. Không có cờ đó, harness dựng một pipeline thứ hai
thiếu vài nhánh — số in ra trông hợp lệ nhưng không phải số của server.

Lưu ý trước khi chạy: 120 truy vấn × ~6–12 s cộng 4 phút dựng container, và
nhánh QA vẫn gọi FPT (tốn quota).
