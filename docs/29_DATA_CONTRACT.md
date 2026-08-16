# 29 — Hợp đồng dữ liệu offline/enrich → online

Tài liệu tham chiếu: **định dạng vào/ra của mọi giai đoạn offline và enrichment**,
**cấu trúc thư mục bắt buộc**, và **trường nào online thực sự đọc**.

Mọi khẳng định ở đây đọc ra từ code ngày 2026-08-07, không phải từ thiết kế mong
muốn. Chỗ nào code và tài liệu cũ lệch nhau, code thắng. Số liệu độ phủ đo trên
`storage/exports_multivideo/` (765 scene, 855 keyframe, 3 video).

Ba mức đánh dấu dùng xuyên suốt:

| Dấu | Nghĩa |
|---|---|
| **BẮT BUỘC** | Thiếu là online không khởi động được, hoặc một nhánh chết |
| **KHÔNG DÙNG** | Có trong contract/đang sinh ra, nhưng **không một dòng code online nào đọc** |
| **RỖNG** | Có đường đọc, nhưng dữ liệu hiện tại trống → tính năng im lặng vô hiệu |

### ⚠️ Chữ "clip" có BỐN nghĩa — đọc kỹ trước khi kết luận

Đây là nguồn nhầm lẫn lớn nhất trong repo. Hai cái đang chạy, hai cái không:

| Thứ | Là gì | Trạng thái |
|---|---|---|
| **Model CLIP** | `openai/clip-vit-large-patch14` | **ĐANG CHẠY** — chính là nhánh `dense_visual`, nhánh mạnh nhất. 855 vector ảnh ở `processed/embeddings/`. UI hiện "Visual (CLIP)" |
| **Phát đoạn video** | cửa sổ `start_sec`–`end_sec` trên video gốc | **ĐANG CHẠY** — [playback.py](../online/services/playback.py), đọc `raw/videos/*.mp4`. **Không liên quan `clips.jsonl`** |
| **`ClipSegment` / `clips.jsonl`** | cửa sổ thời gian có vector pooled, do `offline/clip_pooling.py` sinh | **KHÔNG DÙNG** — xem §4.5 |
| **`processed/clip_embeddings/`** | 426 vector pooled của các `ClipSegment` | **KHÔNG DÙNG** — chỉ `clip_pooling.py` ghi, không ai đọc; và chỉ có V001 |

Khi tài liệu này viết **KHÔNG DÙNG** cho "clip", nó **luôn** nói về hai dòng cuối
— `ClipSegment`/`clips.jsonl`/`clip_embeddings`. Model CLIP thì ngược lại: bỏ nó
đi là mất nhánh retrieval quan trọng nhất.

---

## 1. Bản đồ một trang

```
     notebook / script enrich                offline assemble              online
┌──────────────────────────────┐      ┌────────────────────────┐   ┌──────────────────┐
│ storage/packs/<stage>/       │      │ storage/exports_<x>/   │   │ 8 nhánh retrieval│
│   _SUCCESS.json              │─────▶│   scenes.jsonl      ◀──┼───│ AIC_METADATA_JSONL
│   model_info.json            │      │   keyframes.jsonl   ◀──┼───│ vector file paths│
│   manifests/*.jsonl          │      │   videos.jsonl      ◀──┼───│ source_path      │
│                              │      │   events.jsonl      ◀──┼───│ (chỉ khi bật)    │
│ 9 stage, join theo           │      │   clips.jsonl (ko dùng)│   │ KHÔNG DÙNG       │
│ (video_id, frame_idx)        │      │   dataset_manifest.json│   │ build_id         │
└──────────────────────────────┘      └────────────────────────┘   └──────────────────┘   
                                                 │
     storage/processed/                          │  image_path, vector_uri
       keyframes/<vid>/frame_%06d.jpg   ◀────────┤  (tương đối AIC_DATA_ROOT)
       embeddings/<vid>/frame_%06d.json ◀────────┘
     storage/raw/videos/<vid>.mp4       ◀──── source_path
```

**Bất biến quan trọng nhất:** `AIC_METADATA_JSONL` trỏ tới `scenes.jsonl`, và bốn
file còn lại được tìm bằng `path.with_name(...)` — tức **cả năm file phải nằm
cùng một thư mục**. Trỏ `scenes.jsonl` vào một thư mục còn `keyframes.jsonl` ở
thư mục khác thì `dense_visual` im lặng rỗng.

---

## 2. Cấu trúc thư mục chuẩn

```
storage/                                    ← AIC_DATA_ROOT
├── raw/
│   └── videos/<video_id>.mp4               BẮT BUỘC nếu muốn phát video
├── processed/
│   ├── keyframes/<video_id>/frame_%06d.jpg BẮT BUỘC (ảnh cho UI + VLM rerank)
│   ├── embeddings/<video_id>/frame_%06d.json  BẮT BUỘC cho dense_visual
│   │                                       (vector CLIP mức FRAME — cái đang chạy)
│   └── clip_embeddings/<video_id>/*.json   KHÔNG DÙNG (vector ClipSegment pooled)
├── models/
│   └── clip-vit-large-patch14/             BẮT BUỘC (máy này không tải được HF)
├── packs/                                  đầu vào của assemble
│   ├── video/     ├── scene/    ├── keyframe/  ← BẮT BUỘC (REQUIRED_STAGES)
│   ├── asr/       ├── caption/  ├── ocr/
│   ├── object/    ├── color/    └── embedding/ ← tuỳ chọn
│   └── (mỗi pack: _SUCCESS.json + model_info.json + manifests/)
├── exports_<tên>/                          đầu ra của assemble
│   ├── scenes.jsonl                        ← AIC_METADATA_JSONL
│   ├── keyframes.jsonl
│   ├── videos.jsonl
│   ├── events.jsonl
│   ├── clips.jsonl                         KHÔNG DÙNG (ClipSegment, ko phải model CLIP)
│   ├── dataset_manifest.json
│   ├── assemble_report.json                chỉ để người đọc
│   └── quarantine.jsonl                    chỉ để người đọc
├── cache/
│   ├── fpt_enrich/                         cache VLM theo sha256(ảnh+prompt+model)
│   └── query_translation/                  cache dịch VI→EN
└── indexes/                                KHÔNG DÙNG (xem §7)
```

`processed/` và `raw/videos/` là **hai gốc media công khai duy nhất** —
[routes.py:542](../online/api/routes.py#L542) từ chối mọi đường dẫn khác bằng 403.
Đuôi file cho phép: `.jpg .jpeg .png .webp .svg .mp4 .webm .mkv`.

---

## 3. Tầng 1 — Stage pack (đầu vào của `offline assemble`)

Mỗi giai đoạn trích xuất ghi **một pack tự chứa**. Không notebook nào ghi đè
output của notebook khác, và chạy lại lẻ từng stage được.

### 3.1 Khung một pack

```
<pack_root>/
├── _SUCCESS.json      BẮT BUỘC  {"status": "success", ...}   status != success -> assemble từ chối
├── model_info.json    BẮT BUỘC  {"component","model","version","pack_version","device"}
└── manifests/
    └── <stage>_manifest.jsonl   BẮT BUỘC  một dòng một record
```

Tên file manifest theo `STAGE_MANIFESTS` ([stagepack.py:45](../offline/stagepack.py#L45)):

| stage | file manifest | mức |
|---|---|---|
| `video` | `video_manifest.jsonl` | video |
| `scene` | `scene_manifest.jsonl` | scene |
| `keyframe` | `keyframe_manifest.jsonl` | keyframe |
| `asr` | `asr_segments.jsonl` | video (cắt theo scene lúc assemble) |
| `caption` | `caption_manifest.jsonl` | keyframe |
| `ocr` | `ocr_manifest.jsonl` | keyframe |
| `object` | `object_manifest.jsonl` | keyframe |
| `color` | `color_manifest.jsonl` | keyframe |
| `embedding` | `embedding_manifest.jsonl` | keyframe |

**`video`, `scene`, `keyframe` là BẮT BUỘC.** Thiếu một trong ba thì `assemble`
ném `StagePackError` ngay. Sáu stage còn lại vắng mặt chỉ sinh warning và nhánh
tương ứng im lặng.

### 3.2 Khóa join — quy ước không được phá

```
stage mức video     -> video_id
stage mức scene     -> (video_id, scene_index)
stage mức keyframe  -> (video_id, frame_idx)
```

Stage mức keyframe **KHÔNG được tự đặt `keyframe_id`**. Id canonical nhúng
`scene_idx` bên trong (`{video}_S{scene:04d}_F{frame:06d}`) mà notebook trích
OCR/caption không biết và không nên biết. `offline/assemble.py` là nơi duy nhất
dựng nó.

Trùng `(video_id, frame_idx)` trong cùng một pack → record thứ hai vào
`quarantine.jsonl`, không ghi đè.

### 3.3 Trường bắt buộc từng loại row

Nguồn: [contracts/stage_pack.schema.json](../contracts/stage_pack.schema.json) +
[verify_stage_pack.py:24](../scripts/verify_stage_pack.py#L24).

**`videoRow`**
```json
{"video_id":"L21_V001","source_path":"raw/videos/L21_V001.mp4","fps":30.0,
 "frame_count":37849,"width":1280,"height":720,
 "duration_sec":1261.7,"codec":"h264","audio_present":true,"source_checksum":null}
```
BẮT BUỘC: `video_id, source_path, fps, frame_count, width, height`.
`source_path` **tương đối `AIC_DATA_ROOT`** — đây là đường dẫn phát video.

**`sceneRow`**
```json
{"video_id":"L21_V001","scene_index":0,"start_frame":0,"end_frame":564,
 "detector":"transnetv2","confidence":0.91}
```
BẮT BUỘC: `video_id, scene_index, start_frame, end_frame`.
⚠️ **`end_frame` là INCLUSIVE.** Assemble chuyển sang `end_frame_exclusive = end_frame + 1`.
`scene_id` của notebook (nếu có) bị **bỏ qua** — assemble đánh số lại `0..N-1`
theo thứ tự thời gian, định dạng 4 chữ số.

**`keyframeRow`**
```json
{"video_id":"L21_V001","frame_idx":9987,
 "image_path":"processed/keyframes/L21_V001/frame_009987.jpg",
 "width":1280,"height":720,
 "roles":["representative"],"selection_score":0.82,
 "quality":{"sharpness":142.0,"brightness":0.51,"contrast":0.33,
            "black_frame_ratio":0.0,"duplicate_score":0.1},
 "timestamp_sec":332.9, "source_checksum":null}
```
BẮT BUỘC: `video_id, frame_idx, image_path, width, height`.
⚠️ **`timestamp_sec` bị BỎ QUA** — assemble luôn tính lại `frame_idx / fps`. Lý do
ghi ở [assemble.py:292](../offline/assemble.py#L292): `pts_time` thật từ ffmpeg
có thể lệch `frame_idx/fps` một frame và làm validator scene false-fail.

**`asrRow`**
```json
{"video_id":"L21_V001","start_sec":314.2,"end_sec":328.4,
 "text":"...Ghi nhận sau tại Đồng Tháp.",
 "language":"vi","confidence":0.93,"asr_segment_id":"L21_V001_A000012"}
```
BẮT BUỘC: `video_id, start_sec, end_sec, text`.
Assemble **cắt theo biên scene** (giao khoảng) và đánh lại id — một segment dài
nằm vắt hai scene sẽ xuất hiện ở cả hai, mỗi bên một phần.

**`captionRow`**
```json
{"video_id":"L21_V001","frame_idx":9987,
 "captions":[{"text":"Một người đàn ông đang cầm một tờ báo...",
              "caption_type":"detailed","language":"vi","confidence":null}]}
```
BẮT BUỘC: `video_id, frame_idx, captions`. Item không có `text` bị loại.

**`ocrRow`**
```json
{"video_id":"L21_V001","frame_idx":9987,
 "instances":[{"text":"GIÁ CÁ TRA THẤP, NGƯỜI NUÔI LO LẮNG",
               "bbox":{"x1":0.17,"y1":0.27,"x2":0.73,"y2":0.74},
               "confidence":0.0,"language":"vi","normalized_text":null}]}
```
BẮT BUỘC: `video_id, frame_idx, instances`. Mỗi instance bắt buộc `text` + `bbox`.
`bbox` **chuẩn hoá 0–1**, và bbox không hợp lệ (`x2<=x1` hoặc `y2<=y1`) bị **loại
thẳng**, không "sửa" theo suy đoán.

**`objectRow`**
```json
{"video_id":"L21_V001","frame_idx":9987,
 "objects":[{"label":"thành phố","confidence":0.8,
             "bbox":{"x1":0.0,"y1":0.33,"x2":1.0,"y2":0.68},"attributes":{}}]}
```
BẮT BUỘC: `video_id, frame_idx, objects`. Mỗi object bắt buộc `label` + `bbox`.

**`colorRow`** — hiện **RỖNG hoàn toàn**, xem §6
```json
{"video_id":"L21_V001","frame_idx":9987,
 "dominant_colors":[{"name":"xanh lá","ratio":0.42}],
 "dominant_hex":["#3a7d2c"],"mean_hsv":[0.31,0.55,0.48],
 "hsv_histogram":[...],"regions":{}}
```
BẮT BUỘC: `video_id, frame_idx`. Online **chỉ đọc `dominant_colors[].name`**.

**`embeddingRow`**
```json
{"video_id":"L21_V001","frame_idx":9987,
 "vector":[0.013, -0.072, ...],
 "embedding_refs":[{"embedding_name":"clip_vit_l14_v1","dimension":768,
    "modality":"image","model_name":"openai/clip-vit-large-patch14","normalized":true,
    "storage_locations":[{"backend":"file","index_name":"clip_vit_l14_v1",
       "vector_id":"L21_V001_F009987",
       "vector_uri":"processed/embeddings/L21_V001/frame_009987.json"}]}]}
```
BẮT BUỘC: `video_id, frame_idx`. Ngoài ra phải có **ít nhất một** trong:
`vector` (inline), `vector_path`, hoặc `embedding_refs`.

Hai đường dùng khác nhau, cần **cả hai**:
- `vector` inline → `assemble` pool clip segment
- `embedding_refs[].storage_locations[].vector_uri` → **online dựng vector store**

`vector_uri` **tương đối `AIC_DATA_ROOT`**. Đuôi `.npy` hoặc `.json` đều đọc được.

### 3.4 Kiểm tra pack trước khi assemble

```powershell
python -m scripts.verify_stage_pack storage/packs --all
python -m scripts.verify_stage_pack storage/packs/scene --stage scene
```

Bắt được: `_SUCCESS.json` sai status, thiếu `model_info.json`, thiếu trường bắt
buộc, `(video_id, frame_idx)` trùng, `end_frame < start_frame`, scene không liền mạch.

---

## 4. Tầng 2 — Canonical export (đầu vào của online)

`offline assemble` ghi **5 jsonl + 1 manifest** qua
[datasection/exporter.py:53](../datasection/exporter.py#L53).

### 4.1 `scenes.jsonl` — file trung tâm

Đây là file `AIC_METADATA_JSONL` trỏ vào. Một dòng = một `Scene`, keyframe **lồng
bên trong**.

BẮT BUỘC theo pydantic: `scene_id, video_id, scene_idx, start_frame,
end_frame_exclusive, start_sec, end_sec, segmentation_provenance, keyframes`
(≥1 keyframe — scene không có keyframe nào bị quarantine).

Trường online **đọc thật**, theo
[json_metadata.py:69](../online/adapters/json_metadata.py#L69):

| Trường scene | Dùng để |
|---|---|
| `scene_id` | khóa chính |
| `video_id` | gom nhóm, dedup, TRAKE khoá video |
| `scene_idx` | **ràng buộc tăng dần của chuỗi TRAKE** |
| `start_frame` / `end_frame_exclusive` | khoảng frame, `boundary_distance_frames` |
| `start_sec` / `end_sec` | cửa sổ phát video, suy fps |
| `extensions.event_id` | dedup theo event, metric AVS |
| `schema_version` | → `artifact_version` trong trace |
| `keyframes[]` | xem bảng dưới |
| `captions[].text` | nhánh `bm25_caption` |
| `asr_segments[].text` | nhánh `bm25_asr` |
| `keywords[].normalized_text` (fallback `.text`) | nhánh `bm25_keyword` |
| `action_tags[]` | nhánh `bm25_action` |

**KHÔNG DÙNG** ở mức scene: `transition_in`, `transition_out`,
`boundary_confidence_in`, `boundary_confidence_out`, `scene_clip_path`,
`scene_clip_checksum`, `embedding_refs`, `segmentation_provenance`, `created_at`,
và mọi khối `provenance` lồng bên trong caption/ocr/object/asr.

Trường keyframe online **đọc thật**, theo
[json_metadata.py:18](../online/adapters/json_metadata.py#L18):

| Trường keyframe | Dùng để |
|---|---|
| `keyframe_id` | id vector store |
| `video_id`, `scene_id` | liên kết ngược |
| `frame_idx` | **giá trị nộp bài** |
| `timestamp_sec` | phát video, hiển thị |
| `image_path` | thumbnail UI, ảnh cho VLM rerank |
| `selection_score` | safe-frame (KIS) — **RỖNG** |
| `quality.{sharpness,brightness,contrast,black_frame_ratio,duplicate_score}` | safe-frame — **RỖNG** |
| `captions[].text` | `bm25_caption` |
| `ocr_instances[].text` | `bm25_ocr`, `ocr_fuzzy` |
| `objects[].label` | `bm25_object` |
| `action_tags[]` | `bm25_action` |
| `color.dominant_colors[].name` | `color_search` — **RỖNG** |
| `embedding_refs[].embedding_name` | chỉ để biết "frame này CÓ embedding" |

**KHÔNG DÙNG** ở mức keyframe — đây là danh sách cần nhớ khi thiết kế enrichment
để khỏi tốn công/tốn tiền sinh ra thứ không ai đọc:

- `width`, `height` — bắt buộc theo contract nhưng online không đọc
- `roles`, `source_checksum`, `created_at`, `extensions`
- `ocr_instances[].bbox` / `.confidence` / `.language` / `.normalized_text` — **chỉ `.text` được đọc**
- `objects[].bbox` / `.confidence` / `.attributes` — **chỉ `.label` được đọc**
- `captions[].caption_type` / `.language` / `.confidence` / `.crop_bbox` — **chỉ `.text`**
- `color.dominant_hex` / `.mean_hsv` / `.hsv_histogram` / `.regions` / `.histogram_uri`
- `embedding_refs[].dimension` / `.modality` / `.model_name` / `.normalized`
- mọi `provenance`

> **Lưu ý về bbox OCR.** Không đọc trong đường xếp hạng, nhưng bộ lọc lớp phủ
> (`AIC_OCR_OVERLAY_DF`) dùng vị trí `y` để nhận thanh chữ chạy. Nếu bỏ bbox thì
> mất khả năng đó. Giữ bbox.

### 4.2 `keyframes.jsonl`

Bản phẳng của toàn bộ keyframe. Online đọc **đúng một thứ**:
`embedding_refs[].storage_locations[]` với `backend == "file"` → `vector_uri`
([frame_vector_store.py:65](../online/adapters/frame_vector_store.py#L65)).

Phần còn lại của file là bản sao của dữ liệu đã có trong `scenes.jsonl`.
File vẫn **BẮT BUỘC** phải tồn tại cạnh `scenes.jsonl`, nếu không `dense_visual`
không dựng được.

### 4.3 `videos.jsonl`

Online đọc **đúng ba trường**
([json_metadata.py:140](../online/adapters/json_metadata.py#L140)):

| Trường | Dùng để |
|---|---|
| `video_id` | khóa |
| `source_path` | đường dẫn phát video (`/v1/media/<source_path>`) |
| `frame_count` | validator submission: "frame có thuộc video không" |

**KHÔNG DÙNG:** `fps`, `width`, `height`, `codec`, `duration_sec`,
`audio_present`, `source_checksum`, `probe_provenance`, `extensions`, và — đáng
kể nhất — **`scenes[]`, `clips[]`, `events[]` lồng bên trong**. Ba mảng lồng này
là bản sao đầy đủ của ba file kia và chiếm gần hết 1.9 MB của file.

> Mở rộng lên 876 video, riêng phần lồng thừa này sẽ là ~550 MB đọc rồi vứt lúc
> khởi động. Chỉ cần đọc streaming 3 trường thay vì `json.loads` cả dòng.

### 4.4 `events.jsonl`

Đọc trong hai trường hợp:
- nhánh `event_search` (hiện **TẮT** — xem §8)
- hai route duyệt event `/v1/events/{id}` và neighbours

Ngoài ra `extensions.event_id` trong `scenes.jsonl` là **bản sao đã nướng sẵn**
của quan hệ này, để dedup theo event không phải nạp `events.jsonl` mỗi request.

⚠️ **Hiện chỉ phủ L21_V001.** 69 event / 217 scene, V002 và V003 **không có event
nào**. Hệ quả: dedup theo event và `event_coverage` của AVS chỉ hoạt động trên
1/3 corpus.

### 4.5 `clips.jsonl` (`ClipSegment`) — **KHÔNG DÙNG**

> Đây là `ClipSegment` — cửa sổ thời gian có vector pooled. **Không phải model
> CLIP** (đang chạy, là nhánh `dense_visual`), cũng **không phải chức năng phát
> đoạn video** trong UI (đang chạy, dùng `raw/videos/*.mp4`). Xem bảng bốn nghĩa ở đầu tài liệu.

Exporter ghi ra, `dataset_manifest.json` ghi checksum của nó, nhưng **không một
dòng code online nào đọc**. `Candidate.clip_id` tồn tại nhưng không bao giờ được
gán giá trị thật — [fusion.py:125](../online/services/fusion.py#L125) chỉ chép lại
`None`.

Grep toàn repo, `clips.jsonl` chỉ xuất hiện ở `datasection/exporter.py` (ghi +
`verify_export`), `offline/clip_pooling.py` (tạo), `scripts/export_schemas.py`
(sinh JSON Schema) và hai test. Không có `online/`.

Kèm theo nó, `storage/processed/clip_embeddings/` chứa 426 file vector pooled
(chỉ L21_V001) — cũng không ai đọc.

Trong `storage/exports_multivideo/` **file này còn đang thiếu hẳn** (manifest khai
`clip_count: 426` và có checksum, file không tồn tại). Không gây lỗi vì online
không đọc. Clip pooling cũng là phần duy nhất cần `vector` inline trong
embedding pack.

⚠️ Export này **không qua nổi `verify_export()`**, và lý do đầu tiên còn nặng hơn
clip: `checksum mismatch: videos.jsonl`. Tức `videos.jsonl` đã bị sửa sau khi
build mà manifest không được ghi lại. Xem thêm §4.6.

### 4.6 `dataset_manifest.json`

Online đọc **đúng một trường**: `build_id` → `dataset_version` trong session trace
và `/v1/health`. Phần còn lại (`models[]`, `export_checksums`, các `*_count`) chỉ
để người đọc.

⚠️ File này **không tự cập nhật** khi bạn sửa export bằng script rời. `/v1/health`
sẽ báo `dataset_version` cũ mà không có cảnh báo nào.

Export hiện tại đã lệch thật — `verify_export()` báo `checksum mismatch:
videos.jsonl`. Nghĩa là một script nào đó đã ghi lại `videos.jsonl` sau khi
assemble mà không cập nhật manifest. Chưa gây hại vì online chỉ đọc 3 trường,
nhưng nó làm mất khả năng dùng checksum để biết export có toàn vẹn hay không.
Chạy lại `offline assemble` là cách sửa sạch nhất.

---

## 5. Bản đồ nhánh → trường

Tám nhánh đang chạy với `.env.fpt.local` hiện tại:

| Nhánh | Nguồn dữ liệu | Trạng thái |
|---|---|---|
| `dense_visual` | file vector qua `keyframes.jsonl → embedding_refs → vector_uri` | 855/855 |
| `bm25_caption` | `scene.captions[].text` + `keyframe.captions[].text` | 100% |
| `bm25_ocr` | `keyframe.ocr_instances[].text` | 86% |
| `bm25_asr` | `scene.asr_segments[].text` | 98% |
| `bm25_keyword` | `scene.keywords[]` (assemble sinh từ `objects[].label`) | 99% |
| `bm25_object` | `keyframe.objects[].label` | 99% |
| `bm25_action` | `action_tags[]` (assemble sinh từ caption) | 57% |
| `color_search` | `keyframe.color.dominant_colors[].name` | **0% — RỖNG** |

Tắt: `ocr_fuzzy`, `event_search` (đo được là **có hại**, xem §8).

`dense_visual` cần thêm hai thứ ngoài dữ liệu:
- `storage/models/clip-vit-large-patch14/` — text tower **phải cùng model** với
  vector ảnh, khác model thì cosine vô nghĩa
- Nạp model hỏng thì container **chặn khởi động**, không chỉ cảnh báo

---

## 6. Độ phủ thực tế — cái gì đang RỖNG

Đo trên `storage/exports_multivideo/`. Cập nhật **2026-08-08** sau đợt bù dữ liệu —
cột "trước" là trạng thái cũ để thấy cái gì vừa đổi.

### Mức keyframe (855)

| Trường | Trước | Nay | Ghi chú |
|---|---:|---:|---|
| `captions[]` | 100.0% | 100.0% | |
| `embedding_refs[]` | 100.0% | 100.0% | |
| `objects[]` | 99.5% | 99.5% | |
| `ocr_instances[]` | 86.1% | 86.1% | +49 keyframe V001 có thêm chyron |
| `action_tags[]` | 58.1% | 58.1% | suy từ caption, không phải model |
| `color` | 0.0% | **100.0%** | `scripts/backfill_color_quality.py` |
| `quality.*` | 0.0% | **100.0%** | cùng script |
| `selection_score` | 0.0% | 0.0% | **cố ý để trống**, xem dưới |
| `source_checksum` | 0.0% | 0.0% | KHÔNG DÙNG, không sao |

### Mức scene (765)

| Trường | Trước | Nay | Ghi chú |
|---|---:|---:|---|
| `transition_in` | 100.0% | 100.0% | KHÔNG DÙNG |
| `captions[]` | 99.9% | 99.9% | |
| `keywords[]` | 99.5% | 99.5% | |
| `asr_segments[]` | 97.9% | 97.9% | |
| `extensions.event_id` | 28.4% *(chỉ V001)* | **100.0%** | `scripts/backfill_events.py` |
| `action_tags[]` | 56.9% | 56.9% | |
| `scene_clip_path` | 0.0% | 0.0% | KHÔNG DÙNG |
| `embedding_refs[]` | 0.0% | 0.0% | KHÔNG DÙNG (mức scene) |

### Vì sao `selection_score` vẫn trống — có chủ ý

Khác `quality` (phép đo vật lý khách quan trên ảnh), `selection_score` là phán
đoán "frame này đại diện tốt đến đâu", và `safe_frame._quality` cộng thẳng nó vào
điểm chọn frame. Bịa một công thức ở đây là nhét một tín hiệu xếp hạng chưa từng
được đo vào đường chấm điểm. Muốn có thì phải thiết kế rồi đo riêng.

### Hai phát hiện lộ ra khi lấp

**Thang sharpness của `safe_frame` không khớp corpus.** Đo trên 855 keyframe:

```
sharpness thuc te   p10=241   p50=488   p90=824
safe_frame chuan hoa trong [40, 300]  ->  742/855 (87%) VUOT TRAN, bi kep ve 1.0
```

Tín hiệu sharpness vừa bật đã mất phần lớn khả năng phân biệt.
`SHARPNESS_CEILING` ở [safe_frame.py:33](../online/services/safe_frame.py#L33) được
chọn trước khi có dữ liệu thật. Chỉnh lại là đổi scoring nên phải đo holdout.

**Gom event không có vùng giữa.** `group_scenes_into_events` chỉ có hai chế độ
trên corpus này, không tinh chỉnh được thành "mỗi tin một event":

```
min_text_overlap = 0       ->   9-24 event, p50 55-118s   (gop nhieu tin lam mot)
min_text_overlap >= 0.02   ->  94-194 event, p50 4-7s     (gan bang scene)
```

Cả ba video đều không có khoảng trống scene nào >2s nên `max_gap_sec` chưa bao giờ
tách được gì. Caption VLM của các shot liền nhau khác nhau quá nhiều nên mọi ngưỡng
>0 đều cắt gần hết. Đã chốt `min_text_overlap=0.02` cho **cả ba** video — ưu tiên
tính so sánh được giữa các video hơn là tái tạo con số cũ của V001 (69 event, không
tái tạo được vì caption đã bị enrich lại sau lần assemble đó).

### Phần CÒN CHẶN — không phải vì chưa làm, mà vì thiếu đầu vào

| Thiếu | V001 | V002 | V003 | Chặn cái gì |
|---|:-:|:-:|:-:|---|
| `raw/videos/<id>.mp4` | có | **KHÔNG** | **KHÔNG** | phát video trong UI; mọi thứ cần đọc video gốc |
| OCR chyron | 49 keyframe | **0** | **0** | tên/chức vụ người được phỏng vấn không tra được |

`storage/raw/videos/` chỉ có `L21_V001.mp4` (và `L16_V001.mp4` của bộ cũ). Hai
video còn lại chỉ được cấp **ảnh keyframe + scene manifest + ASR**, không có file
video. Hệ quả cụ thể:

- [playback.py:69](../online/services/playback.py#L69) kiểm tra file tồn tại và trả
  `None` — UI hiện "chưa có video" thay vì một player 404. Đúng hành vi, nhưng
  người chấm không xem được đoạn phim. **Đã hết hiệu lực từ 10/08** — cả ba video
  đã có mặt trong `storage/raw/videos/`.
- `scripts/chyron_backfill.py` bỏ qua hai video này kèm cảnh báo.

Không có cách nào vá bằng code. Cần chính file `L21_V002.mp4` và `L21_V003.mp4`
đặt vào `storage/raw/videos/`, rồi chạy lại đúng một lệnh chyron backfill.

### Kết quả đo sau đợt bù — và một hồi quy phải xử lý

Đo 120 truy vấn gold, `--pipeline container`, VLM rerank tắt. Cột trước là
`outputs/evaluation/quick/novlm.json` (06/08).

| | trước | sau | |
|---|---:|---:|---|
| KIS R@5 | 0.917 | **0.944** | ++ |
| KIS MRR | 0.725 | **0.731** | ++ |
| KIS R@1 / R@20 | 0.583 / 1.000 | 0.583 / 1.000 | = |
| QA R@20 | 0.861 | **0.889** | ++ |
| QA answer_accuracy | 0.583 | **0.611** | ++ |
| QA joint_top1 | 0.389 | **0.417** | ++ |
| AVS nDCG@100 | 0.598 | 0.558 | −0.040 |
| AVS event_coverage | 0.841 | 0.793 | −0.048 |

**TRAKE không tính vào đây.** Con số sau (`video_recall@1` 0.833, `@3` 1.000,
`mean_r` 0.254) trùng KHỚP TỪNG SỐ với bảng trong
[from_sequences.py](../online/services/trake/from_sequences.py) — tức toàn bộ
cải thiện TRAKE là do đổi engine (commit `ac2a260`), đợt bù dữ liệu đóng góp
**đúng 0** cho TRAKE. Baseline `novlm.json` có trước lần đổi đó nên nếu so thẳng
sẽ thấy `+0.292` và quy nhầm công.

**Hồi quy đã tìm ra thủ phạm và xử lý: nhánh `bm25_ocr`.** Sau khi sửa lỗi
`_boxes` (§7.4), nhánh này sống lại thật — và đo được là **gây hại**:

| | KIS R@1 | KIS MRR | QA R@1 | QA ans_acc |
|---|---:|---:|---:|---:|
| `bm25_ocr` bật | 0.500 | 0.680 | 0.444 | 0.583 |
| `bm25_ocr` tắt | **0.583** | **0.738** | **0.583** | **0.611** |

Siết bộ lọc (`AIC_OCR_OVERLAY_MAX_WORDS=6`) chỉ lấy lại một phần (KIS R@1 0.556,
MRR 0.709). Đã chốt `AIC_ENABLE_OCR_BRANCH=false`.

Tắt nhánh **không** xoá dữ liệu OCR — `ocr_texts` vẫn vào evidence pack mà QA đọc
để trả lời, và đó chính là lý do `answer_accuracy` TĂNG khi tắt: bớt nhiễu ở tầng
xếp hạng mà vẫn giữ chữ ở tầng trả lời.

⚠️ Cờ này điều khiển việc **đăng ký** nhánh, không phải trọng số. Nhánh chưa đăng
ký thì `search_options.branches.bm25_ocr.weight` không bật lại được. Muốn tra
tên/số trên màn hình phải đặt `=true` **và khởi động lại** server.

**`AIC_AVS_MAX_RESULTS_PER_VIDEO` 20 → 40.** Trần 20 được chọn khi chỉ V001 có
event; nay cả ba video đều có nên dedup chạy thật và 20 slot là 20 *event* chứ
không còn là 20 *scene*:

```
cap=20   nDCG 0.354   event_cov 0.376
cap=40   nDCG 0.547   event_cov 0.793   <- bão hoà từ đây
cap=100  nDCG 0.547   event_cov 0.793
```

Phần hụt còn lại so với trước (0.598 / 0.841) là **giá thật** của việc bật dedup
ở 2/3 corpus: bớt ảnh gần trùng nên ít lần "trúng" gold hơn, đổi lấy kết quả đa
dạng hơn.

---

## 7. Danh sách KHÔNG DÙNG — đầy đủ

Phần này để không ai tốn công sinh ra thứ không ai đọc.

### 7.1 File

| Thứ | Trạng thái |
|---|---|
| `exports/clips.jsonl` (`ClipSegment`) | Sinh ra, ghi checksum, **không ai đọc**. Hiện còn thiếu file. **Không phải model CLIP.** |
| `processed/clip_embeddings/` | 426 vector pooled (chỉ V001), chỉ `clip_pooling.py` ghi, không ai đọc |
| `storage/indexes/scenes.faiss` `.json` `.ids.json` | `offline index` ghi ra, backend `local` **không đọc** (dựng `InMemoryVectorStore` thẳng từ file vector). Chỉ đường Qdrant mới dùng. |
| `assemble_report.json`, `quarantine.jsonl` | Chỉ để người đọc, code không đọc lại |

### 7.2 Trường trong export

Xem bảng chi tiết §4.1 và §4.3. Tóm tắt: **online chỉ đọc `.text` của mọi khối
văn bản và `.label` của object.** Toàn bộ bbox, confidence, language,
normalized_text, provenance, và mọi metadata model đều không vào đường xếp hạng.

Ngoại lệ duy nhất: bbox OCR được bộ lọc lớp phủ dùng gián tiếp.

### 7.3 Biến môi trường không có consumer

**Cả mục 4 của `.env.fpt.local`** ("Provider-neutral routing") — `AIC_QUERY_LLM_PROVIDER`,
`AIC_VLM_PROVIDER`, `AIC_TEXT_RERANK_PROVIDER`, … **không biến nào được code đọc**.
Đó là thiết kế mong muốn, không phải hiện trạng. Đường thật đi qua mục 3.

Có tên trong `.env` nhưng chưa wire vào `Settings`:
`AIC_FPT_QUERY_LLM_MODEL`, `AIC_FPT_DEEP_LLM_MODEL`, `AIC_FPT_TEXT_EMBEDDING_MODEL`,
`AIC_FPT_VI_EMBEDDING_MODEL`, `AIC_FPT_ASR_MODEL`, `AIC_FPT_ASR_FALLBACK_MODEL`,
`AIC_FPT_TTS_MODEL`, cùng nhóm `AIC_KIS_*` / `AIC_QA_*` / `AIC_TRAKE_*` /
`AIC_AVS_RETRIEVAL_TOP_K` / `AIC_AVS_MMR_LAMBDA` ở mục 15.

Bốn biến `AIC_FPT_*` **thực sự được đọc**:
`AIC_FPT_LLM_MODEL` (QA), `AIC_FPT_FAST_LLM_MODEL` (dịch + đồng nghĩa),
`AIC_FPT_RERANK_MODEL` (text rerank), `AIC_FPT_VLM_MODEL` (VLM rerank + enrichment).

### 7.4 Hai bẫy đã cắn thật

- **`AIC_RERANK_VLM_ENABLED=true` với `AIC_RERANK_VLM_URL=` rỗng VẪN bật nhánh**,
  vì container ưu tiên đường FPT. URL rỗng không phải cách tắt.
- **`AIC_ENV_FILE` không tự dò `.env`.** Không đặt tường minh thì server lên với
  `fpt_enabled=False` và không một dòng cảnh báo.

---

## 8. Checklist tối thiểu để online chạy đủ

### Bắt buộc để khởi động được

- [ ] `scenes.jsonl` hợp lệ theo pydantic `Scene` (mỗi scene ≥1 keyframe)
- [ ] `videos.jsonl` **cùng thư mục**, có `source_path` + `frame_count`
- [ ] `AIC_METADATA_JSONL` trỏ đúng `scenes.jsonl`
- [ ] `AIC_DATA_ROOT` trỏ đúng gốc chứa `processed/` và `raw/`

### Bắt buộc để `dense_visual` sống

- [ ] `keyframes.jsonl` cùng thư mục, có `embedding_refs[].storage_locations[]`
- [ ] File vector tồn tại tại `{AIC_DATA_ROOT}/{vector_uri}`
- [ ] `storage/models/clip-vit-large-patch14/` tải sẵn
- [ ] `AIC_VISUAL_EMBEDDING_MODEL` trỏ **đường dẫn local**, không phải tên repo HF

### Bắt buộc để phát được video

- [ ] `raw/videos/<video_id>.mp4` tồn tại thật —
      [playback.py:69](../online/services/playback.py#L69) kiểm tra file và trả
      `None` thay vì URL 404 nếu thiếu (từ 10/08 cả ba video đều có, nên nhánh này
      không còn kích hoạt trên corpus hiện tại)

### Nên có

- [ ] `events.jsonl` cho **mọi** video, không chỉ V001
- [ ] `dataset_manifest.json` cập nhật sau mỗi lần sửa export
- [ ] `color` + `quality` + `selection_score` (CPU thuần, không tốn API)

### Lệnh kiểm tra

```powershell
# pack đúng contract chưa
python -m scripts.verify_stage_pack storage/packs --all

# dựng canonical
python -m offline assemble --packs storage/packs --out storage/exports_multivideo

# export có nhất quán không (sẽ FAIL vì thiếu clips.jsonl)
python -c "from datasection.exporter import verify_export; from pathlib import Path; verify_export(Path('storage/exports_multivideo'))"

# online thấy gì
curl -s http://127.0.0.1:8000/v1/health
curl -s http://127.0.0.1:8000/v1/search/capabilities
```

⚠️ **`/v1/search/capabilities` liệt kê nhánh đã ĐĂNG KÝ, không phải nhánh CÓ DỮ
LIỆU.** `color_search` vẫn xuất hiện trong danh sách dù 0/855 keyframe có màu —
`ColorSearchRetriever` đăng ký với danh sách tài liệu rỗng. Muốn biết nhánh nào
thật sự trả kết quả thì đọc `branch_status` trong response của một truy vấn thật.

Cơ chế báo trung thực duy nhất ở tầng này là **tên nhánh dense**: không có
embedding thật thì container đăng ký `lexical_hash_fallback` thay vì
`dense_visual`, nên `/v1/search/capabilities` không nói dối về nó.

---

## 9. Khi mở rộng lên 876 video

Bốn chỗ trong hợp đồng này sẽ đau trước tiên:

1. **`videos.jsonl` lồng cả `scenes[]`** — ~550 MB đọc rồi vứt. Sửa: đọc streaming
   3 trường.
2. **Toàn bộ export nạp vào RAM** — `JsonlSceneRepository` giữ mọi `SceneDocument`.
   765 scene hiện tại là ~2 MB; 876 video sẽ là ~250k scene.
3. **`InMemoryVectorStore` đọc từng file vector một** — 855 file `open()` hiện tại
   thành ~250k. Gộp thành một `.npy` duy nhất theo video.
4. **`_media_exists` LRU cache 512** — nhỏ hơn số video, sẽ thrash.

Đo được: quét tuyến tính 250k vector chỉ mất **6.4 ms**, nên **chưa cần ANN** —
nút cổ chai là I/O lúc khởi động và RAM, không phải tìm kiếm.

---

## 10. Tài liệu liên quan

- [docs/27_SYSTEM_ISSUES.md](27_SYSTEM_ISSUES.md) — tổng hợp vấn đề toàn hệ thống
- [docs/28_HUMAN_EVAL_RUNBOOK.md](28_HUMAN_EVAL_RUNBOOK.md) — chạy server thủ công
- [contracts/](../contracts/) — JSON Schema gốc
- [datasection/schemas/](../datasection/schemas/) — pydantic, cổng validate thật
