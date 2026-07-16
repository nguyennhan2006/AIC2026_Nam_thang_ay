# Data Section — Hợp đồng metadata AIC 2026 (V1)

Tài liệu đầy đủ về tầng dữ liệu của hệ thống truy vấn video đa phương thức:
mục tiêu, chức năng của từng thành phần, quy ước bắt buộc, và hướng dẫn sử
dụng cho người viết pipeline (extract keyframe, OCR, caption, embedding) lẫn
người dùng dữ liệu (search, QA, indexing).

Bản tóm tắt quy ước (tiếng Anh, dạng tra cứu nhanh) nằm ở
[`schemas/README.md`](../schemas/README.md). Khi hai tài liệu lệch nhau,
code trong `schemas/` là nguồn sự thật cuối cùng.

---

## 1. Mục tiêu

Mọi dữ liệu sinh ra trong dự án (keyframe, sau này là scene, video,
transcript…) phải đi qua **một contract duy nhất** trước khi được lưu hoặc
index. Contract này đảm bảo:

- **Thống nhất**: mọi thành viên/pipeline tạo metadata theo cùng một cấu trúc,
  cùng quy ước ID, đường dẫn, checksum.
- **Chặn dữ liệu bẩn từ đầu vào**: field thừa, ID sai format, bbox ngược,
  đường dẫn tuyệt đối… đều bị từ chối ngay khi tạo object, không đợi đến lúc
  search mới phát hiện.
- **Truy vết được**: mọi output của model (OCR, caption, detection) đều gắn
  kèm provenance — model nào, phiên bản pipeline nào, khi nào.
- **Di động**: contract được export ra JSON Schema nên tool ngoài Python
  (TypeScript, Go, notebook validator…) cũng dùng được.

## 2. Cấu trúc thư mục

```text
AIC2026_Nam_thang_ay/
├── pyproject.toml              # khai báo package + dependency (pydantic>=2, Python>=3.11)
├── .gitattributes              # chuẩn hóa line ending (LF) cho mọi hệ điều hành
├── schemas/                    # ★ contract chính (Pydantic models)
│   ├── __init__.py             # public API: from schemas import Keyframe, Scene, ...
│   ├── common.py               # thành phần dùng chung cho MỌI entity
│   ├── keyframe.py             # entity Keyframe (hoàn thành)
│   ├── scene.py                # entity Scene — aggregate root (hoàn thành)
│   └── README.md               # tóm tắt quy ước (quick reference)
├── contracts/
│   ├── keyframe.schema.json    # JSON Schema export — hợp đồng cho tool ngoài Python
│   └── scene.schema.json
├── scripts/
│   └── export_schemas.py       # sinh lại contracts/*.json từ Pydantic model
├── tests/
│   ├── test_keyframe_schema.py # contract test (unittest, không cần pytest)
│   └── test_scene_schema.py
└── docs/
    └── data_section.md         # tài liệu này
```

Chiều phụ thuộc giữa các schema — entity mới **chỉ** import từ `common.py`;
riêng `scene.py` import thêm `Keyframe` vì Scene nhúng trọn keyframe con
(cha nhúng con là chiều xuôi, con không bao giờ import cha):

```text
schemas/common.py  ◄──  schemas/keyframe.py  ◄──  schemas/scene.py
                   ◄──  schemas/scene.py
                   ◄──  schemas/video.py    (sắp tới)
```

## 3. Cài đặt

Yêu cầu: **Python ≥ 3.11** (dùng `StrEnum`), **pydantic ≥ 2.12, < 3**.

```bash
# tại thư mục gốc repo
pip install -e .

# hoặc chỉ cài dependency
pip install "pydantic>=2.12,<3"
```

Kiểm tra nhanh sau khi cài:

```bash
python -c "from schemas import Keyframe; print('OK')"
```

## 4. Quy ước bắt buộc (áp dụng cho mọi entity)

### 4.1. ID phân cấp

ID có độ rộng cố định, phân biệt hoa/thường, và **mã hóa quan hệ cha–con**
ngay trong chuỗi:

| Entity   | Format               | Regex                                  | Ví dụ                    |
|----------|----------------------|----------------------------------------|--------------------------|
| Video    | `Ldd_Vddd`           | `^L[0-9]{2}_V[0-9]{3}$`                | `L01_V001`               |
| Scene    | `<video_id>_Sdddd`   | `^L[0-9]{2}_V[0-9]{3}_S[0-9]{4}$`      | `L01_V001_S0003`         |
| Keyframe | `<scene_id>_Fdddddd` | `^L[0-9]{2}_V[0-9]{3}_S[0-9]{4}_F[0-9]{6}$` | `L01_V001_S0003_F001234` |
| ASR gốc (theo video) | `<video_id>_ASRdddddd` | `^L[0-9]{2}_V[0-9]{3}_ASR[0-9]{6}$` | `L01_V001_ASR000123` |
| ASR projection (theo scene) | `<scene_id>_Adddd` | `^L[0-9]{2}_V[0-9]{3}_S[0-9]{4}_A[0-9]{4}$` | `L01_V001_S0003_A0001` |

Các tầng kiểm tra được thực thi tự động khi tạo object:

1. Từng ID phải khớp regex.
2. `scene_id` phải bắt đầu bằng `video_id`; với `Scene`, 4 chữ số sau `S`
   phải bằng `scene_idx` (nên `scene_idx` giới hạn `0–9999`).
3. `keyframe_id` phải bằng `scene_id + "_F" + frame_idx` (6 chữ số, zero-pad).
   Vì vậy `frame_idx` bị giới hạn `0–999999`.
4. Trong một `Scene`: mọi keyframe con phải đúng `video_id`/`scene_id` và nằm
   trong khoảng của scene; ASR projection phải mang tiền tố `scene_id`, ASR
   gốc phải mang tiền tố `video_id`.

ID **không được sinh lại** khi chạy lại model với phiên bản mới — identity
của dữ liệu tách khỏi nội dung do model sinh.

### 4.2. Scene manifest là bất biến

Vì `keyframe_id` chứa `scene_id`, nếu biên scene thay đổi thì identity của
keyframe cũng đổi. Do đó: **scene manifest sau khi publish không được sửa**.
Nếu chạy lại TransNetV2 và biên scene khác đi — tạo dataset revision/namespace
mới, không ghi đè và không tái sử dụng ID cũ.

### 4.3. Đường dẫn media

- Là **đường dẫn POSIX tương đối** so với biến môi trường `AIC_DATA_ROOT`.
- Đường dẫn Windows (`\`) được tự chuẩn hóa thành `/` khi validate.
- Bị từ chối: đường dẫn tuyệt đối, chứa `..`, hoặc là URI.

```text
✔ processed/keyframes/L01_V001/frame_001234.jpg
✘ C:\dataset\frame.jpg      (tuyệt đối Windows)
✘ /dataset/frame.jpg        (tuyệt đối POSIX)
✘ ../outside/frame.jpg      (traversal)
✘ s3://bucket/frame.jpg     (URI không được đặt vào media path)
```

### 4.4. Artifact URI

Các trường tham chiếu artifact (`histogram_uri`, `vector_uri`) nhận **hoặc**
đường dẫn tương đối như trên, **hoặc** URI với scheme trong danh sách:
`az`, `file`, `gs`, `https`, `qdrant`, `s3`. Scheme khác bị từ chối.

### 4.5. Checksum

```text
sha256:<64 ký tự hex viết thường>
```

Tính từ **bytes của chính file ảnh keyframe** mà `image_path` trỏ tới —
không phải video nguồn, không phải pixel array sau decode.

```python
import hashlib
digest = hashlib.sha256(open(image_file, "rb").read()).hexdigest()
source_checksum = f"sha256:{digest}"
```

### 4.6. Thời gian

- Mọi `created_at` phải là datetime **có timezone** (khuyến nghị UTC —
  dùng sẵn `schemas.common.utc_now()`). Datetime naive bị từ chối.
- `frame_idx` là chỉ số frame zero-based trong video gốc → dùng cho identity.
- `timestamp_sec` là thời gian (giây) để mở video/hiển thị kết quả.
- **Khoảng của scene là nửa-mở**: `[start_frame, end_frame_exclusive)` và
  `[start_sec, end_sec)`. Keyframe nằm đúng biên cuối thuộc về scene **kế
  tiếp**, không thuộc scene hiện tại — nhờ vậy hai scene liền nhau không bao
  giờ tranh chấp một frame.
- Kiểm tra chéo `frame_idx / fps ≈ timestamp_sec` sẽ nằm ở validator cấp
  `Video` (nơi giữ `fps`) — không lặp fps vào từng keyframe.

### 4.7. Nguyên tắc chung của mọi model

Mọi model kế thừa `StrictModel` với:

- `extra="forbid"` — field lạ → lỗi ngay (chống typo tên field).
- `validate_assignment=True` — gán lại giá trị sai sau khi tạo cũng bị chặn.
- `str_strip_whitespace=True` — chuỗi tự động trim.
- Enum được **giữ nguyên trong Python** (`KeyframeRole.OCR_RICH`) nhưng xuất
  **chuỗi** trong JSON (`"ocr_rich"`).

## 5. Chức năng từng thành phần

### 5.1. `schemas/common.py` — dùng chung cho mọi entity

| Thành phần | Chức năng |
|---|---|
| `StrictModel` | Base model, cấu hình strict nêu ở mục 4.7 |
| `VideoId` / `SceneId` / `KeyframeId` | Kiểu ID có regex, dùng ở mọi entity |
| `SHA256Checksum` | Kiểu checksum `sha256:<hex>` |
| `RelativeArtifactPath` | Đường dẫn tương đối, tự chuẩn hóa `\` → `/`, chặn tuyệt đối/`..`/URI |
| `ArtifactURI` | Đường dẫn tương đối HOẶC URI thuộc scheme cho phép |
| `Probability` | Float bắt buộc trong `[0, 1]` — dùng cho mọi confidence/score |
| `NonEmptyStr` | Chuỗi không rỗng |
| `utc_now()` | Timestamp UTC có timezone |
| `BoundingBox` | Box XYXY chuẩn hóa `[0, 1]`, độc lập độ phân giải, bắt buộc `x2 > x1`, `y2 > y1` |
| `ModelProvenance` | Truy vết output của model: `model_name`, `pipeline_version` (bắt buộc), revision, prompt, device, tham số |
| `VectorLocation` | MỘT vị trí vật lý của vector: `backend` (`faiss`/`qdrant`/`file`) + `vector_id` + `index_name`; backend `file` bắt buộc có `vector_uri` |
| `EmbeddingReference` | MỘT embedding logic + danh sách mọi backend đang chứa nó; không trùng cặp backend/index |

### 5.2. `schemas/keyframe.py` — entity Keyframe

| Thành phần | Chức năng |
|---|---|
| `KeyframeRole` | Lý do frame được chọn: `representative`, `middle`, `boundary_start`, `boundary_end`, `ocr_rich`, `motion_change`, `manual` |
| `CaptionRecord` | Một caption do model sinh: `short`/`detailed`/`tags`/`crop`; loại `crop` bắt buộc có `crop_bbox` (và chỉ loại `crop` mới được có) |
| `OCRInstance` | Một dòng text nhận dạng được + vị trí bbox + confidence + provenance |
| `ObjectInstance` | Một object detect được (bằng chứng mềm cho search, không phải ground truth) |
| `ColorFeature` | Metadata màu gọn: tối đa 8 màu HEX chủ đạo, mean HSV, URI histogram |
| `QualitySignals` | Tín hiệu chất lượng (sharpness, brightness, contrast, black-frame, duplicate) để chọn/hạ hạng keyframe yếu |
| `Keyframe` | Model tổng: identity + nội dung model sinh + embedding refs + checksum |
| `Keyframe.ocr_text` | Property gộp toàn bộ text OCR theo thứ tự — tiện hiển thị/index text |
| `Keyframe.qdrant_payload()` | Sinh payload gọn để đưa vào Qdrant (xem mục 6.4) |

Lưu ý thiết kế: **embedding không nằm trong metadata**. Metadata chỉ giữ
*tham chiếu* (`EmbeddingReference`) tới vector trong FAISS/Qdrant/file —
tránh phình JSON và tránh hai nguồn sự thật cho vector.

### 5.3. `schemas/scene.py` — entity Scene (aggregate root)

| Thành phần | Chức năng |
|---|---|
| `TransitionType` | Kiểu chuyển cảnh tại biên scene: `cut`, `fade`, `dissolve`, `unknown` |
| `SceneCaptionRecord` | Caption cấp scene (`visual`/`audio_visual`/`summary`/`tags`); `evidence_keyframe_ids` chỉ được trỏ tới keyframe có thật trong scene |
| `ASRSegment` | **Bản chiếu đã clip theo scene** của một segment ASR trên timeline video; luôn giữ `source_segment_id` để truy ngược segment gốc khi nó vắt qua biên scene |
| `SceneKeyword` | Keyword tìm kiếm kèm nguồn gốc (`caption`/`ocr`/`asr`/`object`/`manual`); keyword sinh tự động **bắt buộc** có provenance |
| `Scene` | Aggregate root: biên thời gian nửa-mở + provenance của segmentation + **nhúng trọn** danh sách `Keyframe` con + caption/ASR/keyword/embedding cấp scene |
| `Scene.duration_frames` / `duration_sec` | Độ dài suy ra từ biên |
| `Scene.ocr_text` | Gộp OCR từ keyframe con — **suy diễn**, không lưu thành giá trị thứ hai |
| `Scene.asr_text` | Gộp text ASR theo thứ tự thời gian |
| `Scene.qdrant_payload()` | Payload gọn với `entity_type: "scene"` |

**Quy tắc vận hành quan trọng:** vì Scene nhúng trọn keyframe con, bản Scene
là **bản canonical duy nhất** được publish. Pipeline sinh keyframe → gói vào
scene → chỉ lưu/index scene; không duy trì song song file keyframe JSON rời,
tránh hai bản sao của cùng một keyframe lệch nhau. Và như mục 4.2: scene
manifest sau khi publish là bất biến.

## 6. Hướng dẫn sử dụng

### 6.1. Tạo một Keyframe hợp lệ (phía pipeline)

```python
from schemas import Keyframe, KeyframeRole
from schemas.common import ModelProvenance
from schemas.keyframe import CaptionRecord, OCRInstance
from schemas.common import BoundingBox

prov = ModelProvenance(
    model_name="paddleocr-v4",
    pipeline_version="kf-pipeline-1.2.0",
)

kf = Keyframe(
    keyframe_id="L01_V001_S0003_F001234",
    video_id="L01_V001",
    scene_id="L01_V001_S0003",
    frame_idx=1234,
    timestamp_sec=41.133,
    image_path="processed/keyframes/L01_V001/frame_001234.jpg",
    width=1920,
    height=1080,
    roles=[KeyframeRole.REPRESENTATIVE, KeyframeRole.OCR_RICH],
    ocr_instances=[
        OCRInstance(
            text="HTV9",
            confidence=0.98,
            bbox=BoundingBox(x1=0.05, y1=0.04, x2=0.18, y2=0.11),
            provenance=prov,
        ),
    ],
    source_checksum="sha256:" + "a" * 64,  # thay bằng hash thật của file ảnh
)
```

Nếu bất kỳ giá trị nào sai quy ước, constructor ném `pydantic.ValidationError`
với thông báo chỉ rõ field lỗi — pipeline nên để lỗi này nổi lên thay vì nuốt.

### 6.2. Ghi ra / đọc lại JSON

```python
# ghi (mỗi keyframe một JSON, hoặc JSONL mỗi dòng một keyframe)
json_str = kf.model_dump_json()

# đọc lại + validate trong một bước
kf2 = Keyframe.model_validate_json(json_str)
assert kf2 == kf
```

Mọi file JSON do pipeline khác tạo ra **phải** được nạp qua
`Keyframe.model_validate_json(...)` (hoặc validate bằng
`contracts/keyframe.schema.json`) trước khi index.

### 6.3. Gắn embedding (một vector, nhiều backend)

```python
from schemas.common import EmbeddingReference, VectorLocation

kf.embedding_refs = [
    EmbeddingReference(
        embedding_name="clip_visual_v1",
        modality="image",
        model_name="openai/clip-vit-large-patch14",
        dimension=768,
        storage_locations=[
            VectorLocation(backend="faiss", vector_id="42",
                           index_name="keyframes_v1"),
            VectorLocation(backend="qdrant",
                           vector_id="L01_V001_S0003_F001234",
                           index_name="aic_keyframes_v1"),
        ],
    ),
]
```

- `embedding_name` là duy nhất trong một keyframe.
- Một embedding logic có thể nằm đồng thời ở FAISS (search offline) và
  Qdrant (serve online) — không nhân đôi bản ghi.

### 6.4. Đưa vào Qdrant

```python
payload = kf.qdrant_payload()
```

Payload chỉ chứa field phục vụ **filter + trace + hiển thị kết quả đầu**:
`entity_type`, `schema_version`, 3 ID, `frame_idx`, `timestamp_sec`,
`image_path`, `roles`, `has_ocr`, `has_caption`, `object_labels`,
`pipeline_versions`. Caption đầy đủ, bbox OCR… vẫn ở metadata store chính —
Qdrant không phải nơi lưu toàn văn.

`entity_type` là discriminator: Keyframe (`"keyframe"`) và Scene (`"scene"`)
dùng chung collection vẫn phân biệt được loại điểm.

### 6.5. Tạo Scene hoàn chỉnh (phía pipeline)

Scene gói keyframe đã tạo ở mục 6.1 cùng biên thời gian và bằng chứng cấp
scene. Mọi ràng buộc (keyframe đúng scene, nằm trong khoảng nửa-mở, đúng thứ
tự, ASR đúng tiền tố…) được kiểm ngay khi khởi tạo:

```python
from schemas import Scene
from schemas.scene import ASRSegment, SceneCaptionRecord
from schemas.common import ModelProvenance

scene = Scene(
    scene_id="L01_V001_S0003",
    video_id="L01_V001",
    scene_idx=3,                      # phải khớp 4 chữ số sau S trong scene_id
    start_frame=1200,
    end_frame_exclusive=1300,         # nửa-mở: frame 1300 thuộc scene kế tiếp
    start_sec=40.0,
    end_sec=43.334,
    segmentation_provenance=ModelProvenance(
        model_name="TransNetV2", pipeline_version="seg-1.0.0",
    ),
    keyframes=[kf],                   # nhúng trọn Keyframe từ mục 6.1
    captions=[
        SceneCaptionRecord(
            caption_type="visual",
            text="A news anchor introduces the segment.",
            evidence_keyframe_ids=["L01_V001_S0003_F001234"],  # phải là con thật
            provenance=ModelProvenance(
                model_name="caption-model", pipeline_version="cap-1.0.0",
            ),
        ),
    ],
    asr_segments=[
        ASRSegment(
            segment_id="L01_V001_S0003_A0001",       # projection trong scene này
            source_segment_id="L01_V001_ASR000123",  # segment ASR gốc của video
            start_sec=40.2,
            end_sec=43.334,                          # đã clip theo biên scene
            text="Xin đừng quên nhau",
            language="vi",
            provenance=ModelProvenance(
                model_name="whisper-large-v3", pipeline_version="asr-1.0.0",
            ),
        ),
    ],
)

scene.duration_sec     # 3.334 — suy từ biên
scene.ocr_text         # gộp OCR từ keyframe con (suy diễn, không lưu lại)
scene.asr_text         # text ASR theo thứ tự thời gian
payload = scene.qdrant_payload()   # entity_type = "scene"
```

Lưu ý về ASR: nếu một segment ASR gốc vắt qua biên hai scene thì mỗi scene giữ
một bản chiếu đã cắt (`_A0001`, `_A0002`…) nhưng cả hai cùng trỏ về một
`source_segment_id` — nhờ đó vẫn truy ngược được câu nói nguyên vẹn.

### 6.6. Các lỗi thường gặp

| Làm sai | Thông báo lỗi (tóm tắt) |
|---|---|
| ID không đúng format | `String should match pattern '^L[0-9]{2}_...'` |
| `scene_id` không thuộc `video_id` | `scene_id must belong to video_id` |
| `keyframe_id` lệch `frame_idx` | `keyframe_id must equal L01_V001_S0003_F00123x...` |
| `image_path` tuyệt đối / có `..` / là URI | `artifact path must be relative to AIC_DATA_ROOT` / `must not contain '..'` / `must not be a URI` |
| Checksum thiếu prefix hoặc hex hoa | `String should match pattern '^sha256:[0-9a-f]{64}$'` |
| Thêm field không có trong schema | `Extra inputs are not permitted` |
| Bbox ngược (`x2 <= x1`) | `bbox requires x2 > x1 and y2 > y1` |
| Caption `crop` không có `crop_bbox` | `crop captions require crop_bbox` |
| `created_at` không có timezone | `created_at must include timezone information` |
| `scene_id` không khớp `scene_idx` | `scene_id must equal L01_V001_S000x for video/scene_idx` |
| Keyframe con thuộc scene/video khác | `keyframe ... belongs to another scene/video` |
| Keyframe ngoài khoảng của scene (kể cả đúng biên cuối) | `keyframe ... is outside scene frame/time interval` |
| Keyframe con trùng hoặc sai thứ tự | `keyframe_id values must be unique` / `must be ordered by frame_idx` |
| Caption dẫn chứng keyframe không tồn tại trong scene | `scene caption references unknown keyframes` |
| ASR projection sai scene / vượt khoảng thời gian | `ASR segment ... belongs to another scene` / `is outside scene time interval` |
| Keyword tự động (ocr/asr/caption/object) thiếu provenance | `automatically derived keywords require provenance` |

### 6.7. Dùng contract ngoài Python

`contracts/keyframe.schema.json` và `contracts/scene.schema.json` là JSON
Schema chuẩn — dùng để:

- Validate JSON output của notebook (ví dụ bằng `ajv` cho Node, `jsonschema`
  cho Python thuần).
- Sinh TypeScript types (`json-schema-to-typescript`) cho frontend.
- Làm hợp đồng dữ liệu giữa nhóm Offline (extraction) và Online (search/QA).

Giới hạn: các kiểm tra **liên trường** (khớp `frame_idx` với `keyframe_id`,
quan hệ scene–video, keyframe nằm trong khoảng scene, thứ tự thời gian…)
không biểu diễn được bằng JSON Schema — chỉ Pydantic (hoặc validator tương
đương) mới bắt được. Vì vậy validate bằng JSON Schema là điều kiện *cần*,
validate bằng `schemas.Keyframe` / `schemas.Scene` là điều kiện *đủ*.

## 7. Quy trình khi sửa schema

1. Sửa model trong `schemas/`.
2. Nếu thay đổi phá vỡ tương thích (đổi format ID, bỏ field, đổi ngữ nghĩa)
   → tăng `schema_version`. Nới quy ước âm thầm (ví dụ nới regex ID) là
   **không được phép**.
3. Chạy lại exporter để đồng bộ contract:

   ```bash
   python scripts/export_schemas.py
   ```

4. Chạy test — test cuối cùng sẽ fail nếu quên bước 3:

   ```bash
   python -m unittest discover -s tests -v
   ```

5. Commit **cả** thay đổi model lẫn `contracts/*.json` trong cùng một commit.

## 8. Trạng thái và lộ trình

| Entity | Trạng thái | Ghi chú |
|---|---|---|
| `Keyframe` | ✅ Hoàn thành (V1) | `schemas/keyframe.py` |
| `Scene` | ✅ Hoàn thành (V1) | `schemas/scene.py` — aggregate root, nhúng keyframe, ASR projection, keyword |
| `Video` | ⏳ Tiếp theo | fps (nguồn cho kiểm tra chéo frame/timestamp), resolution, nguồn phát hành, danh sách scene |
| Transcript/ASR gốc | 💡 Dự kiến | manifest ASR cấp video (`Ldd_Vddd_ASRdddddd`); trong scene hiện đã có bản chiếu |

Mọi entity mới lặp lại đúng khung của Keyframe: kế thừa `StrictModel`,
có `schema_version`, ID theo phân cấp, `created_at` UTC, `extensions` cho
field thử nghiệm, provenance cho dữ liệu model sinh, embedding theo tham
chiếu, và `qdrant_payload()` với `entity_type` riêng.
