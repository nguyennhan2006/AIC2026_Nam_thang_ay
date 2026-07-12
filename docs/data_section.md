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
├── schemas/                    # ★ contract chính (Pydantic models)
│   ├── __init__.py             # public API: from schemas import Keyframe, ...
│   ├── common.py               # thành phần dùng chung cho MỌI entity
│   ├── keyframe.py             # entity Keyframe (phần tử đầu tiên, đã hoàn thành)
│   └── README.md               # tóm tắt quy ước (quick reference)
├── contracts/
│   └── keyframe.schema.json    # JSON Schema export — hợp đồng cho tool ngoài Python
├── scripts/
│   └── export_schemas.py       # sinh lại contracts/*.json từ Pydantic model
├── tests/
│   └── test_keyframe_schema.py # 8 contract test (unittest, không cần pytest)
└── docs/
    └── data_section.md         # tài liệu này
```

Chiều phụ thuộc giữa các schema (entity mới **chỉ** import từ `common.py`,
không import chéo lẫn nhau):

```text
schemas/common.py  ◄──  schemas/keyframe.py
                   ◄──  schemas/scene.py    (sắp tới)
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

Ba tầng kiểm tra được thực thi tự động khi tạo `Keyframe`:

1. Từng ID phải khớp regex.
2. `scene_id` phải bắt đầu bằng `video_id`.
3. `keyframe_id` phải bằng `scene_id + "_F" + frame_idx` (6 chữ số, zero-pad).
   Vì vậy `frame_idx` bị giới hạn `0–999999`.

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

`entity_type: "keyframe"` là discriminator: sau này Scene/Video dùng chung
collection vẫn phân biệt được loại điểm.

### 6.5. Các lỗi thường gặp

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

### 6.6. Dùng contract ngoài Python

`contracts/keyframe.schema.json` là JSON Schema chuẩn — dùng để:

- Validate JSON output của notebook (ví dụ bằng `ajv` cho Node, `jsonschema`
  cho Python thuần).
- Sinh TypeScript types (`json-schema-to-typescript`) cho frontend.
- Làm hợp đồng dữ liệu giữa nhóm Offline (extraction) và Online (search/QA).

Giới hạn: các kiểm tra **liên trường** (khớp `frame_idx` với `keyframe_id`,
quan hệ scene–video) không biểu diễn được bằng JSON Schema — chỉ Pydantic
(hoặc validator tương đương) mới bắt được. Vì vậy validate bằng JSON Schema
là điều kiện *cần*, validate bằng `schemas.Keyframe` là điều kiện *đủ*.

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
| `Keyframe` | ✅ Hoàn thành (V1, 8/8 test) | `schemas/keyframe.py` |
| `Scene` | ⏳ Tiếp theo | boundary frames, thời lượng, danh sách keyframe, manifest bất biến |
| `Video` | ⏳ Sau Scene | fps (nguồn cho kiểm tra chéo frame/timestamp), resolution, nguồn phát hành |
| Transcript/ASR | 💡 Dự kiến | gắn theo Video/Scene, cùng quy ước provenance |

Mọi entity mới lặp lại đúng khung của Keyframe: kế thừa `StrictModel`,
có `schema_version`, ID theo phân cấp, `created_at` UTC, `extensions` cho
field thử nghiệm, provenance cho dữ liệu model sinh, embedding theo tham
chiếu, và `qdrant_payload()` với `entity_type` riêng.
