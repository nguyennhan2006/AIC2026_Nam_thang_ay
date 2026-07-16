# 02. Data contract

## ID và interval

| Entity | Pattern | Ví dụ |
|---|---|---|
| Video | `Ldd_Vddd` | `L01_V001` |
| Scene | `<video>_Sdddd` | `L01_V001_S0003` |
| Keyframe | `<scene>_Fdddddd` | `L01_V001_S0003_F001234` |

Frame/time interval của scene là `[start, end_exclusive)`. Timestamp keyframe
phải khớp `frame_idx / fps` trong tolerance hai frame. Scene nằm trong Video,
keyframe nằm trong Scene, caption scene chỉ tham chiếu keyframe con.

## Path, checksum, vector

- Mọi path media tương đối theo `AIC_DATA_ROOT`, dùng POSIX `/`, cấm absolute và `..`.
- Checksum có dạng `sha256:<64 lowercase hex>`.
- Qdrant point ID là UUIDv5 từ business ID; business ID luôn có trong payload.
- Vector lớn không nhúng vào JSONL; metadata chỉ lưu reference/index version.

## Export bắt buộc

`storage/exports/` gồm `videos.jsonl`, `scenes.jsonl`, `keyframes.jsonl`,
`dataset_manifest.json`. Manifest khóa count, pipeline/model/index version và
checksum. Ghi file data trước, `fsync + rename`, sau đó mới publish manifest.
Online không đọc file `.tmp`.

## Migration

Không sửa nghĩa field trong schema `1.0.0`. Thêm field optional được phép ở
minor version; breaking change tăng major và viết converter. Dataset build là
immutable; rollback bằng đổi đường dẫn manifest/export, không sửa tại chỗ.
