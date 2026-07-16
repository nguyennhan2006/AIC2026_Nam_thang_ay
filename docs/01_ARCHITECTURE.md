# 01. Kiến trúc thống nhất

## Mục tiêu V1

V1 ưu tiên contract ổn định, khả năng chạy lại, truy vết output model và tách
compute nặng khỏi UI. Mọi kết quả Online phải quay lại được `scene_id`,
`keyframe_id`, video và timestamp.

```text
Local UI ──HTTPS──> Vast Online API ──> BM25 + Qdrant/FAISS
                              │                    │
                              ├──> canonical JSONL│
                              └──> media read-only│

Offline controller ──HTTP──> Vast GPU worker
        │                         caption/OCR/object/ASR/embed
        └──> Data exporter ──> manifest + JSONL + indexes
```

## Ranh giới trách nhiệm

| Tầng | Ghi dữ liệu | Trạng thái | Scale |
|---|---|---|---|
| Data | canonical metadata + manifest | immutable theo build | theo dung lượng |
| Offline | artifacts, checkpoint, index | resumable/idempotent | batch/GPU |
| Online | không sửa canonical | stateless, load-on-start | replica CPU |
| GPU worker | model output tạm | bounded concurrency | GPU/model |

## Luồng build

1. Probe video, xác định FPS/frame count.
2. Segment scene theo interval half-open.
3. Chọn keyframe và trích ảnh.
4. Enrich caption/OCR/object; ASR chiếu về scene.
5. Validate toàn cây Video/Scene/Keyframe.
6. Ghi JSONL atomic, tính checksum, ghi manifest cuối cùng.
7. Tạo FAISS/local artifact và Qdrant collection/index/payload.
8. Online chỉ khởi động khi manifest/export hợp lệ.

## Quyết định Qdrant + FAISS

FAISS phù hợp local exact/ANN nhanh và artifact snapshot. Qdrant bổ sung payload
filter, named vectors, cập nhật/upsert, persistence và phục vụ nhiều Online
replica. ID nghiệp vụ không dùng trực tiếp làm point ID: dùng UUIDv5 ổn định,
giữ `scene_id`/`keyframe_id` trong payload.
