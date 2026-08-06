# 23 — Định dạng gold query (để soạn query cho video mới)

Viết trước khi có query của L21_V002/V003, để soạn xong là chạy được ngay
không phải sửa lại.

Kiểm trước khi chạy eval:

```bash
python -m scripts.validate_gold --gold <file>.jsonl \
    --metadata storage/exports_multivideo/scenes.jsonl
```

Nó bắt lỗi cấu trúc, kiểm frame có nằm trong phạm vi video không, cảnh báo
frame nào quá xa keyframe gần nhất (truy vấn đó gần như không thể ăn điểm), và
báo số truy vấn mỗi task có đủ để đọc kết quả không.

Một dòng JSON mỗi truy vấn (JSONL). **Bốn task dùng bốn shape khác nhau** — đây
là chỗ dễ sai nhất, và bản đầu của chính validator này cũng đã sai.

---

## Chung cho mọi task

| trường | bắt buộc | ghi chú |
|---|---|---|
| `query_id` | có | duy nhất trong file, vd `KIS_V002_E01` |
| `task` | có | `KIS` \| `VQA` \| `TRAKE` \| `AVS` |
| `target_video` | có | phải khớp `video_id` trong export, vd `L21_V002` |
| `difficulty` | nên có | `Easy` \| `Medium` \| `Hard` — dùng để phân tích lỗi theo độ khó |

---

## KIS — tìm một khoảnh khắc

```json
{"query_id": "KIS_V002_E01", "task": "KIS", "difficulty": "Easy",
 "query_vi": "cửa hàng tiện lợi có bảng hiệu đỏ, nhiều xe máy đậu trước cửa",
 "target_video": "L21_V002",
 "target_intervals": [{"start_frame": 4410, "end_frame": 4500}],
 "representative_frame": 4455}
```

`target_intervals` là khoảng frame ĐƯỢC CHẤP NHẬN, không phải một frame duy
nhất. Nộp bài trúng bất kỳ frame nào trong khoảng đều tính đúng.

## VQA — hỏi đáp

```json
{"query_id": "VQA_V002_E01", "task": "VQA", "difficulty": "Medium",
 "question_vi": "Cửa hàng trong cảnh này tên gì?",
 "answer_canonical": "Circle K",
 "accepted_answers": ["Circle K", "CircleK"],
 "target_video": "L21_V002",
 "target_intervals": [{"start_frame": 4410, "end_frame": 4500}]}
```

VQA dùng `question_vi`, **không** phải `query_vi`. Chấm đúng cần cả ba: đúng
video, frame trong khoảng, và answer khớp `accepted_answers` — nên liệt kê mọi
biến thể viết hợp lệ, kể cả khác dấu cách và viết hoa.

## TRAKE — chuỗi khoảnh khắc theo thứ tự

```json
{"query_id": "TRAKE_V002_E01", "task": "TRAKE", "difficulty": "Medium",
 "query_vi": "Tìm video và căn chỉnh ba khoảnh khắc: (1) ...; (2) ...; (3) ...",
 "target_video": "L21_V002",
 "event_count": 3,
 "events": [
   {"event_order": 1, "description_vi": "...", "representative_frame": 1200,
    "gt_start_frame": 1196, "gt_end_frame": 1204, "semantic_time_sec": 40.0},
   {"event_order": 2, "description_vi": "...", "representative_frame": 1560,
    "gt_start_frame": 1556, "gt_end_frame": 1564, "semantic_time_sec": 52.0},
   {"event_order": 3, "description_vi": "...", "representative_frame": 1980,
    "gt_start_frame": 1976, "gt_end_frame": 1984, "semantic_time_sec": 66.0}
 ]}
```

`representative_frame` là bắt buộc cho MỖI event. `gt_start_frame`/`gt_end_frame`
là cửa sổ chú thích hẹp (bộ hiện có dùng ±4 frame); **cửa sổ CHẤM ĐIỂM rộng hơn
nhiều** và được suy ra lúc chạy từ độ dài scene
(`clamp(scene_duration × 0.5, 2s, 7s)`), nên không cần chú thích chính xác tới
từng frame — chỉ cần mốc ngữ nghĩa đúng.

Frame phải **tăng dần** theo `event_order`: thứ tự là ràng buộc cứng của luật.

## AVS — tìm mọi đoạn thoả tiêu chí

```json
{"query_id": "AVS_V002_E01", "task": "AVS", "difficulty": "Medium",
 "query_vi": "cảnh giao thông đông đúc ban ngày",
 "criteria": "daytime AND heavy traffic",
 "target_video": "L21_V002",
 "relevant_intervals": [
   {"event_id": "E01_GIAO_THONG", "start_frame": 4260, "end_frame": 4560,
    "relevance_grade": 3, "reason": "đường đông kín xe máy"},
   {"event_id": "E02_NGA_TU", "start_frame": 7800, "end_frame": 7980,
    "relevance_grade": 2, "reason": "xe thưa hơn"}
 ]}
```

AVS dùng `relevant_intervals` (**không** phải `target_intervals`) và mỗi khoảng
cần `relevance_grade` 0–3 — nDCG cần thang điểm chứ không chỉ nhãn đúng/sai.
`event_id` dùng để tính `event_coverage`: hai khoảng cùng `event_id` được coi
là cùng một sự kiện, nên hệ tìm được cả hai cũng chỉ tính là phủ một.

---

## Bao nhiêu truy vấn thì đủ

Đây là ràng buộc quan trọng hơn cả chất lượng từng truy vấn.

| số truy vấn / task | 1 truy vấn đáng | đọc được gì |
|---|---|---|
| 8 | 0.125 | gần như không gì |
| 12 (hiện tại) | 0.083 | chỉ thay đổi rất lớn |
| 20 | 0.050 | chênh 2 truy vấn bắt đầu có nghĩa |
| 40 | 0.025 | so được các cấu hình gần nhau |

Bộ hiện có: 12 KIS, 12 VQA, 8 TRAKE, 8 AVS trên một video. Với `top1_pairwise_accuracy`
của KIS thì mẫu còn nhỏ hơn nữa — chỉ **11 quyết định**. Một reranker sửa được
2 truy vấn cho 0.545 -> 0.727, nghe rất to nhưng là 2 truy vấn.

**Nếu chỉ soạn được ít, ưu tiên KIS và TRAKE.** KIS vì đó là nơi cần đo tinh
(top-1 vs top-2). TRAKE vì Stage A chọn video mới là chỗ vừa phát hiện nút thắt
và chỉ đo được khi có nhiều video.

## Vì sao query cho video MỚI có giá trị riêng

Mọi tinh chỉnh tới giờ đều fit lên đúng L21_V001. Query cho V002/V003 mở khoá
thứ không cách nào khác thay được: **holdout thật** — chạy trên video chưa từng
dùng để tinh chỉnh, để biết cải tiến có tổng quát hay chỉ vừa khít một video.

`scripts/eval_tasks.py` tự tách chỉ số theo video khi gold có nhiều
`target_video`, nên chỉ cần nộp file là có bảng so sánh.

## Giới hạn đã biết của dữ liệu

L21_V002/V003 **không có audio**, nên không có ASR. Truy vấn dựa vào lời nói sẽ
không bao giờ ăn điểm ở hai video này, và nhánh `bm25_asr` phải bị tắt trong
mọi phép đo đa video (`--disable-branch bm25_asr`). Đừng soạn truy vấn dạng
"người dẫn nói rằng..." cho hai video đó.
