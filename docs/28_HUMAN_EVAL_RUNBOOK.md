# 28 — Runbook chạy server để đánh giá thủ công

Mọi lệnh dưới đây **đã chạy thật** ngày 2026-08-06 trên chính máy này, không
phải chép từ tài liệu cũ. `docs/12` §2 còn trỏ `storage/exports_l21` — đường
dẫn đó không còn dùng.

---

## Bước 0 — dọn cổng trước khi khởi động

Bỏ qua bước này là nguồn nhầm lẫn tốn thời gian nhất: uvicorn trên Windows để
lại tiến trình cũ giữ cổng 8000, và bạn sẽ thấy **server mới nhưng dữ liệu
cũ**. Tôi vừa dính đúng lỗi này: `/v1/health` báo `video_count: 1` sau khi đã
sửa manifest, chỉ vì tiến trình cũ chưa chết.

```powershell
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }
```

`pkill -f uvicorn` trong Git Bash **không** diệt được tiến trình này.

---

## Bước 1 — chạy backend

```powershell
$env:AIC_ENV_FILE       = ".env.fpt.local"
$env:AIC_METADATA_JSONL = "storage/exports_multivideo/scenes.jsonl"
$env:AIC_DATA_ROOT      = "storage"
$env:PYTHONIOENCODING   = "utf-8"
$env:PYTHONHASHSEED     = "0"

python -m uvicorn online.api.app:app --host 127.0.0.1 --port 8000
```

Ba biến đầu **bắt buộc**:

- `AIC_ENV_FILE` — config không tự dò `.env`; thiếu biến này thì mọi model FPT,
  cờ nhánh và khoá API đều về mặc định, và bạn đánh giá một hệ thống khác.
- `AIC_METADATA_JSONL` — mặc định trỏ export cũ một video.
- `AIC_DATA_ROOT` — để `/v1/media/...` phục vụ được ảnh keyframe cho UI.

`PYTHONHASHSEED=0` không bắt buộc cho việc dùng tay, nhưng giữ nó thì kết quả
trùng với số trong `docs/26`–`docs/27`.

Khởi động mất **~45 giây** (nạp model CLIP). Đừng gõ lệnh tiếp khi chưa thấy
`Application startup complete`.

## Bước 2 — kiểm tra trước khi tin bất cứ kết quả nào

```powershell
curl.exe -s http://127.0.0.1:8000/v1/health
```

Phải khớp **đúng** bốn số này:

```json
{"scene_count": 765, "video_count": 3, "keyframe_count": 855, "branch_count": 8}
```

| số sai | nghĩa là |
|---|---|
| `video_count: 1`, `keyframe_count: 307` | đang chạy tiến trình cũ, hoặc trỏ nhầm export — quay lại bước 0 |
| `branch_count: 10` | `AIC_ENV_FILE` chưa được nạp; `ocr_fuzzy`/`event_search` đang bật và **tụt KIS R@1 từ 0.583 xuống 0.500** |
| `scene_count: 3` | trỏ nhầm `storage/exports` (export demo) |

Kiểm thêm một dấu hiệu về **tốc độ**: một truy vấn KIS phải xong trong **dưới
1 giây**. Nếu mất 8–9 giây thì `AIC_RERANK_VLM_ENABLED` đang bật — nó tiêu 94%
số lời gọi API mà không đổi một chỉ số nào (xem `docs/27` §D0).

`dataset_version` phải là `20260806T084923Z` trở đi.

## Bước 3 — giao diện

Hai lựa chọn, dùng cái nào cũng được.

**Nhanh — UI tĩnh, không cần cài gì:**

```
http://127.0.0.1:8000/ui/
```

**Đầy đủ — UI React, có bảng submission và xem video:**

```powershell
cd online/ui-react
npm install      # chỉ lần đầu
npm run dev      # http://127.0.0.1:5173
```

---

## Bước 4 — truy vấn từng task

Trên UI thì chọn chip task rồi gõ. Bằng `curl` thì ghi body ra file trước —
PowerShell 5.1 nuốt dấu nháy và làm hỏng dấu tiếng Việt nếu nhét thẳng vào `-d`:

```powershell
Set-Content "$env:TEMP\q.json" -Encoding utf8 `
  '{"query":"Tìm cảnh có biển màu vàng đỏ ghi cảnh báo sạt lở nguy hiểm","top_k":20}'

curl.exe -s -X POST http://127.0.0.1:8000/v1/search/kis `
  -H "Content-Type: application/json" --data-binary "@$env:TEMP\q.json"
```

Đổi `kis` thành `qa`, `trake`, `avs` cho ba task còn lại. Đầu ra thật:

```
#1 L21_V001 frame=2130  score=1.488
#2 L21_V001 frame=22392 score=1.327
#3 L21_V001 frame=2217  score=1.318
```

### Bạn chỉ cần duyệt 5 dòng đầu, không phải 20

Đo trên 36 truy vấn KIS với **đúng cấu hình mặc định** (không ghi đè gì):

```
R@1 0.583   R@5 1.000   R@20 1.000      hạng gold tệ nhất = 5
```

**Đáp án nằm trong 5 kết quả đầu ở cả 36/36 truy vấn.** Cứ để `top_k: 20` để
có chỗ dự phòng, nhưng thực tế mắt bạn chỉ cần chạy qua 5 dòng.

### Trả về 15 dòng dù xin 20 là ĐÚNG, không phải lỗi

Chính sách khử trùng của KIS là `max_per_video=5, max_per_event=1`, nên 3 video
cho tối đa 15 dòng. Đó là 15 **sự kiện khác nhau**, không phải 20 dòng gần
trùng nhau.

Và nó tốt hơn thật, không chỉ gọn hơn:

| | R@1 | R@5 | MRR |
|---|---|---|---|
| mặc định (5/video) | 0.583 | **1.000** | **0.733** |
| bỏ trần (`max_results_per_video: 1000000`) | 0.583 | 0.917 | 0.725 |

Nên **đừng nới trần** như `docs/12` §6 gợi ý. Gộp scene gần trùng của cùng một
sự kiện làm top-5 chứa nhiều sự kiện khác nhau hơn — đúng thứ bạn cần khi
duyệt tay.

Mỗi phản hồi có `query_id`; giữ lại để xem lại nguyên trạng về sau:

```powershell
curl.exe -s http://127.0.0.1:8000/v1/search-sessions/<query_id>
```

## Bước 5 — xuất CSV nộp bài

Lấy nguyên mảng kết quả từ bước 4 rồi gửi lại. Chỉ **KIS, QA, TRAKE** nộp bài.

```powershell
# body: {"task":"TEXTUAL_KIS","kis":[ ...nguyên mảng kis từ bước 4... ]}
curl.exe -s -X POST http://127.0.0.1:8000/v1/submissions/build `
  -H "Content-Type: application/json" --data-binary "@$env:TEMP\sub.json"
```

Đầu ra thật, đúng format BTC:

```
KIS    L21_V001,2130
QA     L21_V001,4157,"14,5 tỷ đồng"
TRAKE  L21_V001,5760,5889,6279,6933
```

Sau khi sắp xếp lại bằng tay, chạy `/v1/submissions/validate` với cùng body để
kiểm biên frame và dòng trùng trước khi nộp.

**AVS không nộp bài** — nó là task đánh giá nội bộ, `/v1/submissions/build`
trả 422 có chủ đích. Kết quả AVS lấy thẳng từ `/v1/search/avs`, đã là danh
sách segment có thứ hạng kèm `segment_id`, biên đoạn và `relevance_grade`.

---

## Những gì nên trông đợi khi duyệt tay

Số đo trên 36 truy vấn KIS / 36 QA / 24 TRAKE / 24 AVS, dữ liệu 3 video:

| task | tự động | trần khi bạn duyệt |
|---|---|---|
| KIS | R@1 0.583 | **R@5 1.000** — luôn thấy trong 5 dòng đầu |
| QA | evidence R@1 0.583 | evidence R@20 0.861 |
| TRAKE | mean R-score 0.183 | video đúng 0.542 (chỉ 0.313 trên video lạ) |
| AVS | nDCG 0.598 | event_coverage 0.841 |

**TRAKE là chỗ bạn phải để mắt nhất.** `complete_chain_rate = 0.000` — chưa
truy vấn nào ra chuỗi hoàn chỉnh, và trên video chưa từng tinh chỉnh thì chọn
đúng video chỉ 31%. Đổi lại `video_recall@3 = 0.833`, nên nếu UI cho xem top-3
video thì gần như luôn có video đúng trong đó.

**QA là chỗ duy nhất bạn thật sự cần tới 20 dòng** — evidence R@5 chỉ 0.667
trong khi R@20 là 0.861.

### R@20 sẽ vỡ khi corpus lớn lên

Ngoại suy từ phân bố hạng gold đo được (tệ nhất = 5), theo mô hình "số kẻ chen
trên gold tỉ lệ với kích thước corpus":

| số video | R@5 | R@20 |
|---|---|---|
| 3 (hiện tại) | 1.000 | 1.000 |
| 10 | 0.750 | 1.000 |
| **14** | — | **bắt đầu tụt** |
| 20 | 0.583 | 0.833 |
| 30 | 0.583 | 0.750 |
| 100 | 0.583 | 0.583 |

Ngoại suy từ MỘT điểm đo, nên tin hướng chứ đừng tin con số. Nhưng nếu dataset
thi trên ~14 video thì phải đo lại thật bằng video distractor, không suy diễn.

Ba nhánh luôn rỗng và đó là **đúng như dữ liệu hiện có**, không phải hỏng:
`bm25_action` (nhãn tiếng Anh, 65% là `standing`), `bm25_ocr` (truy vấn hiếm
khi chứa chữ trên màn hình), `color_search` (0/855 keyframe có dữ liệu màu).
Xem `docs/27` §A2–A3.

---

## Dừng server

```powershell
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force }
```
