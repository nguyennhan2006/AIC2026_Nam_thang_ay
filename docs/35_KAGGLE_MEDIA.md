# 35 — Tải keyframe và video từ Kaggle

Nguồn: [`trongnhantran25/aic-nam-thang-ay`](https://www.kaggle.com/datasets/trongnhantran25/aic-nam-thang-ay)
— 115,75 GB, `Keyframes_L21..L30` (L26 chẻ làm 5 phần) và `Videos_L**_a/video/*.mp4`.

Pack thi đấu (`docs/34`) mang caption, ASR và vector nhưng **không kèm một file
ảnh nào**. Đây là bước lấy ảnh về.

---

## 1. Khoá Kaggle

`kaggle.com` → ảnh đại diện → **Settings** → mục **API** → **Create New Token**.
Trình duyệt tải về `kaggle.json`. Chép vào:

```
C:\Users\ASUS\.kaggle\kaggle.json
```

Thư mục `.kaggle` chưa có thì tạo:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

Hoặc không dùng file, đặt biến môi trường:

```powershell
$env:KAGGLE_USERNAME = "<username>"
$env:KAGGLE_KEY      = "<key>"
```

Script **không** cần gói `kaggle` — nó gọi thẳng REST API, chỉ mượn đúng định
dạng khoá đó. `kaggle.json` không được commit (`.gitignore` đã chặn `*.json`
trong thư mục người dùng vì nó nằm ngoài repo, nhưng vẫn đừng chép vào repo).

## 2. Vì sao không chép thẳng thư mục Kaggle

Đây là toàn bộ lý do script này tồn tại.

```
Kaggle  Keyframes_L21/keyframes/L21_V001/002.jpg     ← số thứ tự keyframe
export  processed/keyframes/L21_V001/frame_000090.jpg ← frame index
```

Chép thẳng là 176.707 ảnh nằm sai tên. Hệ **không báo lỗi**: `/v1/media` trả
404, UI hiện ô trống, còn dense và BM25 vẫn chạy bình thường vì chúng không đụng
tới ảnh. Phải tra hết log HTTP mới thấy.

Bảng đổi tên nằm ở cột `source_keyframe_index` → `frame_idx` trong
`canonical/keyframe_scene_mapping.csv` của pack — cùng bảng mà
`scripts/import_competition_pack.py` đã dùng để sinh `image_path`. Vì vậy
`--pack` là tham số bắt buộc.

**Ghép theo SỐ đọc từ tên file, không bao giờ theo thứ tự.** 192/873 video có
`source_keyframe_index` không liên tục (pack loại vài frame vì conflict/orphan),
nên thư mục Kaggle có nhiều ảnh hơn export cần:

| Video | file trên Kaggle | export dùng | chỉ số bị loại |
|---|---:|---:|---|
| L21_V006 | 257 | 256 | 2 |
| L21_V007 | 209 | 208 | 2 |
| L21_V012 | 225 | 224 | 2 |

Ghép theo thứ tự thì mọi ảnh sau chỗ khuyết lệch một nấc — ảnh vẫn hiện ra bình
thường, chỉ là **sai ảnh**, nên không cách nào phát hiện từ giao diện.
`tests/test_fetch_kaggle_media.py` khoá đúng ca này lại.

## 3. Lệnh

Dò bố cục và in kế hoạch trước, không tải gì:

```powershell
python -m scripts.fetch_kaggle_media --what keyframes --batch L21 --limit-videos 2 `
    --pack "D:\Sinh viên CNhan\download\AIC2026_competition_clean_v3 (1).zip" --dry-run
```

Tải một batch (bắt đầu từ L23 — nhỏ nhất, 25 video / 2.326 ảnh):

```powershell
python -m scripts.fetch_kaggle_media --what keyframes --batch L23 `
    --pack "D:\Sinh viên CNhan\download\AIC2026_competition_clean_v3 (1).zip" --workers 8
```

Toàn bộ 176.707 ảnh:

```powershell
python -m scripts.fetch_kaggle_media --what keyframes `
    --pack "D:\Sinh viên CNhan\download\AIC2026_competition_clean_v3 (1).zip" --workers 12
```

Video (chỉ cần cho xem lại và TRAKE refinement — nặng, tải chọn lọc):

```powershell
python -m scripts.fetch_kaggle_media --what videos --batch L21 `
    --pack "D:\Sinh viên CNhan\download\AIC2026_competition_clean_v3 (1).zip"
```

Chạy lại bao nhiêu lần cũng được: ảnh đã có trên đĩa thì bỏ qua, nên đứt giữa
chừng chỉ cần chạy lại.

## 4. Xác nhận

Đếm số file trong thư mục **không phải** phép kiểm đúng — export chỉ dùng một
tập con, và cái cần bắt là ảnh nằm SAI TÊN chứ không phải thiếu ảnh:

```powershell
python -m scripts.fetch_kaggle_media --verify --export storage/exports_competition
```

Nó đối chiếu từng `image_path` mà export đang trỏ tới với file thật trên đĩa, và
liệt kê video nào thiếu bao nhiêu.

## 5. Bố cục được DÒ, không đoán

Script không giả định trước tên file. Nó thử lần lượt `001.jpg`, `0001.jpg`,
`1.jpg`, `00001.jpg`, `000001.jpg`, `001.jpeg` trên một ảnh thật, và thử các thư
mục cha `Keyframes_L26`, `Keyframes_L26_a`, … cho tới khi nhận về đúng một JPEG
(kiểm bằng magic byte, không tin `Content-Type`). Tìm được thì khoá lại cho cả
lượt chạy. Không dạng nào chạy thì **dừng** và in ra đã thử những gì.

Bằng chứng gián tiếp cho quy ước đánh số từ 1: số file mà Data Explorer hiển thị
khớp `max(source_keyframe_index)` của pack ở **15/15 video** đối chiếu được.

## 6. Đã kiểm được tới đâu

| Phần | Trạng thái |
|---|---|
| Bảng đổi tên | **Đã kiểm trên dữ liệu thật** — sinh lại đúng tuyệt đối 855 tên file của L21_V001/V002/V003 (307/262/286) |
| Ca chỉ số không liên tục | Có test (`tests/test_fetch_kaggle_media.py`) |
| Thiếu khoá → báo lỗi đọc được | Đã chạy |
| Dò bố cục, tải, `--verify` | **Chưa chạy được** — máy viết script không có khoá Kaggle |

Phần cuối là phần duy nhất chưa chạy thật. Chạy `--dry-run` trước: nếu nó in ra
được bố cục và vài dòng ánh xạ thì đường mạng và khoá đều đúng.

## 7. Dung lượng và thời gian

Ước lượng, chưa đo: 176.707 ảnh × ~150 KB ≈ **26 GB**. Mỗi ảnh là một lượt HTTP
riêng nên tốc độ phụ thuộc độ trễ chứ không phải băng thông — với 12 luồng,
khoảng vài giờ. Video là phần còn lại của 115,75 GB.

Tải một batch trước, chạy `--verify`, xem UI hiện ảnh đúng chưa, rồi mới thả cả
bộ. Ổ D còn ~500 GB nên chỗ không phải vấn đề.
