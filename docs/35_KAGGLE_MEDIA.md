# 35 — Tải keyframe, video và objects

Pack thi đấu (`docs/34`) mang caption, ASR và vector nhưng **không kèm một file
ảnh nào**. Đây là bước lấy ảnh về.

Có hai nguồn. **Dùng mirror của ban tổ chức** (§0); Kaggle chỉ là dự phòng và có
những cái bẫy ghi ở §4. Đã chạy thật cả hai ngày **2026-08-17**, kể cả phần hỏng.

---

## 0. Mirror ban tổ chức — đường nên đi

Link nằm trong CSV ban tổ chức phát ("Dữ liệu cho vòng Sơ Tuyển AIC 2026 -
Batch1.csv", hai cột `Filenames` / `Download link`), trỏ tới
`https://aic-data.ledo.io.vn/`. Đã kiểm cả 32 link: HTTP 200, đều hỗ trợ
`Range`, đo được **29,5 MB/s** và không bị chặn tốc độ.

| Nhóm | Dung lượng | Ghi chú |
|---|---:|---|
| `Keyframes_L21..L30` (14 file) | **28,69 GB** | ảnh, tách riêng khỏi video |
| `Videos_L**_a` (14 file) | 77,29 GB | chỉ cần cho xem lại / TRAKE |
| `objects-aic25-b1.zip` | 0,60 GB | **178.195 file phát hiện vật thể Open Images** |
| `map-keyframes`, `media-info`, `clip-features-32` | 0,16 GB | |

```powershell
$csv  = "C:\Users\ASUS\Downloads\Copy of Dữ liệu cho vòng Sơ Tuyển AIC 2026 - Batch1.csv"
$pack = "D:\Sinh viên CNhan\download\AIC2026_competition_clean_v3 (1).zip"

# Anh keyframe — 28,69 GB, tai + giai nen + doi ten trong mot lenh
python -m scripts.fetch_aic_data --what keyframes --pack $pack --csv $csv

# Thu mot batch nho truoc cho chac
python -m scripts.fetch_aic_data --what keyframes --batch L23 --pack $pack --csv $csv

# Phat hien vat the (nhanh object_search dang rong)
python -m scripts.fetch_aic_data --what objects --pack $pack --csv $csv

# Video, chon loc
python -m scripts.fetch_aic_data --what videos --batch L21 --csv $csv
```

Script xoá `.zip` ngay sau khi giải nén xong từng file để không tốn gấp đôi chỗ
(`--keep-zip` nếu muốn giữ). Đứt giữa chừng thì chạy lại — `Range` tính từ số
byte đã ghi, và ảnh đã có trên đĩa thì bỏ qua.

Đo trên batch L23: tải + giải nén + đổi tên **2.326/2.326 ảnh, 0 thiếu, 32 giây**.

So với Kaggle: 28,69 GB thay vì 106,13 GB cho cùng ngần ấy ảnh, vì Kaggle gói
chung cả video vào một kho không tách được.

`media-info-aic25-b1.zip` KHÔNG chứa width/height như tưởng — nó là metadata
YouTube (title, description, keywords, length, channel). Vẫn đáng giữ: tiêu đề
và mô tả video là văn bản tìm kiếm được mà hệ chưa hề dùng.

---

## 1. Khoá Kaggle (chỉ cần cho đường dự phòng)

`kaggle.com` → ảnh đại diện → **Settings** → mục **API** → **Create New Token**.
Trình duyệt tải về `kaggle.json`. Mở nó ra, chép hai giá trị vào file env đang
dùng (`.env.fpt.local`), cùng chỗ với mọi khoá khác của dự án:

```ini
KAGGLE_USERNAME=<truong "username" trong kaggle.json — KHONG phai email>
KAGGLE_KEY=<truong "key">
```

Rồi trỏ `AIC_ENV_FILE` vào file đó khi chạy:

```powershell
$env:AIC_ENV_FILE = ".env.fpt.local"
```

Script đọc `AIC_ENV_FILE` bằng đúng cơ chế của `online/config.py`, nên khoá
Kaggle nằm chung một file đã được `.gitignore` chặn và `check_secret_leak.py`
chặn lần hai — thay vì rải thêm một file khoá thứ hai ở `~/.kaggle/`.

`.env.example` có sẵn hai dòng trống để điền. Ai quen `~/.kaggle/kaggle.json`
thì vẫn dùng được, script đọc nó khi không thấy biến môi trường.

Không cần cài gói `kaggle`.

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

## 3. Bố cục được DÒ, không đoán

Script thử lần lượt `001.jpg`, `0001.jpg`, `1.jpg`, `00001.jpg`, `000001.jpg`,
`001.jpeg` trên một ảnh thật, và thử các thư mục cha `Keyframes_L26`,
`Keyframes_L26_a`, … cho tới khi nhận về đúng một JPEG (kiểm bằng magic byte,
không tin `Content-Type`). Tìm được thì khoá lại cho cả lượt chạy.

Kết quả dò thật: **`Keyframes_L23/keyframes/<video_id>/{n:03d}.jpg`**, ảnh
1280×720 — trùng đúng mặc định `--assume-frame-size` của
`import_competition_pack.py`.

## 4. ⚠️ Endpoint tải-từng-file có hạn ngạch rất chặt

**Đo được trên chính tài khoản này: tải được 112 ảnh rồi Kaggle trả 404 cho
MỌI đường dẫn, kể cả đường vừa tải xong. Không hồi phục sau 5 phút.**

404 chứ không phải 429, nên nhìn hệt như "sai đường dẫn" — mất khá lâu mới phân
biệt được. Bằng chứng nó là hạn ngạch chứ không phải lỗi cấu hình: cùng lúc đó
`GET /api/v1/datasets/list` vẫn trả 200 (khoá còn hợp lệ) và endpoint tải-cả-bộ
vẫn trả 206.

Nghĩa là **176.707 ảnh qua đường tải-từng-file là không khả thi**. Đường đó vẫn
giữ trong script vì nó tiện cho vài chục ảnh (thử một video, vá vài ảnh thiếu),
nhưng cho cả bộ thì phải đi đường kho.

### Đường khả thi: tải một lần cả kho rồi giải nén tại chỗ

```powershell
$env:AIC_ENV_FILE = ".env.fpt.local"

# 1. Tai nguyen kho (106,13 GB). Dut giua chung thi chay lai lenh nay — noi tiep.
python -m scripts.fetch_kaggle_media --download-archive D:\aic\aic-nam-thang-ay.zip

# 2. Giai nen + doi ten. KHONG can mang.
python -m scripts.fetch_kaggle_media --archive D:\aic\aic-nam-thang-ay.zip `
    --what keyframes --pack "D:\...\AIC2026_competition_clean_v3 (1).zip"

# 3. Video (tuy chon, nang) — chi nhung video can xem lai
python -m scripts.fetch_kaggle_media --archive D:\aic\aic-nam-thang-ay.zip `
    --what videos --batch L21
```

Bước 1 tính là MỘT lần tải nên không dính hạn ngạch, và hỗ trợ `Range` nên nối
tiếp được. Bước 2 đọc thẳng trong `.zip`, chỉ ghi ra những ảnh export cần.

Chỗ trên đĩa: 106 GB cho kho + ~32,5 GB ảnh giải nén. Xong bước 2 thì xoá kho
được — nhưng giữ lại thì lần sau khỏi tải.

Nếu chỉ muốn xem UI chạy chứ chưa cần cả bộ, dùng đường từng-file cho một video:

```powershell
python -m scripts.fetch_kaggle_media --what keyframes --video L23_V001 `
    --pack "D:\...\AIC2026_competition_clean_v3 (1).zip" --workers 4
```

## 5. Xác nhận

Đếm số file trong thư mục **không phải** phép kiểm đúng — export chỉ dùng một
tập con, và cái cần bắt là ảnh nằm SAI TÊN chứ không phải thiếu ảnh:

```powershell
python -m scripts.fetch_kaggle_media --verify --export storage/exports_competition
```

Nó đối chiếu từng `image_path` mà export đang trỏ tới với file thật trên đĩa, và
liệt kê video nào thiếu bao nhiêu.

## 6. Đã kiểm được tới đâu

| Phần | Trạng thái |
|---|---|
| Bảng đổi tên | **Đã chạy thật** — sinh lại đúng tuyệt đối 855 tên file của L21_V001/V002/V003 (307/262/286) |
| Ca chỉ số không liên tục | Có test |
| Dò bố cục trên Kaggle | **Đã chạy thật** — ra `Keyframes_L23/keyframes/<id>/{n:03d}.jpg` |
| Tải từng file | **Đã chạy thật** — 112 ảnh về đĩa, `--verify` xác nhận **112/112 nằm đúng chỗ export trỏ tới**; rồi dính hạn ngạch (§4) |
| Giải nén từ kho (`--archive`) | **Đã chạy thật** trên kho dựng lại từ 855 ảnh có sẵn — ra đúng 855 tên VÀ nội dung giống byte-for-byte |
| `--download-archive` 106 GB | **Bỏ dở có chủ ý** — mirror §0 lấy cùng ngần ấy ảnh với 28,69 GB |
| Mirror ban tổ chức | **Đã chạy thật** — 32/32 link 200, batch L23 ra 2.326/2.326 ảnh, 0 thiếu |
| `--verify` | Đã chạy |

Chỉ dòng áp chót là chưa chạy trọn vẹn. Phần logic của nó (nối tiếp theo kích
thước file hiện có) không phụ thuộc quy mô.

## 7. Dung lượng

| | |
|---|---:|
| Kho `.zip` cả dataset | 106,13 GB |
| Ảnh keyframe sau giải nén | ~32,5 GB (đo: 193 KB/ảnh × 176.707) |
| Ảnh cho một batch nhỏ (L23) | ~450 MB |

Ổ D còn ~500 GB nên chỗ không phải vấn đề; băng thông mới là.
