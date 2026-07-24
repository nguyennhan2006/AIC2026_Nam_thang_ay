# AIC Local Search UI

Giao diện Streamlit cho Local Search Engine v2:

- tìm scene/frame bằng SQLite FTS5 + BM25;
- tìm thị giác bằng OpenCLIP + FAISS;
- gộp nhiều nhánh bằng RRF;
- xem keyframe, scene clip, caption, OCR, ASR, tags và điểm từng nhánh;
- tìm chuỗi scene đúng thứ tự thời gian.

## 1. Mở project trong VS Code

Giải nén thư mục này vào:

```text
C:\Users\LOQ\Documents\AIC2026\aic_search_ui
```

Trong VS Code, chọn:

```text
Ctrl + Shift + P
→ Python: Select Interpreter
→ aic-search
```

## 2. Cài giao diện

Mở Terminal trong VS Code:

```powershell
conda activate aic-search
cd "C:\Users\LOQ\Documents\AIC2026\aic_search_ui"
python -m pip install -r requirements.txt
```

Engine `aic_local_search` phải được cài trong cùng môi trường:

```powershell
python -c "import aic_local_search; print(aic_local_search.__version__)"
```

Kết quả mong đợi:

```text
0.2.0
```

## 3. Chạy

Cách nhanh:

```powershell
.\run-ui.ps1
```

Hoặc:

```powershell
python -m streamlit run app.py
```

Trình duyệt sẽ mở:

```text
http://localhost:8501
```

Bạn cũng có thể nhấn `F5` trong VS Code và chọn `AIC Search UI`.

## 4. Chọn dữ liệu

Trong thanh bên:

```text
Index directory:
C:\Users\LOQ\Documents\AIC2026\08_local_search_index_050607

Asset root:
C:\Users\LOQ\Documents\AIC2026\component_outputs
```

`Index directory` bắt buộc có:

```text
index_manifest.json
aic_search.db
scene_hnsw.faiss
frame_hnsw.faiss
```

`Asset root` có thể chứa trực tiếp các ZIP output 01 và 04. UI tự đọc keyframe
và scene clip bên trong ZIP, không bắt buộc giải nén.

## 5. Hai chế độ search

### BM25 / FTS5

Chạy caption, OCR, ASR, tags và event. Không cần OpenCLIP:

```text
Công viên địa chất Lạng Sơn được UNESCO công nhận
```

### Hybrid BM25 + FAISS

Chạy các nhánh văn bản và hai nhánh vector. OpenCLIP biến câu mô tả thành vector
512 chiều tương thích với index:

```text
Query:
bản tin về kiểm tra thịt heo có dấu kiểm dịch

Visual query:
a television news report showing veterinary inspection marks on pork
```

Lần đầu OpenCLIP có thể tải trọng số `ViT-B-32`.

## 6. Nếu UI không tìm thấy ảnh hoặc clip

Search vẫn đúng; chỉ phần preview media chưa resolve được. Kiểm tra `Asset root`
có chứa:

```text
01_keyframes_output.zip
04_scene_clips_output.zip
```

Nếu ZIP dùng tên có số phiên bản như `01_keyframes_output(3).zip`, UI vẫn đọc
được.

## 7. Nếu Hybrid chạy bằng CPU

Kiểm tra:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

`False` vẫn chạy được, chỉ chậm hơn. Bạn có thể chọn `cpu` trong thanh bên hoặc
dùng chế độ BM25 trước.
