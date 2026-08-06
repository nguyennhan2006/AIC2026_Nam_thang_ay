# 24 — Kiểm định luồng search

Làm trước khi đầu tư vào rerank và finetune, theo đúng nguyên tắc đã áp cả đợt:
không tối ưu chồng lên một nền đang hỏng.

Mọi kết luận dưới đây đến từ **truy vết một truy vấn thật qua từng tầng**, không
từ đọc code. Cấu hình: `storage/exports_multivideo` (765 scene, 3 video), truy
vấn `V001_KIS_E02`.

---

## 1. Mười nhánh đang chạy

| branch_id | execution_id | index trên | truy vấn nhận được |
|---|---|---|---|
| `dense_visual` | `.raw` | vector CLIP ảnh | bản dịch tiếng Anh |
| `bm25_caption` | `.expanded` | caption scene | cả câu |
| `bm25_ocr` | `.raw` | chữ trên màn hình | cả câu |
| `bm25_asr` | `.raw` | lời nói đã gỡ băng | cả câu |
| `bm25_keyword` | `.expanded` | nhãn object 2–3 từ | **cả câu** ← lệch hạt |
| `ocr_fuzzy` | `.raw` | chữ, khớp mờ | cả câu |
| `bm25_object` | `.raw` | nhãn object | cả câu |
| `bm25_action` | `.raw` | action tag | cả câu |
| `color_search` | `.raw` | màu chủ đạo | cả câu |
| `event_search` | `.raw` | event đã gom nhóm | cả câu |

Số liệu một truy vấn thật:

```
dense_visual.raw       success  n=100  1670ms
bm25_caption.expanded  success  n=100     7ms
bm25_ocr.raw           empty    n=  0     1ms   <- đúng: query không có cue chữ
bm25_asr.raw           success  n=100     3ms
bm25_keyword.expanded  success  n=100     3ms
ocr_fuzzy.raw          success  n= 33  1680ms   <- chậm gấp 240 lần BM25
bm25_object.raw        success  n=100     2ms
bm25_action.raw        empty    n=  0     1ms
color_search.raw       empty    n=  0     0ms
event_search.raw       success  n=100     1ms   <- 100/100 đều là video gold
```

---

## 2. LỖI NẶNG: `agreement` lớn gấp 5–8 lần thiết kế

**Đây là phát hiện quan trọng nhất của lần kiểm định.**

`fuse_candidates` ghi thông tin nhánh vào **`payload`** (`matched_branches`,
`component_scores`), KHÔNG ghi vào trường `Candidate.branch_scores`. Nhưng
`ScoreNormalizers.from_pool()` lại đọc đúng trường đó:

```python
# online/services/normalizers.py:49
ceiling = max((len(candidate.branch_scores) for candidate in candidates), default=1) or 1
```

Đo được: `branch_scores` rỗng ở **100/100 candidate** ⇒ `branch_ceiling` luôn
bằng **1**.

Hệ quả ở `online/services/kis.py:187`:

```python
agreement = len(hit.matched_branches) / branch_ceiling
```

| | thiết kế | thực tế |
|---|---|---|
| `branch_ceiling` | 8 (số nhánh nhiều nhất cùng thấy một candidate) | **1** |
| `agreement` | 1.0, 0.62, 0.88, 0.62, 0.75 | **8.0, 5.0, 7.0, 5.0, 6.0** |

Trong công thức chấm KIS:

```
total = 1.00 × retrieval_norm   (0–1)
      + 0.60 × must_coverage    -> tối đa 0.60
      + 0.30 × safe_score       -> tối đa 0.30
      + 0.25 × rare_score       -> tối đa 0.25
      + 0.15 × agreement        -> ĐÁNG LẼ tối đa 0.15
      + 0.10 × nice_coverage    -> tối đa 0.10
```

`0.15 × agreement` thực tế cho **0.60–1.20**, biến nó thành **số hạng lớn nhất**,
vượt cả `retrieval_weight`. Nghĩa là xếp hạng KIS đang bị quyết định chủ yếu bởi
**đếm xem bao nhiêu nhánh cùng khớp**, chứ không phải độ liên quan.

Điều này giải thích trực tiếp `top1_pairwise_accuracy = 0.545` — chọn giữa hai
ứng viên sát nhau chỉ hơn tung đồng xu, vì thứ phân định là số phiếu nhánh.

**Đường fallback lại ĐÚNG.** Khi `normalizers is None` (test gọi trực tiếp),
`kis.py:173` tự tính `branch_ceiling` từ `hit.matched_branches` và ra giá trị
đúng. Nên test không bắt được: chúng đi đường đúng, production đi đường sai.

> Bài học lặp lại lần thứ tư trong đợt: **một trường không được ai ghi vào vẫn
> đọc ra giá trị hợp lệ** (`0`), và `or 1` biến nó thành một hằng số trông vô
> hại. Không có gì fail; chỉ có công thức chạy sai suốt.

---

## 3. `event_search` là nhánh CHỈ trả về L21_V001

`events.jsonl` trong export đa video có **69 event, toàn bộ của L21_V001**;
V002/V003 bằng 0 — file được chép nguyên từ export gốc lúc dựng distractor và
chưa bao giờ được sinh cho video mới.

Trong truy vết, nhánh này trả 100 candidate **đều thuộc video gold**. Trông như
hiệu năng xuất sắc, thực chất là nó không có gì khác để trả.

Đây là lỗ hổng còn sót của việc cân bằng dữ liệu (PR-4C): tôi đã cân
caption/keyword/object/OCR/ASR nhưng bỏ quên events. Mọi phép đo đa video vẫn
đang có **một nhánh thiên vị**, và nó lại là nhánh cho `n=100` với chi phí 1ms.

---

## 4. `ocr_fuzzy` chậm gấp 240 lần BM25 cùng modality

`1680ms` so với `1–7ms` của các nhánh BM25. Nó khớp mờ trên toàn bộ 765 scene.

Hai hệ quả:

- Nó là một nửa nguyên nhân của `BRANCH-TIMEOUT-01`: `dense_visual` và
  `ocr_fuzzy` bám sát nhau từng mili-giây vì cùng tranh CPU.
- Đóng góp của nó **chưa bao giờ được đo riêng**. Nó trùng modality với
  `bm25_ocr`, nên rất có thể phần lớn giá trị đã được nhánh kia cung cấp với
  chi phí 1ms.

Phép đo công bằng ở PR-4B cho một manh mối: tắt 8 nhánh (gồm `ocr_fuzzy`) làm
TRAKE **tăng** 0.263 -> 0.287.

---

## 5. Lệch hạt ở nhánh keyword

Phía tài liệu là nhãn object 2–3 từ (`"người dẫn chương trình"`); phía truy vấn
là cả câu kể chuyện. Đã dựng bộ tách (`online/services/keyword_extraction.py`)
và đo riêng được nhờ `sparse_terms` của gold:

| cách tách | token_recall | phrase_hit |
|---|---|---|
| cắt 4 từ đầu mệnh đề | 0.374 | 0.333 |
| chọn cụm IDF cao nhất | 0.244 | 0.153 |
| **giữ toàn bộ token nội dung** | **0.524** | **0.514** |

Nhưng đo đầu-cuối thì **không đổi gì** (KIS y hệt, TRAKE −0.026, AVS +0.025 —
đều trong biên nhiễu). Mặc định vẫn TẮT.

Nghi ngờ đáng theo đuổi: nhánh keyword có thể đang bị `agreement` ở mục 2 lấn
át, nên cải thiện nó không thấy được. **Đo lại sau khi sửa mục 2.**

---

## 6. OCR: 84% là lớp phủ, và chúng ở vị trí cố định

Thống kê bbox thật của 2028 chuỗi:

| vùng | số chuỗi | độ dài TB | là gì |
|---|---|---|---|
| `y~0.1, x>0.75` | 762 | 1.6 từ | logo kênh + đồng hồ |
| `y>0.85` | 963 | 9.3 từ | thanh chữ chạy |
| phần giữa | 323 | 4.8 từ | **nội dung thật** |

Chữ chạy là loại hại nhất: nó bơm nội dung của tin B vào khung hình tin A.

Đã xử lý hai đường: lọc theo vị trí lúc dựng index (cho phần có bbox thật, miễn
phí, giữ nguyên dữ liệu), và tô đen hai vùng đó trước khi OCR lại (cho phần bù
không có bbox). Sau khi cắt, chỉ **62/181** frame còn chữ nội dung — 119 frame
thật sự chỉ có lớp phủ.

---

## 7. Việc phải làm TRƯỚC khi đụng rerank/finetune

Theo thứ tự, vì các mục sau phụ thuộc mục trước:

1. **Sửa `branch_ceiling`** (mục 2). Đây là lỗi công thức, không phải lựa chọn
   thiết kế. Nó bóp méo xếp hạng KIS ở đường production và làm mọi thí nghiệm
   ranking từ trước tới nay khó diễn giải.
2. **Sinh events cho V002/V003** hoặc tắt `event_search` trong mọi phép đo đa
   video (mục 3). Không làm thì bàn cân vẫn nghiêng.
3. **Đo riêng đóng góp của `ocr_fuzzy`** (mục 4) — nhiều khả năng tắt được, vừa
   nhanh hơn vừa không mất gì.
4. **Chạy lại baseline** sau ba mục trên, rồi mới so.
5. **Đo lại bộ tách keyword** (mục 5) trên nền đã sửa.

Chỉ sau đó mới nên bàn tới rerank tinh và finetune: cả hai đều là tối ưu ở tầng
xếp hạng, mà tầng đó hiện đang bị một hằng số sai chi phối.
