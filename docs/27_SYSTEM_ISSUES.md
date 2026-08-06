# 27 — Tổng hợp vấn đề toàn hệ thống

Cập nhật 2026-08-06, sau đợt sửa `docs/26`. Mốc: `c623f63` + working tree.
Dữ liệu: 765 scene / 855 keyframe / 3 video, gold 120 truy vấn.

Baseline hiện tại (4 task, metric đã sửa, cap AVS = 20):

| task | chỉ số chính | giá trị |
|---|---|---|
| KIS | R@1 / R@20 / MRR | 0.583 / **1.000** / 0.725 |
| QA | evidence R@1 / R@20 / answer / joint | 0.583 / 0.861 / 0.583 / 0.417 |
| TRAKE | video@1 / mean R-score / complete_chain | 0.542 / 0.183 / **0.000** |
| AVS | nDCG@100 / event_coverage | 0.598 / 0.841 |

Ký hiệu mức: 🔴 chặn đường · 🟠 mất điểm đo được · 🟡 rủi ro chưa hiện · ⚪ nợ kỹ thuật

---

## 0. Cấu hình đã chốt — mọi dòng đều có số đo

`outputs/evaluation/LOCKED_BASELINE.json`. Chạy lại bằng lệnh trong `docs/28`.

| biến | giá trị | bằng chứng |
|---|---|---|
| `AIC_ENABLE_OCR_FUZZY` | `false` | bật → KIS R@1 0.583→0.500 |
| `AIC_ENABLE_EVENT_SEARCH` | `false` | cùng phép đo trên |
| `AIC_RERANK_VLM_ENABLED` | `false` | **0 thay đổi chỉ số, tiêu 94% lời gọi API** |
| `AIC_AVS_MAX_RESULTS_PER_VIDEO` | `20` | nDCG 0.545 vs 0.422 ở cap=3 |
| `AIC_AVS_GRADE_MODE` | `semantic_or_lexical` | hoà 0.598 với `hard_gate`/`no_gate`; `soft` 0.543 |
| `AIC_ENABLE_LLM_EXPANSION` | `false` | FPT-WIRE-01: giảm KIS MRR và AVS nDCG |
| `AIC_ENABLE_QUERY_TRANSLATION` | `true` | cải thiện lớn nhất từng đo |
| `AIC_ENABLE_KEYWORD_EXTRACTION` | `false` | DROP, đo lại sau khi sửa scoring |
| `KisConfig` (trọng số) | mặc định | WEIGHT-01: mọi biến thể overfit V001 |

Hai dòng đầu và dòng VLM **mới được sửa 2026-08-06** — trước đó cấu hình server
khác hẳn cấu hình được đo.

---

## A. Tầng dữ liệu

### 🟠 A1 — Lọc lớp phủ OCR áp dụng KHÔNG nhất quán

Hai đường đọc hai phiên bản OCR khác nhau:

```
nhánh bm25_ocr        lọc theo vị trí, bỏ 74% instance   -> 574 instance
KisProcessor          đọc toàn bộ document.ocr_texts     -> 2246 instance
AVS grade             đọc pack.rerank_text()             -> 2246 instance
```

Trong 12 266 từ OCR mà `KisProcessor` chấm điểm, **9 925 từ (81%) là lớp phủ** —
tên kênh, ticker, "60 giây". Chúng chiếm 11% tổng văn bản scene và làm loãng cả
`must_coverage` lẫn `rare_hits`.

Bộ lọc đã có sẵn (`_in_overlay_band` trong `online/adapters/bm25.py`), chỉ là
chưa được dùng ở tầng repository. Đưa nó lên chỗ dựng `SceneDocument` thì cả ba
đường đọc cùng một dữ liệu.

**Nghiên cứu tiếp:** đo riêng tác động lên KIS và AVS. Dự đoán: nhỏ nhưng dương,
vì lớp phủ giống nhau ở mọi scene nên nó là nhiễu nền chứ không thiên vị.

### 🟠 A2 — `action_tags` bằng tiếng Anh, và 65% là một nhãn duy nhất

```
591 nhãn / 855 keyframe, chỉ 435/765 scene có
standing 382 (65%) · sitting 35 · talking 27 · walking 25 · driving 22
```

Truy vấn là tiếng Việt, nhãn là tiếng Anh — nên `bm25_action` **rỗng 120/120
truy vấn**. Nhánh này hiện không đóng góp gì.

Kể cả dịch sang tiếng Việt, `standing` chiếm 65% thì giá trị phân biệt cũng gần
bằng 0.

**Nghiên cứu tiếp:** hoặc sinh lại action tag bằng tiếng Việt với prompt đòi
hành động CỤ THỂ (không nhận "standing"), hoặc gỡ hẳn nhánh. Không nên giữ một
nhánh rỗng.

### 🟠 A3 — Không có dữ liệu màu, nhánh `color_search` rỗng hoàn toàn

`0/855` keyframe có trường `color`. Nhánh rỗng 120/120 truy vấn.

Đây là mất mát thật: truy vấn KIS đầy mô tả màu (`biển màu vàng đỏ`, `áo đen`,
`bộ đồ sọc xanh trắng`, `sườn đồi đỏ rực`).

**Nghiên cứu tiếp:** trích histogram màu là việc offline thuần CPU, không tốn
API. `scripts/caption_qwen3vl.py` đã có sẵn `extract_hsv_features()`.

### 🟡 A4 — Một keyframe mỗi scene: đủ cho metric hiện tại, rủi ro nếu luật thi khác

```
1.12 keyframe/scene — 697/765 scene có ĐÚNG 1 keyframe
độ dài scene p50 = 4.1s  ->  lấy mẫu ~0.24 fps
```

Đối chiếu với gold TRAKE: **101/110 bước hành động (92%) không chứa keyframe
nào** trong khoảng của chúng.

Nhưng eval chấm theo **cửa sổ dung sai 2–7 giây**, không đòi frame nằm trong
khoảng. Với cửa sổ đó, `frame_oracle_coverage = 0.845` — tức mật độ keyframe
KHÔNG phải nút thắt của chỉ số hiện tại.

**Rủi ro cần xác minh trước, không phải nghiên cứu:** nếu ban tổ chức đòi frame
nộp nằm trong khoảng hành động, TRAKE sẽ về gần 0 bất kể thuật toán. Đây là câu
hỏi về LUẬT, phải hỏi chứ không đo được.

### 🔴 A4b — V002 và V003 CHƯA TỪNG được tách cảnh

Ba video được dựng theo hai cách hoàn toàn khác nhau:

| video | scene | keyframe | kf/scene | dài p50 | dài max | nguồn |
|---|---|---|---|---|---|---|
| V001 | 217 | 307 | 1.41 | 4.6s | **32.2s** | `transnetv2_pytorch` — cắt cảnh thật |
| V002 | 262 | 262 | **1.00** | 4.0s | **7.0s** | `csv_keyframe_grid` |
| V003 | 286 | 286 | **1.00** | 4.1s | **7.0s** | `csv_keyframe_grid` |

`input/scene_manifest.jsonl` có 336 scene và **toàn bộ là V001**. V002/V003
không có manifest; scene của chúng do `scripts/build_distractor_export.py` suy
ra từ lưới keyframe trong `input/L21_V002.csv` / `L21_V003.csv`. Một "scene" ở
đó là cửa sổ ~4s quanh một keyframe lấy mẫu, **không phải một cú máy**.

**Không sửa được.** `storage/raw/videos/` chỉ có `L21_V001.mp4` và
`L16_V001.mp4`. V002/V003 chỉ có ảnh JPG, nên không có gì để chạy lại phát
hiện cắt cảnh.

**Điều này KHÔNG giải thích khoảng cách TRAKE**, dù nghe có vẻ hợp lý:

- KIS R@1 **giống hệt nhau 0.583 trên cả ba video** — chất lượng tách cảnh
  không ảnh hưởng KIS;
- cửa sổ dung sai TRAKE suy từ độ dài scene, nhưng chênh lệch chỉ 15%
  (2.30s ở V001 so với 2.00s ở V002) vì phần lớn đều chạm sàn 2 giây;
- `video_recall@1` là chọn VIDEO, không phụ thuộc độ mịn của scene.

Nên kết luận "TRAKE khớp riêng V001" vẫn đứng. Ghi lại đây vì đây là một khác
biệt hệ thống giữa các video mà mọi phép so V001↔V002/V003 phải tính đến, và
vì nó là trần cứng của A4 (1 keyframe mỗi scene) trên hai phần ba dữ liệu.

### 🟠 A4c — Toàn bộ tín hiệu chất lượng keyframe đều rỗng

```
quality.sharpness         None ở 307/855, THIẾU HẲN ở 548 keyframe còn lại
quality.brightness        None
quality.black_frame_ratio None
quality.contrast          None
quality.duplicate_score   None
selection_score           0/855
boundary_confidence_in    None ở cả 765 scene
transition_in             'unknown' ở cả 765 scene
```

Hệ quả cho `safe_frame`: `_quality()` trả 0.5 trung tính cho **mọi** keyframe,
và `_blur_penalty()` luôn trả 0.0. Hai trong năm thành phần của safe-frame chết
trên toàn bộ dữ liệu — nó rút gọn còn `semantic + centrality − boundary`.

Khớp với kết quả sweep: đặt `safe=0.0` gần như không đổi gì (`docs/26` §11).

Đáng chú ý: `input/scene_manifest.jsonl` CÓ trường `confidence` và
`boundary_repaired` cho V001, nhưng chúng không được mang vào export.

### 🟡 A5 — Chỉ 3 video, và một trong ba đã bị dùng để tinh chỉnh

Holdout thật chỉ còn **80 truy vấn** (V002 + V003). Chênh lệch giữa V001 và hai
video kia đủ lớn để thấy rõ đã khớp riêng V001:

| | V001 | V002 | V003 |
|---|---|---|---|
| TRAKE video@1 | **1.000** | 0.375 | 0.250 |
| TRAKE mean R | 0.287 | 0.179 | 0.081 |

Ngoài ra chưa có video **distractor** thật — video không chứa đáp án nào. Corpus
thi sẽ toàn distractor.

**Nghiên cứu tiếp:** thêm video distractor là cách rẻ nhất để mô phỏng corpus
lớn mà không cần gold mới. Đã có `scripts/build_distractor_export.py`.

### ⚪ A6 — Embedding chỉ ở tầng keyframe

`embedding_refs`: 855/855 keyframe, **0/765 scene**. Dense retrieval vì thế là
frame-level. Không sai, nhưng nghĩa là không có vector mức scene để so sánh
toàn cảnh.

---

## B. Tầng retrieval và chấm điểm

### 🔴 B1 — TRAKE Stage A: chọn sai video ở 2/3 số truy vấn holdout

```
video@1   0.542 tổng   —   1.000 V001, 0.375 V002, 0.250 V003
video@3   0.833
```

`video@3 = 0.833` so với `video@1 = 0.542` nghĩa là **đúng video thường nằm
trong top-3 nhưng không lên được hạng 1**. Đây là bài toán xếp hạng, không phải
recall.

Phân rã R-score: `0.542 (chọn video) × 0.399 (chọn frame, so với oracle) ×
0.845 (trần oracle) ≈ 0.183`.

**Đây là chỗ mất điểm lớn nhất còn lại.** 8/40 truy vấn chỉ mang về ~1.0 điểm.

**Nghiên cứu tiếp:** hai hệ số nhân đều còn dư địa. Ưu tiên chọn video vì
`video@3 = 0.833` cho thấy tín hiệu đã có sẵn.

### 🟢 B2 — Tầng chấm điểm KIS: trung tính theo R@1, nhưng CHỊU TẢI theo R@20

Kết luận cũ ở mục này — *"trung tính, ngừng đầu tư"* — **đúng theo R@1 và sai
theo mục tiêu thật**. Người dùng chốt 2026-08-06: ưu tiên là **có đáp án trong
20 kết quả đầu**, không phải hạng 1.

Với mục tiêu đó, chỉ số cần theo dõi là **hạng gold TỆ NHẤT**, vì đó là thứ
quyết định khi nào R@20 vỡ:

| cấu hình | R@1 | p50 | p90 | **hạng tệ nhất** |
|---|---|---|---|---|
| BASE | 0.583 | 1 | 5 | **8** |
| bỏ `safe` | 0.583 | 1 | 4 | **7** |
| bỏ `agreement` | 0.583 | 1 | 5 | 9 |
| chỉ retrieval (bỏ signature) | 0.583 | 1 | **3** | **16** |
| bỏ `must` | 0.556 | 1 | **3** | **20** |

Bỏ signature cho p90 TỐT hơn (3 so với 5) nhưng hạng tệ nhất **nhảy từ 8 lên
16–20**. Giá trị của tầng này nằm đúng ở chỗ kéo những truy vấn khó nhất lên —
mà "khó nhất" chính là cái quyết định R@20.

**Giữ nguyên, và đo bằng hạng tệ nhất từ giờ trở đi.** R@1 và MRR không phản
ánh mục tiêu.

### 🟢 B2b — Trần khử trùng mặc định đang GIÚP, đừng nới

`TASK_POLICIES[TEXTUAL_KIS] = {max_per_video: 5, max_per_event: 1}`. Mọi lượt
eval trước đây chạy `--max-per-video 0` (bỏ trần), tức **chưa từng đo đúng
hành vi production**. Đo lại:

| | R@1 | R@5 | MRR |
|---|---|---|---|
| mặc định (5/video) | 0.583 | **1.000** | **0.733** |
| bỏ trần | 0.583 | 0.917 | 0.725 |

Giả thuyết ban đầu của tôi — "trần cắt mất recall" — **sai**. Gộp scene gần
trùng của cùng một sự kiện làm top-5 chứa nhiều sự kiện khác nhau hơn. Với
cấu hình mặc định, đáp án nằm trong **5 dòng đầu ở 36/36 truy vấn**.

Hệ quả: `docs/12` §6 gợi ý nới `max_results_per_video` để "lấy đủ top_k" — lời
khuyên đó làm giảm chất lượng. Xem `docs/28`.

### 🟠 B3 — Negative constraint vô hiệu hoàn toàn (NEG-02)

`0/765` scene bị loại bởi bất kỳ constraint nào, kể cả hai constraint ĐÚNG.
`_NEGATION_RE` bắt tới dấu câu nên cụm dài 5–9 từ, mà điều kiện là "mọi từ đều
có mặt".

Hoãn vì thiếu chỗ đo: chỉ 5/120 truy vấn có mệnh đề phủ định ⇒ trần lợi ích 4%,
ngưỡng nhiễu 2.8%.

**Nghiên cứu tiếp:** khi quay lại, dùng **phạt mềm** chứ không lọc cứng — lọc
cứng xoá nhầm gold khi metadata thiếu, mà metadata đang thiếu thật (A2, A3).

### 🟠 B4 — Chỉ số QA có nhiễu ±1 truy vấn do LLM (QA-REPRO-01, đã đóng)

`AIC_QA_LLM_RANK_MODE=boost` cho LLM tác động **cả `evidence_rank` lẫn nội dung
đáp án**, nên nhiễu LLM đi thẳng vào cả bốn chỉ số QA.

Đo bằng ba lượt chạy liên tiếp không đổi một dòng code:

```
bốn chỉ số tổng:      trùng khít tuyệt đối cả ba lượt
predicted_answers:    2/36 truy vấn KHÁC nhau giữa lượt 1 và lượt 3
```

**Chỉ số tổng khớp không chứng minh được tính tất định.** Những khác biệt ở mức
truy vấn tình cờ không lật chỉ số nào. Tôi đã suýt kết luận ngược từ đúng ba
dòng tổng trùng nhau — ghi lại vì nó là cái bẫy sẽ lặp lại.

Toàn bộ chênh lệch giữa baseline và lượt sau giải thích được bằng số học: một
truy vấn đi từ hạng 1 sang hạng 2 làm MRR giảm đúng `(1 − 1/2)/36 = 0.0139`
(đo được 0.014), và một truy vấn khác đổi đáp án dòng đầu làm `joint_top1`
tăng 1.

**Hệ quả:** mọi delta QA dưới **±0.028** không kết luận được, kể cả khi hai
lượt trông giống hệt nhau ở mức tổng. Ba task còn lại (KIS, TRAKE, AVS) tái lập
chính xác vì không có LLM trong đường xếp hạng.

**Nghiên cứu tiếp:** nếu cần số QA ổn định để so cấu hình, hoặc tắt
`boost` (đổi sang chế độ không LLM) khi đo, hoặc chạy 3 lượt và lấy trung bình.

---

## C. Tầng đo lường

### 🟠 C1 — Mọi kết luận AVS trước 2026-08-06 phải đo lại

nDCG cũ tính sai và **vượt 1** (max 2.357). Nó không phải quan hệ thứ tự nên
không xếp hạng được cấu hình nào với cấu hình nào. Bị ảnh hưởng: AVS-GRADE-01,
việc chọn `grade_mode`, và con số 0.522 trong `docs/25`.

Đã sửa và khoá bằng test. Nhưng **những lựa chọn đã ra dựa trên nó vẫn chưa
được đo lại** — cụ thể là `grade_mode=semantic_or_lexical` và `semantic_tau`.

### 🔴 C2 — Không đủ power thống kê để phân biệt cải thiện nhỏ

36 truy vấn KIS ⇒ 1 truy vấn = 0.028. 24 truy vấn AVS/TRAKE ⇒ 1 truy vấn =
0.042. Mọi delta dưới 2 truy vấn là không phân biệt được với nhiễu.

Điều này **đã gây ra một kết luận sai**: bộ trọng số KIS "tốt nhất" (+2 truy
vấn) hoá ra là −2 trên holdout. Nếu không có kiểm tra leave-one-video-out thì
nó đã được nhận.

**Nghiên cứu tiếp:** hoặc thêm gold, hoặc bắt buộc bootstrap CI + tách theo
video cho mọi quyết định. Hiện `scripts/sweep_kis_weights.py --holdout` đã làm
vế thứ hai cho KIS; ba task còn lại chưa có.

### ⚪ C3 — Chưa có hạch toán chi phí thật

Số liệu chi phí hiện suy ra từ đếm file cache, không phải từ `usage` mà API trả
về. `FptClient` đã đính kèm `usage` trong mỗi phản hồi nhưng chưa ai gom lại.

---

## D. Vận hành

### 🟠 D0 — VLM rerank đã ngốn phần lớn hoá đơn API mà không đổi gì

**Phân tích chi phí ở bản trước của tài liệu này SAI.** Nó quy phần lớn chi phí
cho enrichment (2248 lời gọi VLM một lần). Đếm thật số lời gọi trong MỘT truy
vấn cho thấy thủ phạm khác:

| | VLM bật | VLM tắt |
|---|---|---|
| KIS | 17 lời gọi, 8.8s | **1 lời gọi, 0.6s** |
| QA | 29 lời gọi, 17.1s | 6 lời gọi, 4.8s |
| AVS | 11 lời gọi, 5.6s | **1 lời gọi, 0.5s** |

Mỗi truy vấn KIS tốn 16 lời gọi VLM kèm ảnh. Nhân với **2668 lượt truy vấn**
qua 66 lần eval ⇒ khoảng **40 000 lời gọi ảnh**, tức gấp ~18 lần toàn bộ
enrichment. Đây gần như chắc chắn là phần lớn số tiền đã tiêu, không phải
enrichment.

Và nó **không đổi một chỉ số nào** trên cả 4 task (xem §0).

Cái bẫy khiến nó sống lâu: đặt `AIC_RERANK_VLM_URL=` rỗng **không** tắt được
nhánh này. Container ưu tiên đường FPT (`enable_vlm_rerank and fpt_enabled and
fpt_vlm_model`), nên URL rỗng chỉ khiến người đọc config tưởng nó đã tắt.

### 🟠 D1 — Enrichment tốn 2.63 lần gọi cho mỗi keyframe

```
export_enrich 850 · export_caption 548 · ocr_backfill 362
fpt_enrich 307 · ocr_crop 181            = 2248 lần gọi / 855 keyframe
```

Năm vòng riêng biệt trên cùng một tập keyframe, do phải làm lại: bug kẹp toạ độ
bbox, rồi thêm cắt lớp phủ, rồi backfill. **Hệ số lãng phí 2.63×**, không phải
do thiết kế.

### 🔴 D2 — Trần 50 RPM là ràng buộc thời gian, không phải tiền

| số video | keyframe | gọi VLM | thời gian @50 RPM |
|---|---|---|---|
| 10 | 2 850 | 7 500 | 2.5 giờ |
| 30 | 8 550 | 22 500 | 7.5 giờ |
| 100 | 28 500 | 75 000 | **25 giờ** |

Nếu dataset thi phát dưới một ngày trước giờ thi, 100 video là **không kịp**,
bất kể ngân sách. Với hệ số lãng phí 2.63× thì càng không.

**Nghiên cứu tiếp:** chốt prompt và schema trên MỘT video rồi mới chạy toàn bộ;
hai harness replay đã làm vòng lặp sửa-đo rẻ đi. Và cần biết trước dataset thi
phát lúc nào.

### 🟡 D3 — Chưa từng chạy trên corpus lớn

Toàn bộ số đo trên 765 scene. Ngoại suy từ điểm đo 1→3 video (hạng gold xấu
nhất 3→8, mô hình dự đoán 7, khớp):

| số video | KIS R@20 |
|---|---|
| 3 | 1.000 |
| 10 | 0.917 |
| 30 | 0.750 |
| 60 | 0.583 |

Mốc 0.65 tổng thể giữ được tới khoảng **30 video**, vỡ quanh 50–60. Ngoại suy
từ MỘT điểm đo nên hướng đáng tin hơn con số.

---

## Thứ tự nghiên cứu — xếp lại theo mục tiêu R@20

Mục tiêu chốt 2026-08-06: **đáp án phải nằm trong 20 kết quả đầu**, người dùng
duyệt tay. Không phải R@1, không phải MRR. Thứ tự dưới đây đã xếp lại theo đó.

1. **A5 thêm video distractor** — lên số một. R@20 hiện là 1.000 nhưng ngoại
   suy cho thấy nó **bắt đầu tụt quanh 14 video**, và đó là ngoại suy từ MỘT
   điểm đo. Đây là cách duy nhất biết ngưỡng thật thay vì đoán. Không tốn API,
   `scripts/build_distractor_export.py` đã có sẵn.
2. **B1 TRAKE Stage A** — 8/40 truy vấn chỉ mang về ~1.0 điểm. Lưu ý đây là bài
   toán R@1 (chọn đúng video), không phải R@20, nên nó phục vụ tổng điểm chứ
   không phục vụ mục tiêu duyệt tay.
3. **A3 dữ liệu màu** — offline thuần CPU, không tốn API; nhánh đang rỗng hoàn
   toàn và truy vấn KIS đầy mô tả màu. Tín hiệu MỚI là thứ duy nhất kéo được
   hạng tệ nhất xuống.
4. **A1 thống nhất lọc lớp phủ** — sửa một chỗ, ba đường đọc cùng hưởng.
5. **C1 đo lại `semantic_tau`** trên metric đã sửa (`grade_mode` đã đo xong).
6. A2 action tag, C2 power thống kê, B3 negative constraint — sau.
7. **A4b dựng lại scene V002/V003** — hoãn. Đo được: KIS R@1 giống hệt nhau
   trên cả ba video, và mọi nội dung đều bắt nguồn từ keyframe nên đổi ranh
   giới scene chỉ gộp lại cùng một thông tin. Nó phục vụ TRAKE và tính công
   bằng khi so V001↔holdout, không phục vụ R@20.

**Không nên làm tiếp:** chỉnh trọng số `KisConfig` (overfit V001), thêm API
reranker (không cải thiện), distillation (chưa có model đắt nào phán đúng để
học theo), **nới `max_results_per_video`** (đo được: làm R@5 tụt 1.000 → 0.917).
