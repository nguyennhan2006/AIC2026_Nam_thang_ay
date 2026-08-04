# 20. Nhật ký thí nghiệm online search

Mỗi mục ghi đúng theo Definition of Done của
`AIC2026_Online_Search_Experiment_Validation_Guide.md` §14: hypothesis, biến
thay đổi duy nhất, số đo, và **quyết định giữ/bỏ**. Thí nghiệm bị bỏ vẫn được
ghi đầy đủ — biết một hướng không chạy được là kết quả, không phải thất bại.

Benchmark: `examples/AIC2026_L21_V001_queries_4tasks.jsonl` (40 query — 12 KIS,
12 QA, 8 AVS, 8 TRAKE) trên `storage/exports_l21/scenes.jsonl` (217 scene,
307 keyframe, 383 đoạn ASR, 1 video).

**Cảnh báo cỡ mẫu áp dụng cho toàn bộ tài liệu này:** 12 query/task nghĩa là
một query đổi hạng làm R@1 nhảy 8.3 điểm phần trăm. Không mục nào dưới đây
được coi là bằng chứng thống kê; chúng là *tín hiệu định hướng* kèm lý do cơ chế.

---

## EVAL-01 — Prefix invariance khi nới output cap

**Trạng thái: FAIL (đúng như thiết kế thí nghiệm mong đợi), đã sửa 2 bug thật.**

### Giả thuyết
Nới `fusion.max_results_per_video` không được đổi thứ hạng đã có.

### Kết quả
24/40 query vi phạm, KIS 12/12, có query lệch ngay vị trí thứ 2.

### Hai nguyên nhân, đã tách bạch

**1. Mẫu số chuẩn hoá lấy từ lát cắt, không phải từ truy vấn.**
`kis.rank` dùng `max(hit.score)` + `max(len(hit.matched_branches))`,
`qa.answer` dùng `max(frame_scores)`, `avs.rank` dùng `max(scores)` — tất cả
tính SAU `deduplicate_for_task`. Đổi cap ⇒ đổi mẫu số ⇒ điểm của candidate
không liên quan gì tới phần mới thêm cũng đổi theo.

Sửa: `online/services/normalizers.py::ScoreNormalizers.from_pool()` chốt một
lần trên pool đã fuse **trước** dedup.

Đo được (cùng cap=100, một biến duy nhất):

| KIS | trước | sau |
|---|---|---|
| R@1 | 0.333 | 0.417 |
| MRR | 0.513 | 0.566 |

= đúng 1 query đổi hạng. Giữ vì sửa phụ thuộc sai về nguyên tắc, **không**
vì con số này đủ ý nghĩa.

**2. `max_results_per_video=None` KHÔNG có nghĩa "không giới hạn"** — nó rơi
về mặc định của task (KIS = 5). Không có cách nào diễn đạt "bỏ cap", nên mọi
"baseline không cap" trước đây vẫn đang bị cap. Thêm `--max-per-video 0`.

Ảnh hưởng tới kết luận cũ:

| | cap mặc định (5) | bỏ cap |
|---|---|---|
| KIS R@20 | 0.750 | 0.917 |
| AVS nDCG | 0.000 | 0.200 |
| AVS P@100 | 0.000 | 0.292 |

⚠️ **"AVS không dùng được, phần lớn query trả 0" trong `docs/17` là chẩn đoán
sai** — AVS không hỏng, nó bị trần dedup bóp. Mọi kết luận cũ dựa trên
R@20/50/100 phải coi là `SUPERSEDED_BY_EVALUATION_FIX`.

### Chỗ không đồng ý với tài liệu
Yêu cầu "100% prefix invariance" **không thể đạt** với kiến trúc hiện tại, và
đuổi theo nó là sai hướng: cap nằm *trước* `KisProcessor.rank`, mà processor
đó là một reranker. Cho reranker ít/nhiều đầu vào ra kết quả khác nhau là
đúng bản chất. Kết luận đúng chính là §3.1 của tài liệu — cap phải là mối
quan tâm của **tầng trình bày**, tách khỏi `evaluation.max_per_video`.

Commit: `5aa6fdb`. Test: `tests/test_score_normalizers.py`.

---

## ROUTE-01 — Cho OCR/ASR về đúng 0 khi query không có cue

**Trạng thái: DROP làm mặc định. Giữ cơ chế sau cờ `allow_zero_modality`.**

### Giả thuyết
Query không có dấu hiệu chữ/lời nói thì OCR/ASR chỉ tạo false positive; cho
chúng về 0 sẽ giảm nhiễu mà không giảm recall.

### Biến thay đổi duy nhất
Sàn `OCR=0.35 / ASR=0.25` ⇢ cho phép đúng `0.0`. Mọi thứ khác cố định: index,
candidate limit, `rrf_k`, trọng số visual/caption, reranker, dedup, seed, và
bản vá evaluation của EVAL-01. Cap bỏ hẳn (`--max-per-video 0`).

### Số đo — ablation 3 nhánh

| KIS | R@1 | R@5 | MRR |
|---|---|---|---|
| A — sàn cũ | **0.417** | **0.833** | **0.585** |
| B — zero cả hai | 0.333 | 0.750 | 0.520 |
| C — giữ OCR, zero ASR | 0.333 | 0.833 | 0.515 |

QA: A MRR 0.179 > B 0.169 > C 0.164. AVS: cả ba bằng nhau (0.375).

**A thắng ở mọi biến thể.**

### Error analysis — vì sao giả thuyết sai với corpus này

OCR trong bộ dữ liệu này là **lower-third bản tin mô tả chính cảnh đó**, không
phải chữ ngẫu nhiên trong khung hình:

```
Gold scene có OCR trùng từ khoá query:     11/12 KIS
Gold scene có caption trùng từ khoá query: 12/12 KIS

KIS_E02 "cột nước phun lên từ lòng đất"
  OCR của gold scene chứa: nuoc, phun, cao
```

Tỉ lệ tín hiệu/nhiễu so với scene ngẫu nhiên:

```
OCR: gold 3.58 token trùng  vs  ngẫu nhiên 2.25  =  1.6x
ASR: gold 8.50 token trùng  vs  ngẫu nhiên 6.75  =  1.3x
```

Cả hai đều **trên nền ngẫu nhiên** — không phải nhiễu thuần. Tắt chúng đi mất
nhiều hơn được.

Đồng thời con số đó chỉ ra vấn đề thật: scene ngẫu nhiên đã trùng sẵn 2.25
token OCR / 6.75 token ASR. **Nền nhiễu của token-overlap mới là bệnh**, không
phải "sai modality" — đúng phạm vi của BM25-01, không phải ROUTE-01.

### Case regression "cột nước phun lên từ lòng đất"

```
A: đúng ở rank 1 và 3   (rank 2, 4 = lở đất + áo mưa, lọt qua bm25_ocr/bm25_asr)
B: đúng ở rank 1 và 2   (hai false positive OCR/ASR biến mất)
```

Zero-gating **có** làm đúng việc nó hứa trên case này. Nhưng aggregate cho
thấy cái giá lớn hơn cái lợi, nên case đơn lẻ không đủ để giữ.

Cháy rừng lên rank 3 ở B chỉ vì hai thứ trên nó rớt ra; nó còn trụ nhờ
`bm25_caption` khớp token rời — BM25-01 mới xử lý được.

### Cái được giữ lại

- Cơ chế zero-gating sau cờ `RuleBasedQueryPlanner(allow_zero_modality=True)`.
  Tiền đề "query không nhắc chữ ⇒ OCR không liên quan" **đúng với nhiều corpus
  khác** (phim, video đời thường); bật cờ là đủ, không phải viết lại.
- Orchestrator báo `state="disabled"` kèm lý do thay vì `state="empty"` — hai
  chuyện khác hẳn nhau khi đọc log.
- Cue list mở rộng (biển hiệu, nhãn, logo, phụ đề / giọng, hội thoại, phỏng vấn…).
- Ngoại lệ `ocr_fuzzy`: nhánh khớp gần-nguyên-chuỗi không thể tạo false
  positive kiểu "trùng token phổ biến", nên không bị tắt theo modality OCR.
  Phát hiện qua `test_container_flags` fail thật với query `"hen ngay gap lai"`
  — truy vấn gõ thẳng chữ trên màn hình, không hề chứa cue.

### KHÔNG đo được
Latency. Hai nhánh chạy tuần tự cùng tiến trình, CLIP nạp lạnh ở nhánh A, nên
con số +49.8% của nhánh B vô nghĩa. Cần harness riêng có warmup.

Commit: `d242201` (implement) + revert mặc định. Test:
`tests/test_modality_routing.py` (13 test, gồm 2 test khoá quyết định DROP).

---

## BM25-01 — Concept coverage cho lexical branch

**Trạng thái: DROP làm mặc định. Giữ cơ chế sau `CoverageConfig`.**

### Giả thuyết
Candidate đúng khớp nhiều phần khác nhau của query và khớp token phân biệt
cao; candidate sai chỉ khớp một mảnh hoặc khớp token phổ biến. Chấm coverage
sẽ tách được hai loại đó.

### Biến thay đổi duy nhất
`CoverageConfig` truyền vào `BM25Index`. Evaluator, RRF, modality weight,
candidate limit giữ nguyên. Cộng bonus/penalty **nhân** vào điểm BM25, KHÔNG
lọc cứng.

### Số đo — ablation A–E

| KIS | R@1 | R@5 | MRR | recall | w/t/l |
|---|---|---|---|---|---|
| A baseline | **0.417** | 0.833 | **0.585** | 0.917 | — |
| B unique | 0.333 | 0.833 | 0.578 | 0.917 | 2/9/1 |
| C idf | 0.333 | 0.833 | 0.564 | 0.917 | 1/10/1 |
| D groups | 0.333 | 0.833 | 0.577 | 0.917 | 2/8/2 |
| E idf+groups | 0.333 | 0.833 | 0.577 | 0.917 | 2/8/2 |

| QA | R@1 | R@5 | MRR | recall | w/t/l |
|---|---|---|---|---|---|
| A baseline | 0.083 | 0.250 | 0.179 | **0.917** | — |
| B–E (giống hệt nhau) | **0.167** | **0.417** | **0.278** | 0.833 | **7/3/2** |

AVS: năm nhánh bằng nhau.

### Vì sao DROP — cơ chế hỏng đúng trên case mục tiêu

Case regression "cột nước phun lên từ lòng đất" cho ranking **không đổi một
chữ** giữa A và E. Chẩn đoán:

```
nhóm khái niệm sinh ra: [['cột','nước','phun','lên'], ['lòng','đất']]
                        -> 2 nhóm, không phải 3

gold-like  group = 0.5
"lở đất"   group = 0.5      <- Y HỆT NHAU
```

Tách nhóm bằng ranh giới hư từ **không hoạt động với tiếng Việt viết liền**:
"cột nước phun lên" không có hư từ ở giữa nên gộp làm một nhóm, thay vì tách
chất (nước) khỏi chuyển động (phun lên). Gold và false positive nhận cùng
điểm nhóm ⇒ concept-group coverage vô hiệu.

Chỉ `unique`/`idf` coverage phân biệt được (0.67 vs 0.17) — và đó mới là thứ
tạo ra thay đổi số liệu, không phải nhóm khái niệm.

### Đối chiếu tiêu chí giữ

| Tiêu chí | Kết quả |
|---|---|
| hard-negative pair accuracy tăng rõ | KHÔNG (đi ngang/giảm nhẹ) |
| case "lở đất" bị đẩy xuống | KHÔNG — ranking không đổi |
| KIS R@5 không giảm | Đạt (0.833) |
| candidate recall không giảm đáng kể | KHÔNG — QA 0.917 -> 0.833, mất 1 query |
| không chỉ cải thiện một query | Đạt cho QA (7 query) |

Ba tiêu chí trượt ⇒ DROP.

### Lead đáng theo (một vòng có giới hạn, KHÔNG làm ngay)

QA MRR +55% (0.179 -> 0.278), R@5 +67%, 7 thắng / 2 thua — nhất quán qua cả
bốn biến thể, tức nó đến từ `idf coverage + partial_penalty` chứ không phải
từ nhóm khái niệm. Đây là tín hiệu lớn nhất đo được trong cả đợt. Cái giá là
1 QA query rớt khỏi top-100. Một vòng chỉnh riêng `partial_penalty` (thấp
hơn 0.3) rất có thể giữ được phần lợi mà không mất recall — nhưng phải là
một thí nghiệm riêng, có điểm dừng riêng.

Muốn nhóm khái niệm chạy thật thì cần từ điển cụm hoặc LLM decomposition
(§5.2 của tài liệu giả định `concept_groups` do LLM sinh), không phải chỉnh
thêm danh sách hư từ.

Test: `tests/test_lexical_coverage.py` (8 test, gồm 2 test khoá lại GIỚI HẠN
của segmenter để không ai giả định nhầm là nó đang chạy).

---

## Việc tiếp theo

**BM25-01 — concept coverage.** Đây là kết luận chung của cả hai thí nghiệm
trên: bệnh nằm ở token-overlap có nền nhiễu cao, không ở chỗ chọn modality.
Query `"cột nước phun lên từ lòng đất"` phải bị tách thành 3 nhóm khái niệm
bắt buộc (nước / phun lên / từ mặt đất); candidate chỉ khớp `đất` phải bị phạt
coverage.

**TRAKE — chưa đủ điều kiện tune.** `mean_r_score = 0.000` và
`complete_chain_rate = 0.000` ở **mọi** cấu hình đã thử. Phải đo oracle theo
từng tầng (Stage A có gold video? mỗi event có candidate trong cửa sổ? sequence
search nhận đủ event? frame output đúng hệ toạ độ?) trước khi tune ranking.
