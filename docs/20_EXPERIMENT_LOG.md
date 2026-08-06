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

## TRAKE T1–T4 — Chẩn đoán oracle theo tầng

**Kết luận: T1 đạt, T2 đạt, chết ở T3 — chọn sai vùng trong video, lệch trung
bình 254 giây. KHÔNG phải do lấy mẫu thưa.**

> ⚠️ **Bản chẩn đoán đầu tiên (commit `894d5dc`) SAI và đã bị rút lại**, đánh
> dấu `SUPERSEDED_BY_TRAKE_TOLERANCE_CLARIFICATION`. Nó kết luận "TRAKE chết ở
> T2 vì keyframe thưa hơn cửa sổ gold 13 lần" dựa trên cửa sổ ±4 frame trong
> file gold. Luật chấm thật rộng **3–6 giây tuỳ độ dài scene** (maintainer xác
> nhận), tức ±45 đến ±90 frame ở 30 fps. Phương pháp oracle T1–T4 vẫn dùng
> được; chỉ kết luận nguyên nhân phải rút lại.

### Bài học quy trình

Cửa sổ trong file gold là **mốc ngữ nghĩa**, không phải **dung sai chấm**. Lấy
nhầm cái này làm cái kia sinh ra một chẩn đoán sai hoàn chỉnh, tự nhất quán, và
suýt dẫn tới việc trích lại toàn bộ video dày gấp 14 lần — một khoản đầu tư
lớn cho vấn đề không tồn tại. Evaluator giờ nhận `--trake-tolerance-sec` và
quy đổi bằng FPS thật đọc từ `videos.jsonl` (không giả định 30 fps).

### T1 — Video oracle: ĐẠT
`correct_video_rate = 1.000` ở mọi cấu hình.

### T2 — Event candidate oracle: ĐẠT ở dung sai thật

| dung sai | event có keyframe trong cửa sổ | query phủ đủ chuỗi |
|---|---|---|
| ±1.0s | 18/35 | 0/8 |
| ±1.5s | 25/35 | 3/8 |
| ±2.0s | 28/35 | 4/8 |
| ±2.5s | 33/35 | 6/8 |
| **±3.0s** | **34/35** | **7/8** |

Keyframe cách nhau trung bình 123 frame = **4.11s**, đủ cho cửa sổ 6s. Lưới
lấy mẫu hiện tại về cơ bản là đủ.

### T3 — Sequence oracle: TRƯỢT, và đây mới là chỗ hỏng

`mean_r_score` theo dung sai:

```
±0.13s (≈ cửa sổ gold cũ)  ->  0.000
±1.5s                      ->  0.075
±2.0s                      ->  0.075
±3.0s                      ->  0.075     <- CHỮNG LẠI
```

Nới gấp đôi cửa sổ không cải thiện gì ⇒ frame dự đoán **lệch xa**, không phải
suýt trúng. Đo trực tiếp khoảng lệch:

```
lệch trung bình của dự đoán       :  7620 frame = 254.0 s
lệch trung bình TỐT NHẤT có thể   :    34 frame =   1.1 s
bước dự đoán nằm trong ±3s        :   3/35
bước TỐT NHẤT có thể trong ±3s    :  34/35
```

Ứng viên đúng nằm cách gold trung bình **1.1 giây** và có sẵn cho 34/35 bước.
TRAKE vẫn chọn frame cách **254 giây**. Ví dụ `TRAKE_E01`: dự đoán bám quanh
frame ~20400 trong khi gold nằm ở ~29200–30500.

Đáng chú ý: chuỗi dự đoán **tự nhất quán** (các bước tăng dần, khoảng cách hợp
lý) nhưng neo vào **vùng sai hoàn toàn**. Đó là dấu hiệu ràng buộc thứ tự/khoảng
cách đang lấn át độ liên quan của từng bước — beam tìm được một chuỗi "đẹp về
hình thức" ở sai chỗ.

### T3-ablation — ràng buộc thời gian KHÔNG phải thủ phạm

Giả thuyết "ordering_weight/gap_penalty lấn át độ liên quan" đã đo và **sai**:

| | mean_r_score | lệch TB | bước trong cửa sổ |
|---|---|---|---|
| A hiện tại | **0.138** | **249.8s** | 6/35 |
| B order_weight=0 | 0.075 | 351.3s | 3/35 |
| C gap_penalty=0 | 0.075 | 316.2s | 3/35 |
| D cả hai = 0 | 0.075 | 316.2s | 3/35 |

Tắt ràng buộc làm **tệ đi**. Chúng đang cứu vãn chứ không phá — nghĩa là đầu
vào của beam đã sai sẵn.

### Nút thắt thật: retrieval của TỪNG BƯỚC

```
query tách đúng số bước                    :  8/8   ĐẠT
frame ứng viên tồn tại quanh gold (±2–7s)  : 34/35  ĐẠT
retrieval từng bước tìm ra vùng gold /top-100: 13/35  TRƯỢT
   trong số tìm được: top-1 = 1, top-5 = 6, rank trung vị = 10
```

**22/35 bước không có vùng gold trong top-100 của chính bước đó.** Beam không
thể ghép đúng từ một pool không chứa đáp án. Chuỗi nó trả về tự nhất quán vì
ràng buộc thứ tự/khoảng cách vẫn hoạt động — chỉ là trên tập ứng viên sai.

Lưu ý cách đọc: text mỗi bước đến từ việc tách `(1)...(2)...` trong query,
KHÔNG phải từ gold — gold event chỉ có cửa sổ frame, không có trường mô tả
(`event_description_*` rỗng ở cả 35 event).

### Baseline chính thức — đo TRONG Stage B thật

Con số `13/35` ở trên là **proxy sai**, đo bằng cách gọi
`search(task=TEXTUAL_KIS)` cho từng event. Hai đường khác nhau về vật chất:

```
Stage B thật   : _retrieve(event_plan, candidate_limit) -> _hydrate
                 KHÔNG dedup, KHÔNG KisProcessor.rank, KHÔNG cắt top_k
proxy           : _retrieve -> dedup -> rerank -> _hydrate
                          -> _format_results(top_k) -> KisProcessor.rank
```

Dedup và KisProcessor đẩy vùng gold xuống nên proxy bi quan hơn thật. Baseline
đúng, đo bằng `scripts/diagnose_trake_stage_b.py` (deterministic — hai lần
chạy ra số y hệt):

```
pool mỗi bước               : 100 candidate
gold_region_recall@20       : 15/35
gold_region_recall@50       : 18/35
gold_region_recall@100      : 21/35
khi tìm được                : rank trung vị 12, top-1 = 1, top-5 = 8, top-20 = 15

query có ĐỦ mọi event retrieve được : 1/8      <- RÀNG BUỘC QUYẾT ĐỊNH
```

**`1/8` mới là con số chặn `complete_chain_rate`.** 7/8 query có ít nhất một
bước mà vùng gold không bao giờ vào pool — chuỗi đầy đủ là bất khả thi bất kể
beam tốt tới đâu. Đây là điều kiện CẦN, phải sửa trước khi nói tới alignment.

### Việc phải làm

Vấn đề là **chất lượng khớp ngữ nghĩa của một câu mô tả ngắn** ("cột nước
được quay từ xa") với caption của scene. Đây đúng là khoảng trống đã xác định
từ đầu đợt: hệ thống có 4 nhánh BM25 lexical + 1 nhánh CLIP ảnh, **không có
nhánh dense text nào trên caption**. Câu ngắn + BM25 token = giòn.

Dấu hiệu ủng hộ: **khi** retrieval tìm được vùng gold thì nó xếp khá tốt
(top-5 cho 8/21). Vấn đề thuần tuý là **độ phủ**, không phải thứ tự — đúng
chữ ký của lệch từ vựng mà dense text xử lý.

⇒ Thí nghiệm tiếp theo cho TRAKE là **DENSE-TEXT-01**, không phải tinh chỉnh
beam. Hai metric để chấm:

```
gold_region_recall@100          21/35  ->  mục tiêu > 25/35
query có đủ mọi event retrieve   1/8   ->  mục tiêu >= 4/8
```

⚠️ Ngưỡng "13/35 -> 20/35" đặt trước đó tính trên baseline sai; baseline thật
đã là 21/35. Ngưỡng phải đặt lại như trên.

### T4 — chưa xét
Chỉ có nghĩa sau khi tỉ lệ 13/35 ở trên được cải thiện.

---

## DENSE-TEXT-01 — Caption dense branch (E5)

**Trạng thái: KHÔNG đạt ngưỡng giữ. Nhưng đo được tín hiệu thật và định vị
được nút thắt kế tiếp — nằm ở OFFLINE, không phải retrieval.**

### Chuẩn bị
- Model tải bằng `scripts/download_hf_model.py` (curl, né lỗi SSL của Python).
- Index: `scripts/build_caption_dense_index.py`, schema `caption_dense_v1`
  (caption + object + action + keyword; **chưa** có OCR/ASR để không lẫn gain).
  216/217 scene có text dùng được, dim 1024, fingerprint `e6cfd51b353220d7`.
- Nhánh online: `online/adapters/dense_text.py::CaptionDenseRetriever`, prefix
  `query:`/`passage:` đọc từ manifest để online-offline không lệch.
- Sanity check trước full run: dense cứu `"cán bộ phát biểu trong cuộc họp"`
  từ rank 59 → 1, nhưng mất `"bác sĩ tiến hành phẫu thuật"` (4 → không thấy).

### Số đo — trên Stage B THẬT

| | R@20 | R@50 | R@100 | median rank | đủ mọi event |
|---|---|---|---|---|---|
| A baseline | 15/35 | 18/35 | 21/35 | 13 | 1/8 |
| B dense only | 17/35 | 18/35 | 19/35 | **3** | 0/8 |
| C baseline+dense | **17/35** | **20/35** | **22/35** | 6.0 | 1/8 |

Đối chiếu ngưỡng đã chốt:

| Tiêu chí | Kết quả |
|---|---|
| R@100 > 25/35 | **TRƯỢT** — 22/35 |
| đủ mọi event ≥ 4/8 | **TRƯỢT** — 1/8, không đổi |
| R@20 không giảm | Đạt — 15 → 17 |

### Điều quan trọng hơn con số

Dense cải thiện mạnh **thứ hạng** (median 13 → 3) nhưng gần như không cải
thiện **độ phủ** (21 → 22). Nghĩa là: khi caption có mô tả đúng nội dung,
dense tìm ra nó tốt hơn hẳn BM25. Nhưng 14 bước không tìm được thì không phải
vì retriever kém.

### Error analysis — 14 bước không bao giờ tìm thấy

```
tìm thấy                             : 21/35
KHÔNG có scene nào chứa frame gold   :  5/35   <- lỗ hổng phân đoạn scene
scene tồn tại nhưng document rỗng    :  0/35
scene + caption có, nhưng không khớp :  9/35   <- chất lượng caption
```

Ví dụ chế độ hỏng thứ hai:

```
event  "xe cứu hỏa bật đèn xanh"
caption "một đám cháy rừng dữ dội với ánh sáng đỏ rực và khói bốc lên cao"
        -> đúng hiện trường, nhưng caption không hề nhắc tới xe cứu hỏa

event  "rùa được thả từ thuyền xuống biển"
caption "những người đang giúp đỡ một người khác lên tàu trên biển"
        -> bỏ sót hoàn toàn con rùa, tức chủ thể của sự kiện
```

Còn gặp một caption lẫn tiếng Trung (`色彩鲜艳的建筑在背景中`) — lỗi của
VLM lúc enrichment.

**Không encoder text nào cứu được hai chế độ này.** Caption không chứa thông
tin thì embedding nó cũng không có. 5 bước còn lại thậm chí không có scene nào
chứa frame gold — scene không lát kín video.

### Quyết định

- **Không promote** caption dense làm mặc định: trượt 2/3 tiêu chí.
- **Giữ code** (`dense_text.py`, builder, runner) vì nó đã chứng minh cải
  thiện thứ hạng rõ rệt và sẽ có giá trị NGAY khi caption tốt lên.
- **KHÔNG tải BGE-M3 lúc này.** So hai encoder khi 14/35 thất bại nằm ở dữ
  liệu chứ không ở encoder là so nhầm chỗ. Điều kiện để tải: sau khi sửa
  caption/scene, `R@100` vượt ~25/35 mà vẫn còn khoảng cách đáng kể.

### Nút thắt kế tiếp (offline)

```
1. 9/35 — caption bỏ sót chủ thể của sự kiện
   -> prompt enrichment phải liệt kê phương tiện/động vật/vai trò người,
      không chỉ tả không khí chung của cảnh
   -> caption lẫn ngôn ngữ khác cần bị bắt và sinh lại

2. 5/35 — không scene nào chứa frame gold
   -> kiểm tra scene có lát kín [0, frame_count) không; nếu có khoảng trống
      thì gold rơi vào đó và không candidate nào tồn tại
```

Đây là lần thứ tư trong đợt mà nút thắt lùi thêm một tầng: evaluator →
routing → lexical → retrieval → **chất lượng dữ liệu offline**.

---

## SCENE-COVERAGE-01 — Scene có lát kín video không?

**Trạng thái: TRƯỢT. Tìm ra nguyên nhân gốc chính xác, chưa sửa.**

### Số đo

```
L21_V001: 217 scene / 37849 frame
coverage = 0.786441          <- chỉ phủ 78.6% video
gap      = 84 (mất 8083 frame)
overlap  = 0    zero_length = 0    out_of_range = 0
```

Scene ID **không liên tục**: `S0000 → S0002`, `S0004 → S0006`, `S0006 → S0009`.
Đây là dấu hiệu scene bị mất khi export, không phải detector bỏ sót — chế độ
hỏng C, không phải A (gap thật) hay B (nhầm quy ước interval).

### Nguyên nhân gốc

`storage/exports_l21/quarantine.jsonl` có đúng **119 dòng**, tất cả cùng một
lý do, ở stage `keyframe`:

```
scene L21_V001_S0001 [9, 54) không có keyframe nào
  — canonical Scene bắt buộc >= 1 keyframe
```

Đối chiếu:

```
input/scene_manifest.jsonl (TransNetV2) : 336 scene
keyframe đã trích                        : 307
scene sống sót vào export                : 217        (336 - 119)

scene BỊ LOẠI  : dài trung vị 61 frame, min 9,  max 204
scene GIỮ LẠI  : dài trung vị 98 frame
stride trích keyframe ~123 frame
```

**Keyframe được trích theo stride cố định ~123 frame, không phải theo scene.**
Scene ngắn hơn stride rơi trọn qua lưới, không nhận được keyframe nào, rồi bị
schema canonical (`>= 1 keyframe`) loại bỏ. 119 scene biến mất, để lại 84 gap.

Schema không sai — scene không có keyframe thì thật sự không dùng được. Sai ở
**bước trích keyframe**: nó phải bảo đảm mỗi scene ít nhất một frame.

### Vì sao điều này chặn mọi thứ phía trên

5/35 bước TRAKE có frame gold rơi vào các gap này. Candidate tương ứng **không
tồn tại trong corpus**, nên caption tốt hơn, dense retrieval, BM25 thông minh
hơn hay reranker đều không thể tìm ra. Đây là trần cứng.

### Cách sửa (offline)

```
1. Trích lại keyframe THEO SCENE: mỗi scene ít nhất 1 frame (vd giữa scene),
   thay vì stride toàn cục.
   -> 119 scene × ≥1 frame; nguồn có sẵn: input/scene_manifest.jsonl +
      storage/raw/videos/L21_V001.mp4
2. Caption 119 scene mới (FPT VLM) — không có caption thì scene tồn tại nhưng
   vẫn không khớp được truy vấn nào.
3. offline assemble lại, kiểm tra coverage = 1.0.
4. Chạy lại Stage B baseline + E5.
```

Bước 2 tốn tiền API nên cần quyết định trước khi chạy.

### ĐÃ SỬA — gộp gap vào láng giềng có keyframe

Không trích lại keyframe, không caption lại: **keyframe BTC đã chứa câu trả
lời**, vấn đề chỉ là scene chứa mốc gold bị xoá. `scripts/repair_scene_coverage.py`
gộp mỗi gap vào láng giềng có keyframe GẦN TÂM GAP NHẤT (hoà thì ưu tiên
scene trước — tất định, không phụ thuộc thứ tự duyệt).

```
84 gap, 8083 frame  ->  coverage 0.786 -> 1.0, gap = 0
```

Tác dụng phụ có lợi và đúng luật: cửa sổ chấm suy từ độ dài scene
(`clamp(duration*0.5, 2, 7)`). Frame trong gap trước đây không có scene nên
rơi về fallback tối thiểu **±1.0s** — chính những mốc cần cửa sổ rộng nhất lại
bị chấm ngặt nhất. Sau khi gộp chúng thuộc scene dài hơn nên nhận cửa sổ đúng.

### Kết quả trên Stage B

| | R@20 | R@50 | R@100 | median | đủ mọi event |
|---|---|---|---|---|---|
| baseline, export gốc | 15/35 | 18/35 | 21/35 | 12 | 1/8 |
| baseline, export đã sửa | 16/35 | 19/35 | 22/35 | 13.5 | 1/8 |
| dense only, export gốc | 17/35 | 18/35 | 19/35 | 3 | 0/8 |
| **dense only, export đã sửa** | **21/35** | **23/35** | **24/35** | **2.5** | 0/8 |
| fused, export đã sửa | 18/35 | 21/35 | 24/35 | 6.0 | 1/8 |

**Repair giúp dense nhiều hơn hẳn baseline** (R@20 17→21 so với 15→16). Lý do
cơ chế: scene gộp dài hơn ⇒ cửa sổ rộng hơn ⇒ keyframe mà dense tìm ra rơi vào
dung sai thường xuyên hơn. Hai sửa đổi cộng hưởng chứ không cộng tuyến tính.

Tổng tiến bộ so với điểm xuất phát: **R@100 21→24, R@20 15→21 (+40%),
rank trung vị 12→2.5**.

### Một quan sát ngược trực giác

`B dense only` **thắng** `C baseline+dense` ở R@20 (21 so với 18). Thêm bốn
nhánh BM25 vào làm TỆ ĐI ở độ sâu nông — chúng đẩy kết quả tốt của dense
xuống. Với truy vấn ngữ nghĩa ngắn của TRAKE, lexical đang là nhiễu nhiều hơn
tín hiệu. Chưa đủ dữ liệu để chốt bỏ hẳn lexical cho TRAKE, nhưng đáng thành
một thí nghiệm riêng.

### Còn thiếu gì để đạt ngưỡng

```
R@100          24/35, ngưỡng > 25/35   -> còn thiếu 1-2 bước
đủ mọi event   0-1/8, ngưỡng >= 4/8    -> CHƯA nhúc nhích
```

Phần còn lại chính là 9/35 bước có scene + caption nhưng caption bỏ sót chủ
thể sự kiện — CAPTION-ENRICH-01. Đó mới là thứ chặn `đủ mọi event`.

### Kỳ vọng sau khi sửa

```
coverage ratio                    0.786 -> 1.0
gold event có scene chứa nó       30/35 -> 35/35
gold_region_recall@100            không được giảm
queries_with_all_events           không được giảm
```

Chưa kỳ vọng retrieval tìm đúng cả 5 ngay — bước này chỉ bảo đảm chúng **tồn
tại để có thể được tìm**.

Công cụ: `scripts/check_scene_coverage.py`, artifact
`outputs/evaluation/scene_coverage_l21.json`.
Test: `tests/test_scene_coverage.py` (8 test, gồm quy ước nửa mở và trường hợp
scene ngắn hơn stride).

---

## CAPTION-ENRICH-01 — và phát hiện quan trọng nhất của cả đợt

**Trạng thái: caption tốt lên thật, nhưng metric không đổi — vì đo nhầm tầng.
Retrieval scene ĐÃ ĐẠT 35/35. Nút thắt thật là CHỌN FRAME.**

### Đã làm
11 scene mục tiêu (chỉ những scene có keyframe + có caption nhưng retrieval
vẫn trượt), 19 keyframe. Prompt `caption_event_factual_v1` bắt liệt kê riêng
phương tiện/động vật/đồng phục/vật thể nhỏ. Gate: JSON hợp lệ, không lẫn CJK,
phải có ít nhất một vật thể/hành động cụ thể. Trượt gate thì giữ caption cũ.

10/11 caption được nhận. Caption mới phủ THÊM từ khoá event ở **6/10** scene.

### Hai lỗi provider/code phát hiện khi chạy
1. **FPT VLM chỉ nhận 1 ảnh mỗi prompt** (HTTP 400 `At most 1 image(s) may be
   provided in one prompt`). Variant "multi-frame" vẫn làm được nhưng phải gọi
   từng ảnh rồi hợp nhất — và đó chính là điều cần thiết, vì chủ thể nhỏ
   (con chó, xe cứu thương) chỉ xuất hiện ở đúng một keyframe. 5/11 ca trượt
   ban đầu là do ràng buộc này, không phải chất lượng caption.
2. Model trả phần tử danh sách dạng object (`{"name": ..., "description": ...}`)
   dù prompt xin chuỗi. `str()` thẳng nhét nguyên dict repr vào caption —
   rác cho cả BM25 lẫn embedding. Phải rút lấy phần chữ, không siết prompt
   (siết prompt không bảo đảm được).

### Kết quả Stage B — KHÔNG ĐỔI

| | R@20 | R@50 | R@100 | median | đủ mọi event |
|---|---|---|---|---|---|
| dense only, trước enrich | 21/35 | 23/35 | 24/35 | 2.5 | 0/8 |
| dense only, sau enrich | 21/35 | 23/35 | 24/35 | 2.5 | 0/8 |

Caption tốt lên mà số không nhúc nhích — mâu thuẫn này buộc phải tách metric.

### Tách tầng: retrieval scene vs chọn frame

```
SCENE đúng được retrieve : 35/35     <- HOÀN HẢO
FRAME nằm trong dung sai : 24/35
```

**Retrieval không còn là nút thắt.** Nó tìm đúng scene cho cả 35/35 bước, phần
lớn ở rank 1–6. Toàn bộ khoảng cách còn lại nằm ở việc chọn frame nào trong
scene đã tìm đúng.

`gold_region_recall` từ trước tới nay **trộn hai tầng làm một** vì nó kiểm
`step.contains(hit.best_frame_idx, tol)` — tức đo frame, nhưng bị đọc như đo
retrieval. Đây là lý do CAPTION-ENRICH-01 cải thiện caption mà chỉ số không
đổi: caption chỉ ảnh hưởng tầng đã đạt 100%.

### 11 bước hỏng chia làm hai loại

**Loại 1 — keyframe trong dung sai CÓ tồn tại nhưng hệ thống nộp frame khác
(4/11), sửa được bằng code:**

```
người hướng dẫn đi cùng hai con chó   rank  1, kf lệch 0.4s, dung sai 1.0s
cận cảnh biển số của các xe            rank  6, kf lệch 0.1s, dung sai 1.8s
cá mú lớn bơi giữa đàn cá              rank 15, kf lệch 1.5s, dung sai 3.5s
một người đàn ông tiến sát cột nước    rank  4, kf lệch 2.3s, dung sai 3.5s
```

Đây là lỗi Stage C (`frame_refinement`): scene có nhiều keyframe nhưng nó
không chọn cái gần mốc sự kiện nhất.

**Loại 2 — không keyframe nào trong dung sai (7/11):** scene ngắn 2.9–5.4s chỉ
có 1 keyframe, lệch 1.6–3.1s trong khi dung sai chỉ ±1.0–1.3s. Loại này cần
trích dày hơn, không sửa bằng code được.

### Bổ sung: có dùng OCR/ASR làm ngữ cảnh cho VLM chưa?

**Chưa.** Kiểm tra lại prompt: nó chỉ gửi `caption cũ` + `event text`.
`ocr_old` được thu thập trong danh sách mục tiêu nhưng **không bao giờ dùng**
(field chết); ASR thì không thu thập.

Đây là thiếu sót đáng kể vì dữ liệu tồn tại và rất giàu:

```
7/11 scene có OCR,  11/11 scene có ASR

S0224 "xe cứu hỏa"      ASR: "Hơn 1.000 lính chữa cháy và 20 máy bay được huy động…"
S0265 "cá mú lớn"       ASR: "Cá múa nâu biểu tượng của địa trung hải…"
S0014 "bờ sông sụt lún" ASR: "phòng chống sụp lúng đất, sạt lở bờ sông"
S0168 "biển số các xe"  ASR: "lao vào lề đường, va chạm vào 3 xe máy khác…"
```

Đã nối OCR + ASR vào prompt và đo ba biến thể (cùng 10 scene, cùng gate):

| | phủ thêm từ khoá event |
|---|---|
| v1 — không OCR/ASR | **6/10** |
| v2 — có OCR/ASR + cảnh báo chống bịa NẶNG | 4/10 |
| v3 — có OCR/ASR + cảnh báo NHẸ | 5/10 |

**Thêm ngữ cảnh làm TỆ ĐI**, ngược hẳn kỳ vọng. Cơ chế đã xác nhận được một
phần: cảnh báo nặng ("TUYỆT ĐỐI KHÔNG thêm… lời dẫn thường lệch thời điểm")
khiến model *né* dùng ngữ cảnh và trả caption ngắn/mơ hồ hơn. Nới cảnh báo kéo
lại được 4→5 (S0294 từ 0 lên 2, có hẳn "một người hướng dẫn đi cùng"), nhưng
vẫn không vượt được bản không có ngữ cảnh.

Cảnh báo về cách đọc: chỉ số này là đếm trùng token giữa event text và caption
— thô. Caption có thể tốt hơn về ngữ cảnh mà không tăng trùng token. Nhưng đây
là thước đo đang có, và nó KHÔNG ủng hộ việc thêm OCR/ASR.

Giữ lại code nối OCR/ASR (`used_ocr`/`used_asr` ghi vào từng record) vì nó
đúng và có thể có giá trị với prompt khác — nhưng KHÔNG bật mặc định.

### Quyết định

- **Không promote** caption enrichment: đo được cải thiện caption (6/10) nhưng
  không cải thiện chỉ số nào ở tầng đang bị chặn. Giữ export
  `storage/exports_l21_enriched/` để dùng lại khi tầng frame được sửa.
- **Việc tiếp theo là Stage C frame selection**, không phải retrieval, không
  phải fusion, không phải encoder. 4/11 sửa được ngay bằng code.

### Bài học đắt nhất của đợt

Ba thí nghiệm liên tiếp (DENSE-TEXT-01, SCENE-COVERAGE-01, CAPTION-ENRICH-01)
đều tối ưu tầng retrieval, trong khi tầng đó đã đạt 35/35 từ sớm. Nguyên nhân:
một metric duy nhất trộn hai tầng, và không ai tách nó ra cho tới khi gặp mâu
thuẫn "đầu vào tốt lên, đầu ra đứng yên". Mâu thuẫn đó mới là thứ ép phải tách.

---

## FRAME-REFINE-01 — Chọn frame trong Stage C

**Trạng thái: DROP mọi thay đổi. Ba chiến lược chọn frame khác nhau đều cho
CÙNG một kết quả — chọn frame không phải đòn bẩy.**

### Đã khoá metric trước khi sửa

`scripts/diagnose_trake_stage_c.py` tách hẳn ba tầng, đúng bài học của
CAPTION-ENRICH-01 (một metric trộn tầng đã khiến ba thí nghiệm tối ưu nhầm chỗ):

```
scene_recall                          : 30/35
frame_oracle_coverage                 : 23/35   scene đúng VÀ có keyframe hợp lệ
frame_selection_accuracy_given_oracle : 19/23   <- chỉ số DUY NHẤT được phép cải thiện
frame_hit tổng                        : 19/35

không có keyframe hợp lệ (trần cứng)  :  7/35
```

### Chẩn đoán ban đầu — chính xác nhưng chỉ đúng một nửa

Cả 4 bước hỏng đều trả về **đúng bằng anchor**:

```
S0046  có=[5760,5889,6099]        hợp lệ=[6099]  anchor=5760  -> CHỌN 5760
S0294  có=[34032,34092]           hợp lệ=[34092] anchor=34032 -> CHỌN 34032
S0265  có=[30722,30932,31124,31290] hợp lệ=[31124] anchor=31290 -> CHỌN 31290
S0168  có=[20612,20702]           hợp lệ=[20702] anchor=20612 -> CHỌN 20612
```

Khoảng cách tới keyframe hợp lệ: 339, 60, 166, 90 frame — đều **vượt
`window_frames = 45`**. Cửa sổ ±1.5s hẹp hơn dung sai chấm (2–7s) nên nó loại
đúng cái frame lẽ ra tính là trúng, TRƯỚC khi `score_frames` kịp nhìn thấy.

### Nhưng nới cửa sổ ra thì NET ZERO

| | frame_selection_accuracy_given_oracle |
|---|---|
| A — cửa sổ ±45 hiện tại | 19/23 |
| B — bỏ lọc cửa sổ khi pool nhỏ | **19/23** |
| C — chọn bằng CLIP text↔image (embedding đã cache) | **19/23** |

Nới cửa sổ sửa được S0294 + S0168 nhưng làm hỏng S0046 + S0035 — hai ca trước
đó đúng **do may**, vì anchor tình cờ là frame hợp lệ. CLIP cũng đúng 19/23,
sai ở bốn ca khác.

Ba cách tiếp cận độc lập cùng chạm một trần ⇒ 4 lỗi còn lại không phải lỗi
thuật toán chọn. Xem kỹ: scene có 2–4 keyframe **trông rất giống nhau** (cùng
cảnh cột nước, cùng góc phố), chỉ một cái rơi vào cửa sổ gold. Phân biệt
"người đàn ông tiến sát cột nước" với "cột nước quay từ xa" đòi hỏi chi tiết
mà cả token overlap lẫn CLIP toàn ảnh đều không giữ được.

### Quyết định

- **Revert** thay đổi cửa sổ: đo được là trung tính, mà đổi hành vi không có
  lợi ích đo được thì chỉ thêm rủi ro. Cùng nguyên tắc đã áp cho ROUTE-01 và
  BM25-01.
- **Giữ** `scripts/diagnose_trake_stage_c.py` — đây mới là sản phẩm thật của
  vòng này: từ nay không ai còn nhầm ba tầng với nhau nữa.

### Đòn bẩy thật nằm ở đâu

```
selection accuracy   19/23 = 83%   <- gần trần, ba cách đều thế
oracle coverage      23/35 = 66%   <- 12 bước không có gì để chọn
   trong đó  7/35 scene đúng nhưng KHÔNG keyframe nào hợp lệ
             5/35 scene không được retrieve (cấu hình baseline)
```

Cải thiện selection tối đa còn 4 bước. Cải thiện oracle coverage còn 12.
**DENSE-FRAME-01 có headroom gấp ba lần** — và nó là thứ duy nhất chạm được
vào 7 bước đang bị trần cứng.

Phạm vi hẹp cho DENSE-FRAME-01: chỉ trích thêm frame ở scene có
`oracle_keyframe_exists = false`, stride 0.5–1.0s (hoặc 5 mốc
start/25%/center/75%/end). Scene ngắn 2.9–5.4s nên chi phí rất nhỏ; KHÔNG
dense toàn video.

---

## FIX-DETERMINISM-01 — Nguồn nhiễu KHÔNG phải hash order

**Trạng thái: ĐÃ SỬA. Chẩn đoán ban đầu ("PYTHONHASHSEED rò vào ranking") SAI.**

### Cách truy

`scripts/dump_search_trace.py` đổ dấu vết theo TỪNG TẦNG (branch → fused →
dedup → hits → per-event → task output) rồi so xuyên seed, thay vì chỉ so
metric cuối. Kết quả: mọi tầng dùng chung GIỐNG HỆT nhau; chỉ `trake_events`
(đường retrieval per-event) khác.

Thu nhỏ tiếp: gọi `_retrieve` ba lần với CÙNG plan, trong CÙNG tiến trình:

```
run0: dense_visual timeout 3004ms > deadline 3000ms   -> 0 candidate
run1: dense_visual failed "Cannot copy out of meta tensor"
run2: dense_visual success 142ms, 100 candidate       -> top khác HẲN
```

**Không liên quan gì tới hash seed.** Cùng một tiến trình, cùng một plan, ba
kết quả khác nhau.

### Nguyên nhân gốc — hai lỗi chồng nhau

1. **Nạp model lười trong request path.** `LocalClipTextEncoder` nạp CLIP ở
   lần `encode()` đầu tiên (~3s), vượt deadline 3000ms của nhánh, nên
   `dense_visual` bị `asyncio.wait_for` huỷ và bỏ qua trong im lặng.
2. **`_load()` không nguyên tử.** Bị huỷ giữa `from_pretrained` để lại model
   kẹt trên `meta` device, nên lần gọi TIẾP THEO hỏng hẳn.

Hệ quả: **1–2 truy vấn đầu tiên của MỌI tiến trình chạy không có nhánh dense**,
cho ranking khác hẳn các truy vấn sau. Trong `eval_tasks` 40 query, thứ tự
query quyết định ai chịu phạt — nên chạy `--tasks TRAKE` (8 query) và chạy đầy
đủ (40 query) cho số khác nhau, và hai lần chạy liên tiếp cũng khác nhau.

Cố định `PYTHONHASHSEED` làm số ổn định chỉ vì nó đổi thời điểm/thứ tự, KHÔNG
phải vì hash order — một tương quan giả mà tôi đã nhận nhầm thành nhân quả.

### Sửa

- `_load()` dựng vào biến cục bộ rồi mới gán vào `self`, có `threading.Lock`.
  Huỷ giữa chừng thì `self._model` vẫn None ⇒ lần sau nạp lại sạch.
- Thêm `warmup()`, gọi ở `build_container()` và `build_service()` — nạp NGOÀI
  mọi deadline. Lỗi warmup chỉ in cảnh báo, không chặn khởi động.

### Xác minh

```
Ba lần _retrieve liên tiếp: dense success 244/250/245ms, top GIỐNG HỆT
Toàn benchmark, PYTHONHASHSEED = 0 / 1 / 42:
  KIS R@1 0.250  MRR 0.442
  QA  R@1 0.083  MRR 0.175  answer_accuracy 0.333
  TRAKE mean_r_score 0.075
  AVS nDCG@100 0.238
                                        <- ba seed ra số Y HỆT
```

### Ảnh hưởng ngược lại các kết luận cũ

Mọi so sánh chênh 1–2 query trong tài liệu này đều có thể đã bị nhiễu bởi lỗi
cold-start, không phải bởi biến đang thí nghiệm. Cụ thể đáng ngờ nhất:

- EVAL-01 "KIS R@1 0.333 → 0.417" (đúng 1 query)
- ROUTE-01 A/B/C (chênh 1 query giữa các nhánh)
- BM25-01 A–E (KIS chênh 1 query)

Các phát hiện CẤU TRÚC không bị ảnh hưởng vì chúng là phép đếm, không phụ
thuộc thứ hạng: coverage 78.6%→100%, scene retrieval 35/35, oracle coverage
23/35, `frame_selection_accuracy_given_oracle` 19/23.

**Cần chạy lại toàn bộ benchmark với bản vá này trước khi dùng bất kỳ so sánh
chi tiết nào để ra quyết định.**

Test: `tests/test_encoder_warmup.py` (nạp nguyên tử, khoá đồng thời, và khoá
sự tồn tại của `warmup` vì caller dò bằng `hasattr`).

---

## FPT-WIRE-01 — Cắm 4 model FPT vào các tầng còn rỗng

### Phát hiện làm đổ mọi con số cũ

`scripts/eval_kis.py::build_service` là **định nghĩa pipeline THỨ HAI**, không
đi qua `online/api/container.py`. Nó thiếu nhánh object/action/color/event,
thiếu VLM rerank, và không bọc encoder bằng `TranslatingTextEncoder`. Mọi số
trong tài liệu này trước FPT-WIRE-01 đều là số của bản dựng đó, **không phải
của hệ thống mà server chạy** — dù chúng trông hoàn toàn hợp lệ.

Đã thêm `--pipeline container` cho cả `eval_kis.py` và `eval_tasks.py`. Bản
`legacy` giữ lại vì các mode ablation (`metadata_only`/`vector_only`/
`ocr_only`) cần nó, nhưng **không được dùng để báo cáo điểm nữa**.

Riêng việc đo đúng pipeline đã đổi kết quả nhiều hơn bất kỳ thí nghiệm ranking
nào từng chạy:

| | legacy (số cũ) | container (A) |
|---|---|---|
| KIS MRR | 0.442 | **0.547** |
| QA answer_accuracy | 0.333 | **0.500** |
| TRAKE mean_r_score | 0.075 | **0.225** |

QA tăng vì QA LLM trước đó **hỏng ở mọi lệnh gọi** mà không ai biết (xem dưới).

### QA LLM chưa từng chạy được lần nào

`AIC_FPT_LLM_MODEL=Qwen3.6-27B` là model reasoning. Với
`response_format={"type":"json_object"}`, bản triển khai của FPT đặt TOÀN BỘ
câu trả lời vào `reasoning_content` và để `content=None`, dù `finish_reason`
là `"stop"` và chỉ tốn 22 token. `FptQaAnswerer` đọc `content` -> ném
`SchemaInvalidError` -> `QaProcessor` lặng lẽ rơi về rule-based.

Nên trước đợt này chỉ **2/9** model FPT thật sự sống (text rerank + VLM cho
enrichment offline), không phải 3 như đã ghi.

Bỏ `response_format` thì `content` đúng nhưng tốn **1091 token** thay vì 22 —
nên đường json_object rẻ hơn 50 lần và đáng giữ, chỉ cần đọc đúng field.

### Model nào trả thẳng, model nào reasoning

Đo trên FPT, cùng một câu dịch VI→EN:

| model | token | `content` |
|---|---|---|
| gemma-4-31B-it | 9 | OK |
| gemma-3-27b-it | 11 | OK |
| gpt-oss-20b | 165 | OK |
| Llama-3.3-70B-Instruct | 4 | OK (nhưng dịch cụt) |
| DeepSeek-V4-Flash | >200 | None (reasoning) |
| GLM-5.2 | >200 | None (reasoning) |
| Qwen3.6-27B | 1652 | OK (reasoning) |

`DeepSeek-V4-Flash` được ghi sẵn ở vai "LLM nhanh" nhưng chính nó là model
reasoning — sai vai. Đổi sang `gemma-4-31B-it`.

Catalog thật có 17 model; `SaoLa3.1-medium` và `GLM-5.1` trong các dòng
ablation đã comment **không tồn tại**.

### Ablation — một biến mỗi lần, PYTHONHASHSEED=0, `--max-per-video 0`

| metric | A gốc | B dịch | C expand | D dịch+expand |
|---|---|---|---|---|
| KIS R@1 | 0.333 | **0.500** | 0.333 | **0.500** |
| KIS R@5 | 0.750 | **0.917** | 0.750 | **0.917** |
| KIS R@20 | 0.917 | **1.000** | 0.917 | **1.000** |
| KIS MRR | 0.547 | **0.720** | 0.512 | **0.720** |
| QA R@1 | 0.083 | 0.083 | 0.167 | 0.167 |
| QA MRR | 0.216 | 0.261 | 0.266 | **0.291** |
| QA answer_acc | 0.500 | **0.583** | 0.500 | **0.583** |
| TRAKE mean_r | 0.225 | **0.263** | 0.225 | 0.231 |
| AVS nDCG@100 | 0.238 | **0.299** | 0.201 | **0.299** |
| AVS P@100 | 0.333 | 0.312 | 0.271 | 0.312 |

**B (dịch VI→EN trước CLIP text tower) — GIỮ.** Thắng ở 8/10 chỉ số, và là
thay đổi lớn nhất từ trước tới nay: KIS R@20 đạt trần 1.000. Nguyên nhân rõ
ràng chứ không phải may: vector ảnh sinh bằng `openai/clip-vit-large-patch14`,
mà text tower của CLIP chỉ được huấn luyện trên tiếng Anh — đưa thẳng truy vấn
tiếng Việt vào là so một câu model chưa từng học với vector ảnh.

**C (mở rộng đồng nghĩa tiếng Việt cho BM25) — KHÔNG giữ cho KIS/AVS.** Nó
làm GIẢM KIS MRR (0.547 -> 0.512) và AVS nDCG (0.238 -> 0.201). Đúng kiểu
query drift: thêm term đồng nghĩa kéo theo candidate chỉ khớp term phụ.

**E (dịch + VLM rerank) = B ĐÚNG TỪNG CHỮ SỐ trên cả 10 chỉ số.** Tốn
~1400 lệnh gọi FPT và nhiều phút, dịch chuyển đúng 0 con số. Nguyên nhân nhìn
thấy được khi thử lẻ: VLM cho điểm rất phân cực (0.80 cho scene đúng, 0.00 cho
phần còn lại), nên đa số candidate hoà điểm 0.0 và sort ổn định giữ nguyên thứ
tự cũ. Giá trị của nó nằm ở việc BÁC BỎ dương tính giả, mà bộ gold này không
ép được điều đó — `correct_video_rate` đã là 1.000 từ đầu. **Không bật mặc
định.**

**D cho thấy hai biến gần như trực giao.** KIS/AVS của D bằng hệt B (expansion
không thêm gì khi đã có dịch), còn QA MRR cao nhất (0.291). Nên cân nhắc bật
expansion **chỉ cho QA**, tắt cho KIS/AVS — hiện `AIC_ENABLE_LLM_EXPANSION` là
cờ toàn cục, chưa tách theo task được.

### Kiểm kê prompt — đã gom về một nơi

Trước: 4 prompt nằm rải rác dạng hằng số module trong 3 adapter. Không biết
prompt nào đang chạy phiên bản nào, và ngân sách token bị hard-code tại chỗ
gọi (chính là cách QA hỏng 100%). Nay `online/prompts/registry.py` giữ cả 6,
mỗi cái khai **vai model** (`fast`/`reasoning`/`vlm`) thay vì tên model — tên
model là chuyện môi trường, còn "việc này có cần suy luận nhiều bước không" là
thuộc tính của chính việc đó.

### Hai cố vấn LLM mới

`FptWeightRecommender` — đề xuất trọng số nhánh, KHÔNG tự áp. Kiểm chứng thật
với truy vấn `bảng hiệu có chữ "Gừng cay muối mặn"`: `bm25_ocr=3.0`,
`ocr_fuzzy=2.5`, và tắt hẳn 7 nhánh còn lại kể cả `dense_visual=0.0`.

`FptEvidenceSelector` — lọc bằng chứng thô. Giữ nội dung cảnh, loại `HTV9 HD`,
`06:33:29`, `60 GIÂY`, `Lắk: Giảm 4 đơn vị hành chính...` như lớp phủ của đài.
Truy vấn không khớp thì trả `supports:false` thay vì dựng bằng chứng giả.

### Còn hở: LLM confidence bị bỏ khi xếp hạng QA

`QaProcessor._enhance_with_llm` thay `answer`/`answer_type`/`verifier_status`
nhưng **không đụng `joint_score`**, mà `joint_score` mới là thứ quyết định thứ
hạng. Nên `confidence` của LLM bị vứt.

Quan sát cụ thể trên submission thật, câu hỏi "Cột nước phun lên từ đâu?":

    rank 1  L21_V001,6933,nguồn nước gần người đàn ông
    rank 2  L21_V001,6933,người
    rank 3  L21_V001,6099,giếng          <- LLM chấm 0.95, đây là đáp án đúng

`joint_top1` = 0.083 phần lớn đến từ đây. Đây là thí nghiệm ranking nên phải
đo trước khi giữ, không sửa thẳng.

---

## TRAKE-CONSTRAINT-01 — Ràng buộc hình thức và cửa sổ chấm

**Trạng thái: DROP cả hai giả thuyết. Nút thắt là CHỌN FRAME.**

Nền: cấu hình B (dịch bật, expansion tắt, VLM tắt), `PYTHONHASHSEED=0`,
`--pipeline container --tasks TRAKE --max-per-video 0`.

### Giả thuyết 1 — ràng buộc hình thức lấn át độ liên quan: SAI

| cấu hình | mean_r_score |
|---|---|
| mặc định (`gap=0.002`, `order=0.6`) | 0.263 |
| `--trake-gap-penalty 0` | 0.263 |
| `--trake-order-weight 0` | 0.263 |
| cả hai = 0 | **0.231** |

Tắt riêng từng cái không đổi gì; tắt cả hai còn tệ hơn. Nghi vấn ghi ở mục
"Việc tiếp theo" của TRAKE T1–T4 là **sai**.

Nhìn lại thì rõ vì sao: thứ tự vốn ĐÃ là cổng cứng — `sequence_search.py:87`
bỏ qua mọi hit có `best_frame_idx < last_frame + min_gap_frames`. Còn
`max_gap_sec=300s` không bó gì khi khoảng cách bước thực chỉ ~12s. Hai tham số
này chưa bao giờ là thứ đang quyết định.

### Giả thuyết 2 — cửa sổ chấm quá hẹp: chỉ đúng một nửa, và không cứu được gì

| nửa cửa sổ | mean_r_score | trần lý thuyết | khoảng cách tới trần |
|---|---|---|---|
| 2s (mặc định) | 0.263 | 0.800 | 0.537 |
| 3s | 0.294 | 0.971 | 0.677 |
| 4s | 0.319 | 1.000 | 0.681 |
| 5s | 0.350 | 1.000 | 0.650 |

Đúng là cửa sổ ±2s chặn trần ở 0.800 — khoảng cách xa nhất từ frame gold tới
keyframe gần nhất là 92 frame = 3.07s, còn keyframe cách nhau trung vị 120
frame. Nhưng nới lên 4s đẩy trần **+0.200** mà điểm thật chỉ được **+0.056**:
khoảng cách tới trần *rộng ra*, không hẹp lại.

### Đính chính một kết luận cũ

`docs/21` từng ghi "7/35 bước TRAKE có candidate không tồn tại trong corpus —
trần cứng". **Sai.** Candidate luôn tồn tại; 7/35 là hệ quả của cận dưới 2s
trong `clamp(scene_duration * 0.5, 2.0, 7.0)`, không phải của dữ liệu.

### Giả thuyết 3 — beam bỏ sót chuỗi tốt hơn: SAI

Cài `search_sequences_dp` (quy hoạch động chính xác, cùng hàm mục tiêu với
beam) + `--trake-strategy beam|dp`.

| | mean_r_score |
|---|---|
| beam | 0.263 |
| DP thiếu `max_gap_sec` | 0.094 |
| DP đầy đủ ràng buộc | 0.231 |

**Phát hiện phụ quan trọng:** bản DP đầu bỏ mất `max_gap_sec` và tụt xuống
0.094 — TRAKE_E02 cho chuỗi trải 980 giây, nhảy tới frame 36764. `max_gap_sec`
là chặn CỨNG và đang làm việc thật; `gap_penalty_per_sec` là phạt MỀM và không
ảnh hưởng gì. Hai tham số này từng bị tôi gộp làm một nhóm "ràng buộc hình
thức" — chúng không cùng loại.

Với ràng buộc đầy đủ: **7/8 query giống hệt beam**, chỉ TRAKE_H01 khác
(0.250 vs 0.000). Dưới ngưỡng kết luận. Beam rộng 50 đã đủ chính xác.

Test: `tests/test_trake_dp.py` (6 test, gồm property test trên 60 input ngẫu
nhiên khoá "DP không bao giờ thua beam trên hàm mục tiêu").

### Còn lại đúng một giả thuyết

Hệ **đang chọn nhầm frame** dù ứng viên đúng nằm sẵn trong tầm với, và cả ba
giả thuyết về khâu tìm kiếm đều đã bị loại. Nguyên nhân còn lại chưa thử: điểm từng bước hiện là **tương đối** (điểm retrieval:
"frame này tốt hơn frame kia không") trong khi R-score hỏi một câu **tuyệt
đối** ("frame này có nằm trong cửa sổ không"). Thiết kế ở
`docs/22_TRAKE_CHAIN_SCORING.md`.

CLI mới: `--trake-gap-penalty`, `--trake-order-weight`, `--trake-missing-penalty`.

---

## METRIC-SPLIT-01 + QA-JOINT-01 (PR-1)

### Tách metric — làm trước mọi thứ khác

Không dùng một metric gộp làm căn cứ duy nhất nữa. Bốn task, mỗi task tách
thành tầng riêng. Ngay khi tách đã lộ hai thứ mà số gộp che mất:

| Task | Chỉ số tách | Giá trị | Đọc ra điều gì |
|---|---|---|---|
| KIS | `candidate_recall@20` | 1.000 | tìm kiếm ĐÃ xong việc |
| KIS | `top1_pairwise_accuracy` | 0.545 (6/11) | 11/12 truy vấn có đáp án ở hạng 1–2, chọn đúng chỉ hơn tung đồng xu |
| QA | `evidence_recall` | 0.833 | frame đúng gần như luôn có mặt |
| QA | `pairing_accuracy` | 0.875 (7/8) | **ghép KHÔNG sai** — có đủ hai mảnh thì chúng đã cùng dòng |
| TRAKE | `frame_oracle_coverage` | 0.800 | trần do dữ liệu |
| TRAKE | `frame_selection_accuracy` | 0.328 | trong phần với tới được chỉ chọn đúng 1/3 |
| AVS | `zero_result_rate` | 0.375 | 3/8 truy vấn trả về RỖNG |

`pairing_accuracy = 0.875` sửa lại chẩn đoán trước đó của tôi. Tôi đã viết QA
"ghép sai dòng"; thực ra ghép đúng, chỉ là **dòng đúng bị xếp thấp**.

**Đồng thời phát hiện `joint_top1` bị thổi lên.** Bản cũ tính
`answer_ok and rank == 1`, trong đó `answer_ok` là "CÓ DÒNG NÀO ĐÓ đúng cả ba"
còn `rank == 1` là "DÒNG ĐẦU đúng video+frame" — hai vế rơi vào hai dòng khác
nhau vẫn được tính đúng. Đã sửa thành "dòng đầu đúng cả ba".

### QA-JOINT-01 — bốn cách tính `joint_score`

`joint = evidence_conf × answer_conf × verifier_weight`, tính TRƯỚC khi LLM
chạy, nên `confidence` của LLM không bao giờ vào thứ hạng.

| mode | joint_top1 | QA MRR |
|---|---|---|
| `keep` (cũ) | 0.083 | 0.261 |
| `scale` (× conf) | 0.083 | 0.261 |
| `boost` (× (1+conf)) | **0.333** | **0.420** |
| `answer_first` (conf là chính) | 0.333 | 0.420 |
| `promote` (× 2, ĐỐI CHỨNG) | 0.333 | 0.418 |

**GIỮ `boost`.** joint_top1 gấp 4 lần (1/12 → 4/12), trên ngưỡng nhiễu 2 query.

**Kết quả quan trọng nhất là của biến thể đối chứng.** `promote` nhân một hằng
số, hoàn toàn không dùng `confidence`, mà cho kết quả y hệt `boost`. Nghĩa là
**`confidence` của LLM gần như không mang thông tin** — cái ăn điểm chỉ là việc
ưu tiên dòng do LLM trả lời lên trên dòng rule-based.

Điều này ảnh hưởng thẳng tới PR-6: **chưng cất `confidence` của LLM sang một
reranker nhỏ không có cơ sở.** Nếu chưng cất thì phải chưng cất tín hiệu khác.

### Một lỗi hạ tầng đã sửa

`--json-out` không tự tạo thư mục cha: script chạy xong, in đủ số, rồi ném
`FileNotFoundError` ở dòng cuối và mất trắng kết quả một lần chạy dài.

### Cảnh báo về tính tái lập

Một lần đọc được `evidence_recall = 0.750` trong khi bốn lần khác đều 0.833
(cùng cấu hình). Lần dị thường đó chính là lần bị lỗi ghi file. Chưa giải thích
được, và FPT rerank nằm trong đường xếp hạng nên kết quả không bit-reproducible.
**Cần một phép đo repeat-stability trước khi tin các chênh lệch 1 query.**

---

## AVS-GRADE-01 + REPEAT-STABILITY-01 (PR-2)

**Trạng thái: GIỮ `semantic_or_lexical`. Cổng từ vựng là thủ phạm, đã chứng minh trực tiếp.**

### Đo trước/sau cổng — trả lời dứt điểm câu hỏi nguyên nhân

Câu hỏi: *candidate đúng có thật sự bị cổng loại, hay 3 truy vấn kia vốn đã
không có candidate tốt?*

| | hard_gate | semantic_or_lexical |
|---|---|---|
| candidate TRƯỚC cổng | 48.4 | 48.4 |
| candidate SAU cổng | **5.0** | 48.4 |
| `correct_candidate_dropped_by_grade` | **1.000** | 0.000 |

Cổng vứt **90% candidate**, và ở **cả 8/8 truy vấn** nó loại mất candidate đúng
theo gold. Không phải pool nghèo — là cổng.

### Ablation, 5 lần mỗi variant, chế độ FPT thật

| variant | nDCG mean | min | max | sd | zero_result_rate |
|---|---|---|---|---|---|
| A `hard_gate` | 0.2995 | 0.2995 | 0.2995 | 0.0000 | 0.375 |
| B `no_gate` | 0.4401 | 0.4210 | 0.4528 | 0.0174 | 0.000 |
| C `soft` | 0.4384 | 0.3731 | 0.4881 | 0.0437 | 0.000 |
| D `semantic_or_lexical` | **0.4528** | 0.4528 | 0.4528 | **0.0000** | 0.000 |

Đọc theo khung đã định trước:

- **B > A** -> xác nhận cổng cứng là thủ phạm. Chênh 0.14, ngoài mọi biên nhiễu.
- **C ≈ B** -> KHÔNG kết luận được C hơn B: chênh 0.0017 trong khi sd của C là
  0.0437. Nếu chỉ chạy một lần, C ra 0.4881 và trông như thắng rõ — đó chính là
  con số tôi báo cáo trước khi có repeat-stability, và nó là đỉnh của dải.
- **D > C** -> giữ D. Cao nhất về trung bình VÀ tất định tuyệt đối.

### Vì sao D tất định còn C thì không

C để điểm ngữ nghĩa làm tín hiệu xếp hạng chính, nên nó hứng trọn dao động của
FPT rerank. D dùng ngưỡng để QUYẾT ĐỊNH GIỮ, rồi vẫn xếp hạng bằng công thức cũ
(grade chiếm ưu thế) nên miễn nhiễm.

Kết luận rộng hơn: **mở cổng làm lộ ra dao động vốn đã có sẵn.** Khi chỉ còn 5
candidate thì thứ tự của reranker gần như không đổi được gì; giữ 48 candidate
thì reranker mới thật sự quyết định đầu ra — và cùng lúc, sự bất định của nhà
cung cấp mới nhìn thấy được.

### REPEAT-STABILITY-01 — tách hai nguồn dao động

`scripts/repeat_stability.py`. Bắt buộc tách hai chế độ:

| chế độ | kết quả |
|---|---|
| `local` (AIC_FPT_ENABLED=false) | **sd = 0 trên mọi metric**, 1 dấu vân tay thứ hạng |
| `fpt` | dao động chỉ xuất hiện ở variant để ngữ nghĩa dẫn dắt (C, B) |

Hệ thống **tất định hoàn toàn khi không gọi mạng**. Toàn bộ dao động quan sát
được là **do nhà cung cấp**, không phải do thuật toán. Hai thứ này phải báo cáo
riêng, gộp lại là quy nhầm nguyên nhân.

Điều này cũng giải thích được lần đọc `evidence_recall = 0.750` dị thường ở
QA-JOINT-01: FPT nằm trong đường xếp hạng.

### Giới hạn của chính metric `correct_candidate_dropped_by_grade`

Chạy full với D cho `post_grade_candidate_count = 25.0` và
`correct_candidate_dropped_by_grade = 0.875` — tức D VẪN loại mất candidate
đúng ở 7/8 truy vấn, dù nó cho nDCG cao nhất.

Không mâu thuẫn, nhưng cho thấy metric tôi định nghĩa quá thô: nó chỉ hỏi "có
candidate đúng nào bị loại không", không phân biệt

- loại mất candidate LẼ RA đã lọt vào đầu ra  (có hại), với
- loại một candidate đúng nhưng thừa, vốn không bao giờ chen được vào top-3
  vì `max_per_video = 3` đã chặn  (vô hại).

Với trần 3 kết quả, phần lớn "drop" thuộc loại thứ hai. Muốn metric này dùng
được để ra quyết định thì phải giới hạn nó vào các candidate nằm trong tầm
đầu ra, hoặc đo sau khi đã bỏ trần `max_per_video`.

Quyết định giữ D vẫn dựa trên nDCG/`zero_result_rate` và độ ổn định — không
dựa trên chỉ số này.

### Còn một cổng nữa chưa đụng tới

`AvsConfig.max_per_video = 3`. Với dataset MỘT video, nó chặn cứng AVS ở 3 kết
quả — khớp đúng `result_count` quan sát được (0,0,0,1,2,3,3,3). Biến môi trường
`AIC_AVS_MAX_RESULTS_PER_VIDEO=20` có trong file env nhưng KHÔNG được code đọc.
Giữ nguyên trong PR này để chỉ đổi một biến, nhưng đây là ràng buộc bó tiếp theo
của AVS.

Test: `tests/test_avs_grade_gate.py` (6 test), gồm case hồi quy "phương tiện
cứu hộ" vs "xe cứu thương" — cùng nghĩa, gần như không chung token.

---

## EVAL-MULTIVIDEO-01 (PR-3)

**Trạng thái: KIS khoẻ thật. TRAKE sụp ở Stage A — tầng chưa từng được đo.**

### Đã dựng

`scripts/build_distractor_export.py` — L21_V002/V003 chỉ có ảnh keyframe + CSV
mapping (`n, pts_time, fps, frame_idx`), KHÔNG có scene manifest và KHÔNG có
ASR như V001. Scene được suy từ chính lưới keyframe: mỗi keyframe mở một scene
kéo tới keyframe kế tiếp. Ghi `model_name: csv_keyframe_grid:scene-fallback`
để không ai đọc nhầm đó là ranh giới ngữ nghĩa do detector cắt.

Kết quả: **765 scene / 3 video** (V001 217, V002 262, V003 286).

`scripts/embed_export_keyframes.py` — **855 vector** trên cả 3 video.

Bốn ràng buộc schema phát hiện khi dựng: `source_path` (repository đọc để phục
vụ `/v1/media`), `width`/`height` phải > 0, `segmentation_provenance` có schema
cố định, `transition_in/out` là enum không nhận `None`.

**Bẫy đáng ghi:** repository đọc keyframe **lồng trong scene**, không đọc
`keyframes.jsonl`. Cập nhật một file mà quên file kia thì vector nằm trên đĩa
còn nhánh dense im lặng bỏ qua toàn bộ video mới.

### Kết quả — và vì sao không dùng được

| | 1 video | 3 video | Δ |
|---|---|---|---|
| KIS R@1 | 0.500 | 0.583 | +0.083 |
| KIS `top1_pairwise_accuracy` | 0.545 | 0.636 | +0.091 |
| KIS MRR | 0.720 | 0.762 | +0.042 |
| QA MRR | 0.421 | 0.483 | +0.062 |
| QA `joint_top1` | 0.333 | 0.333 | 0.000 |
| TRAKE `correct_video_rate` | 1.000 | **0.875** | −0.125 |
| TRAKE `mean_r_score` | 0.263 | 0.231 | −0.031 |
| AVS `event_coverage` | 0.308 | 0.267 | −0.042 |

Mọi chênh lệch đều là **đúng 1 query**: KIS R@1 +0.083 = 1/12; `top1_pairwise`
6/11 -> 7/11; TRAKE 8/8 -> 7/8. Tất cả dưới ngưỡng 2 query. **Không có gì dịch
chuyển thật.**

Nguyên nhân: V002/V003 chưa có caption, nên chúng chỉ cạnh tranh ở
`dense_visual`. Chín nhánh còn lại (`bm25_caption`, `bm25_ocr`, `bm25_asr`,
`bm25_keyword`, `ocr_fuzzy`, `bm25_object`, ...) **không thể nhầm vì chúng
không nhìn thấy video mới**. Fusion vì thế bị chi phối bởi các nhánh miễn
nhiễm với distractor.

**Không được đọc KIS tăng là tin tốt.** Thêm distractor mà điểm tăng thì hoặc
là nhiễu, hoặc là chuẩn hoá điểm đổi theo pool — cả hai đều không phải cải
thiện năng lực.

### Tín hiệu thật duy nhất

`TRAKE.correct_video_rate` rời khỏi 1.000 lần đầu tiên. Đây là chỗ duy nhất
nhầm lẫn xuyên video biểu hiện được, và nó chỉ có thể xảy ra từ khi có video
thứ hai — đúng lý do PR-3 tồn tại.

### Giai đoạn 2 — distractor ĐẦY ĐỦ (548 keyframe đều có caption)

| | 1 video | 3v thị giác | 3v đầy đủ | Δ |
|---|---|---|---|---|
| KIS R@1 | 0.500 | 0.583 | **0.500** | 0.000 |
| KIS R@20 | 1.000 | 1.000 | **1.000** | 0.000 |
| KIS MRR | 0.720 | 0.762 | **0.718** | −0.003 |
| KIS `top1_pairwise` | 0.545 | 0.636 | **0.545** | 0.000 |
| QA `joint_top1` | 0.333 | 0.333 | 0.333 | 0.000 |
| QA `evidence_recall` | 0.833 | 0.833 | 0.833 | 0.000 |
| QA `answer_accuracy` | 0.583 | 0.583 | 0.500 | −0.083 |
| **TRAKE `correct_video_rate`** | **1.000** | 0.875 | **0.625** | **−0.375** |
| TRAKE `mean_r_score` | 0.263 | 0.231 | 0.144 | −0.119 |
| AVS `P@100` | 0.500 | 0.500 | 0.295 | −0.205 |
| AVS nDCG@100 | 0.453 | 0.462 | 0.401 | −0.051 |

### Kết luận 1 — KIS không suy chuyển, NHƯNG phép thử quá dễ

> **ĐÍNH CHÍNH (PR-4B).** Kết luận "KIS khoẻ thật" dưới đây KHÔNG được dữ liệu
> ủng hộ: V002/V003 chỉ có caption, nên 6/10 nhánh của KIS không thể trả về gì
> ngoài V001. Xem mục PR-4A+PR-4B.

KIS **không suy chuyển một chút nào**: R@1, R@5, R@20, `top1_pairwise` giống
hệt; MRR lệch 0.003. Thêm 548 scene đối thủ có caption thật mà không mất gì.

Đây là kết quả TÍCH CỰC và nó cũng nói rằng bài toán còn lại của KIS
(`top1_pairwise = 0.545`) là bài toán phân biệt tinh, **không** phải bài toán
nhiễu xuyên video.

### Kết luận 2 — TRAKE sụp, và sụp ở tầng CHƯA TỪNG được đo

Phân rã:

| | 1 video | 3v đầy đủ |
|---|---|---|
| đúng video | 8/8 | **5/8** |
| `mean_r` toàn bộ | 0.263 | 0.144 |
| `mean_r` **chỉ trên video đúng** | 0.263 | **0.230** |

Chọn frame gần như không tệ đi (0.263 -> 0.230). **Gần như toàn bộ mất mát nằm
ở Stage A — chọn video.** Ba query hỏng: `TRAKE_E01`, `TRAKE_E02`, `TRAKE_H02`
— hai trong đó là query DỄ.

Điều này đảo ngược thứ tự ưu tiên của TRAKE. Mọi thí nghiệm TRAKE từ trước tới
nay — cửa sổ chấm, DP vs beam, `gap_penalty`, `frame_refinement` — đều tối ưu
Stage B/C, trong khi Stage A đúng 100% **do không có gì để nhầm**. Nó chưa bao
giờ được thử thách, và giờ nó là nút thắt lớn nhất.

`frame_selection_accuracy` đã được sửa để tính TRÊN các query đúng video; tính
gộp thì nó là chỉ số "chọn video" trá hình.

### Kết luận 3 — AVS mất nhiều nhất theo tỉ lệ

`P@100` 0.500 -> 0.295, mất 41% giá trị. `zero_result_rate` vẫn 0.000, nên PR-2
vẫn đứng vững; cái mất là độ chính xác khi có đối thủ thật.

### Hai lỗi hạ tầng đã gặp và sửa

**HTTP 429 — 50 RPM.** `concurrency=6` làm hỏng 233/545 keyframe. Hạ
`concurrency` KHÔNG phải cách sửa: nó chặn số lệnh gọi ĐỒNG THỜI chứ không chặn
TỐC ĐỘ. Đã thêm cổng tốc độ thật (`--rpm`, cấp khe theo `60/rpm`). Hệ quả cho
kế hoạch: VLM rerank với `top_k=20 × 3 frame = 60 lệnh gọi/truy vấn` **vượt hạn
mức 50 RPM chỉ với MỘT truy vấn** — thêm một lý do độc lập để không bật nó.

**Caption keyframe và caption scene có schema KHÁC NHAU.** `caption_type` ở
keyframe là enum `short|detailed|tags|crop` (không có `visual`), không nhận
`evidence_keyframe_ids`; ở scene thì ngược lại. Dùng nhầm một dạng cho cả hai
làm eval hỏng lúc NẠP — tức sau khi đã trả tiền cho toàn bộ 548 lệnh gọi VLM.
Cache theo `sha256(ảnh+prompt+model)` cứu được lần dựng lại.

---

## PR-4A + PR-4B — Khoá phép đo, rồi sửa TRAKE Stage A

### PR-4A — hai lỗi của chính công cụ đo

`repeat_stability.py` nay báo cáo `mean/min/max/sd`, ranking fingerprint,
**query nào đổi hạng**, và **nhánh nào hỏng**. Ngay khi bật, nó tự phơi ra hai
vấn đề:

**1. Chỉ số "nhánh hỏng" của tôi sai.** Nó đếm `disabled` là hỏng và báo động
giả **40/40 lượt**. `disabled` là định tuyến CÓ CHỦ Ý — truy vấn không có manh
mối chữ/lời nói thì OCR/ASR nhận trọng số 0 và nhánh không chạy (ROUTE-01,
`allow_zero_modality`). Báo động giả kiểu này che mất hỏng thật.

**2. Sau khi sửa, có hỏng THẬT:** `dense_visual` lỗi **1/40 lượt** ở chế độ
`fpt`, **0/40** ở `local`. Đó là lệnh gọi dịch VI→EN thỉnh thoảng hỏng. Hệ
thống hành xử đúng thiết kế (thà `failed` còn hơn lặng lẽ encode tiếng Việt),
nhưng ~2.5% truy vấn mất nhánh mạnh nhất. Đáng cân nhắc retry riêng cho lệnh
gọi dịch hoặc cache bền qua các lần chạy.

`fpt` cũng cho **2 dấu vân tay** dù không metric nào dao động — thứ hạng đổi ở
đâu đó mà chỉ số tổng không thấy. Đúng công dụng của fingerprint.

Định nghĩa `local` đã siết lại: tắt MỌI thứ cần mạng, không chỉ
`AIC_FPT_ENABLED` — vì container fail-fast nếu bật dịch mà không có provider.
Ghi rõ: `local` KHÔNG đo cùng cấu hình với `fpt`; nó đo độ tất định của phần
máy móc chạy tại chỗ.

### PR-4B — `duplicate_penalty` phạt chính TRUE POSITIVE

Chẩn đoán trực tiếp trên `rank_videos` cho ba query hỏng: video ĐÚNG thắng áp
đảo ở `context` (0.892 / 0.844 / 0.924 so với 0.23–0.32) nhưng vẫn thua.

Phân rã TRAKE_E01 (V002 1.398 vs V001 1.307):

| thành phần | tác động |
|---|---|
| `ordering` V002 0.67 vs V001 0.33 | V002 **+0.204** |
| `context` V001 0.892 vs V002 0.308 | V001 +0.234 |
| `duplicate` V001 0.50 vs V002 0.25 | V001 **−0.125** |

Cơ chế: sự kiện của một diễn biến CÓ THẬT thì tập trung gần nhau nên nhiều
step trỏ về cùng scene và bị phạt; hit của video SAI thì rải rác ngẫu nhiên
nên thoát. Hình phạt vốn nhắm "đoạn tóm tắt đầu bản tin" lại bắn vào đúng thứ
nó phải bảo vệ.

`ordering` còn bị tính HAI LẦN: Stage B đã có cổng cứng thứ tự
(`sequence_search.py:87`), ở Stage A nó lại được tính như điểm mềm trên một
proxy mong manh ("hit tốt nhất mỗi step").

### Ablation

| variant | `video_recall@1` | `mean_r_score` | `mean_r_on_correct` |
|---|---|---|---|
| A hiện tại | 0.625 | 0.144 | 0.230 |
| **B `duplicate_penalty=0`** | **1.000** | **0.263** | 0.263 |
| C `context_weight=1.5` | 1.000 | 0.263 | 0.263 |
| D cả hai | 1.000 | 0.263 | 0.263 |

Nghi ngờ rằng C chỉ thắng nhờ dữ liệu lệch (xem dưới) đã bị BÁC BỎ: chạy lại
với chỉ 2 nhánh công bằng, C vẫn đạt 1.000. B và C không phân biệt được trên 8
query. **Chọn B** vì nó GỠ BỎ một tác hại đã chứng minh, thay vì thêm một
trọng số đã tinh chỉnh.

Kết quả phụ của phép đo công bằng: tắt 8 nhánh lại CẢI THIỆN TRAKE
(0.263 -> 0.287). Chúng đang thêm nhiễu cho task này.

### Đính chính kết luận về KIS ở PR-3

Bản trước ghi *"KIS khoẻ thật, không phải ảo giác một-video"*. **Không được dữ
liệu ủng hộ.** L21_V002/V003 CHỈ có caption:

| trường | V001 | V002 | V003 |
|---|---|---|---|
| captions | 216/217 | 262/262 | 286/286 |
| keywords | 170/217 | **0** | **0** |
| action_tags | 136/217 | **0** | **0** |
| asr_segments | 210/217 | **0** | **0** |
| ocr | 98/217 | **0** | **0** |
| objects | 170/217 | **0** | **0** |

Chỉ **2/10 nhánh có cạnh tranh thật**. Sáu nhánh còn lại về mặt cấu trúc không
thể trả về gì ngoài V001. KIS giữ nguyên điểm một phần vì phần lớn nhánh của
nó KHÔNG CÓ CÁCH NÀO NHẦM.

Điều này làm kết luận TRAKE MẠNH HƠN: nó vẫn tụt 1.000 -> 0.625 dù V001 đang
có lợi thế cấu trúc đó.

Đã thêm `--disable-branch` để chạy phép đo công bằng.

### Kết quả cuối trên 3 video

| | 1 video | 3v trước | 3v sau PR-4B |
|---|---|---|---|
| TRAKE `mean_r_score` | 0.263 | 0.144 | **0.263** |
| TRAKE `frame_selection_accuracy` | 0.328 | 0.180 | **0.328** |
| AVS nDCG@100 | 0.453 | 0.401 | 0.428 |
| AVS P@100 | 0.500 | 0.295 | 0.358 |
| KIS R@1 / MRR | 0.500 / 0.720 | 0.500 / 0.718 | 0.500 / 0.718 |

**TRAKE khôi phục chính xác về mức một-video.**

AVS và QA cũng khác giữa hai lần chạy, nhưng KHÔNG được quy cho PR-4B:
`duplicate_penalty` chỉ tồn tại trong `VideoRetrieverConfig` của TRAKE. Kiểm
tra per-query cho thấy đúng 1 query đổi ở mỗi task (`VQA_M01`, `AVS_M03`) —
đó là dao động giữa các lần chạy, không phải tác dụng của thay đổi. Nó cũng
cho thấy AVS/QA trên đa video kém ổn định hơn mức đã đo trên một video, nên
cần chạy repeat-stability lại cho hai task đó trước khi kết luận gì thêm.

### Không làm được: holdout theo video

Cả 40 gold query đều target L21_V001. Không có query nào cho V002/V003 nên
không có gì để hold out. Muốn có thì phải viết gold mới — việc thủ công cần
người xem video.

---

## PR-4C — Cân bằng dữ liệu distractor + hạ tầng chuẩn bị

### Vấn đề

Sau PR-3, L21_V002/V003 chỉ có caption. Trong 10 nhánh retrieval, **chỉ 2 có
cạnh tranh thật**; sáu nhánh còn lại về mặt cấu trúc không thể trả về gì ngoài
L21_V001. Mọi phép đo "hệ khoẻ khi có distractor" vì thế dễ hơn thực tế, và
kết luận "KIS khoẻ" ở PR-3 đã phải rút lại.

### Đã sinh

`scripts/enrich_export_keyframes.py` — MỘT lệnh gọi VLM cho mỗi keyframe trả
về caption + object + OCR + action, thay vì bốn lượt riêng (ảnh phải mã hoá
base64 gửi lại mỗi lần, nên gộp là khác biệt lớn dưới trần 50 RPM).
`keywords` KHÔNG gọi model — suy từ nhãn object đúng cách `offline/assemble.py`
làm.

Độ phủ scene sau khi enrich V002/V003 (544/548 thành công):

| trường | V001 (trước) | V002 | V003 |
|---|---|---|---|
| keywords | 78% | 99% | 99% |
| objects | 78% | 99% | 99% |
| ocr | 45% | 80% | 78% |
| asr | 97% | **0%** | **0%** |

### Đổi một thiên lệch lấy một thiên lệch khác thì vô nghĩa

Distractor giờ GIÀU HƠN video gốc ở object/OCR/keyword. Không giải quyết được
gì — chỉ đảo chiều thiên lệch. Nên enrich lại V001 bằng CÙNG prompt.

**Nhưng ghi đè là sai.** V001 được enrich bằng prompt tinh chỉnh riêng ở
CAPTION-ENRICH-01, và một số gold query là dạng OCR ("bảng hiệu có chữ ...").
Thay dữ liệu đã kiểm chứng bằng dữ liệu của prompt tổng quát có thể làm hỏng
đúng những truy vấn đang dùng để đo — tức tự tạo ra một "suy giảm" không liên
quan gì tới hệ thống.

Nên script chuyển sang **HỢP NHẤT**: giữ nguyên object/OCR cũ, chỉ thêm cái
mới chưa có. Kiểm chứng trên 5 keyframe: không mất gì, `ocr` thêm được
`HTV9 HD`/`06:30:14`, object thêm biến thể chi tiết hơn.

### ASR không cân bằng được

V002/V003 không có audio. `bm25_asr` vĩnh viễn là nhánh chỉ-V001, nên mọi phép
đo đa video phải tắt nó (`--disable-branch bm25_asr`). Đây là giới hạn của bộ
dữ liệu, không phải lựa chọn thiết kế.

### Hạ tầng cắt thời gian chạy

**Cache dịch trên đĩa** (`storage/cache/query_translation`). Mỗi lần eval trước
đây dịch lại cả 40 truy vấn dù nhiệt độ 0 nên kết quả tất định. Cũng che luôn
lỗi dịch ~1/40 lượt đã đo ở PR-4A: lần chạy sau dùng bản dịch đã có thay vì
tung xúc xắc lại.

**Retry trong adapter dịch.** `FptClient` chỉ retry lỗi HTTP; "trả về chuỗi
rỗng" nó coi là thành công — đó chính là cách `dense_visual` hỏng. Nay thử 3
lần rồi mới bỏ cuộc.

**`scripts/validate_gold.py`** — kiểm gold TRƯỚC khi chạy eval. Viết nó lại
phát hiện một điều tưởng đã biết: **bốn task dùng bốn shape khác nhau**. AVS
dùng `relevant_intervals` (kèm `relevance_grade`), KIS/VQA dùng
`target_intervals`, VQA dùng `question_vi` chứ không `query_vi`. Bản đầu của
validator giả định chung một tên và báo sai 8 lỗi trên file đang chạy tốt.

**Tách chỉ số theo video** trong `eval_tasks`, tự bật khi gold có nhiều
`target_video`. Cần cho holdout ngay khi có query của video mới.

**`AIC_AVS_MAX_RESULTS_PER_VIDEO`** giờ được đọc thật (trước đây có trong env
nhưng không code nào dùng, đang chặn cứng AVS ở 3 kết quả).

**`correct_candidate_dropped_by_grade`** siết vào tầm đầu ra — bản cũ đếm cả
candidate đúng-nhưng-thừa nên báo 0.875 cho chính cấu hình tốt nhất.

Tài liệu: `docs/23_GOLD_QUERY_FORMAT.md`.

---

## PR-4C (kết) — Bàn cân đối xứng hoàn toàn, và baseline mới

Người dùng cung cấp ASR đã phiên âm sẵn cho L21_V002/V003
(`faster-whisper:large-v3`, 253 và 254 đoạn) nên không tốn lệnh gọi API nào.
`scripts/ingest_asr.py` chiếu vào scene.

### Độ phủ sau khi cân bằng

| trường | V001 | V002 | V003 |
|---|---|---|---|
| captions / keywords / objects | 99–100% | 98–100% | 98–100% |
| ocr | 83% | 79% | 78% |
| **asr** | **97%** | **98%** | **98%** |
| action_tags | 77% | 47% | 49% |

`--disable-branch bm25_asr` không còn cần. Cả 10 nhánh đều có cạnh tranh thật.

`action_tags` còn lệch vì V001 giữ cả nhãn cũ lẫn mới sau bước hợp nhất; nhánh
`bm25_action` phần lớn vẫn trả `empty` nên chưa đáng cân tiếp.

### Baseline: trước cân bằng -> sau cân bằng (cùng cap=3)

| | trước | sau | Δ |
|---|---|---|---|
| KIS R@1 | 0.500 | **0.583** | +0.083 |
| KIS R@5 | 0.917 | **1.000** | +0.083 |
| KIS `top1_pairwise_accuracy` | 0.545 | **0.700** | +0.155 |
| QA `answer_accuracy` | 0.583 | 0.417 | −0.167 |
| QA `evidence_recall` | 0.833 | 0.750 | −0.083 |
| TRAKE `video_recall@1` | 1.000 | 0.875 | −0.125 |
| TRAKE `mean_r_on_correct_video` | 0.263 | 0.300 | +0.037 |
| AVS nDCG@100 | 0.428 | 0.445 | +0.016 |

**KIS đi lên, QA/TRAKE đi xuống.** Không mâu thuẫn: metadata giàu hơn giúp KIS
phân biệt top-2 (đúng chỗ nó đang yếu), nhưng cũng làm distractor cạnh tranh
được ở QA evidence và ở Stage A của TRAKE — đúng mục đích của việc cân bằng.

### Ba lỗi hạ tầng trong đợt này

**Tôi tự tạo ra một confound.** Wire `AIC_AVS_MAX_RESULTS_PER_VIDEO` (biến có
trong env nhưng chưa code nào đọc) đã đổi `max_per_video` 3 -> 20 ngay ở lần
chạy baseline kế tiếp, khiến `result_count` nhảy từ 3–9 lên 15–50. Lần chạy đó
đổi HAI biến cùng lúc nên bảng AVS không đọc được.

> **Nguyên tắc: wire một biến môi trường chưa từng được đọc là một THAY ĐỔI
> HÀNH VI, không phải dọn dẹp.** Nó phải đi kèm ablation riêng.
> `max_per_video` nay là biến của PR-5.

**Hai lỗi schema chỉ lộ ra lúc NẠP**, sau khi đã ghi file:

1. Đoạn ASR phải nằm TRONG khoảng scene, không chỉ giao nhau. V001 làm đúng:
   giữ nguyên text, CẮT mốc theo biên scene — cùng câu xuất hiện ở S0002
   [10.29,11.43], S0003 [11.43,13.70], S0004 [13.70,15.27].
2. Id nguồn: file whisper dùng `L21_V002_A000000`, schema đòi `..._ASR000000`.

Đây là lần thứ ba trong đợt một script ghi thành công rồi mới hỏng ở tầng
validate (trước đó: schema caption keyframe vs scene). Đáng thêm một bước "nạp
thử" ngay trong script ghi.

### `P@100` không dùng để so đa video được

`AVS.P@100` tụt 0.358 -> 0.136, nhưng đó là **hiện tượng số học**: gold chỉ nằm
ở V001, mà `max_per_video=3` cho mỗi video 3 suất, nên với 3 video thì trần
trên của precision đã là 0.333. Thêm video là tự động kéo P@100 xuống bất kể
chất lượng.

nDCG có tính vị trí nên không bị vậy (0.428 -> 0.445). **Dùng nDCG, không dùng
P@100, khi so giữa các bộ có số video khác nhau.**

---

## BRANCH-TIMEOUT-01 — Nhánh mạnh nhất biến mất trong im lặng

**Nguồn nhiễu lớn nhất của cả đợt, và nó KHÔNG phải thuật toán.**

`dense_visual` ở trạng thái `timeout` tại **25–60% truy vấn**, thay đổi giữa
các lần chạy cùng cấu hình (14/40 rồi 23/40). Đó là lý do hai lần chạy giống
hệt nhau cho KIS MRR 0.753 và 0.653, và một truy vấn rơi từ hạng 1 xuống
*không tìm thấy*.

### Chẩn đoán — không phải cái tôi nghi

Tôi nghi lệnh gọi dịch VI→EN. **Sai**: cache dịch có 68 mục và đang hoạt động.

| | chạy riêng | chạy cùng 10 nhánh |
|---|---|---|
| `dense_visual` | ~200ms | 1.6–4.6s |
| `ocr_fuzzy` | — | 1.3–3.9s |

Hai nhánh bám sát nhau từng mili-giây (1643/1330, 3724/3666, 3609/3552) — dấu
hiệu **tranh chấp CPU**. Cả hai đều nặng CPU: `dense_visual` encode CLIP rồi
quét 855 vector, `ocr_fuzzy` khớp mờ trên 765 scene.

Deadline 3000ms được chọn khi corpus có **217 scene**. Ở 765 scene thì quá chặt.

### Sửa

`AIC_BRANCH_TIMEOUT_MS`, mặc định **8000ms** (chọn theo max đo được 4.6s).
Kết quả: **0/8 timeout**, trước là 3/6.

Kèm một lỗi khác phát hiện lúc đọc code: `resolve_timeout_ms` dùng
`override.timeout_ms` bất cứ khi nào request chạm tới nhánh — mà
`BranchRuntimeOptions` mang default 3000. Nên chỉ cần request đổi `weight` của
một nhánh là **vô tình ép nhánh đó về 3000ms**. Nay chỉ dùng override khi
`timeout_ms` được đặt tường minh (`model_fields_set`).

### Ảnh hưởng ngược

Mọi số đo từ khi bật dịch VI→EN đều có một số lượng KHÔNG XÁC ĐỊNH truy vấn
thiếu nhánh dense. Kết luận định tính vẫn đứng (chênh lệch lớn hơn nhiễu này
nhiều), nhưng **con số cụ thể không đáng tin tới chữ số thứ hai**. Không kiểm
ngược được các lần đo một-video vì `branches` chỉ mới ghi từ PR-4A.

**`ocr_fuzzy` chậm ngang `dense_visual`** mà đóng góp của nó chưa từng đo
riêng — ứng viên ablation rõ ràng.

---

## OCR-BACKFILL-01 — Chữ có thật, prompt mới là thứ hỏng

150/765 scene (20%) không có OCR nào. Sáu truy vấn gold khai
`required_modalities: ["ocr"]` rơi đúng vào phần thiếu, tức không thể ăn điểm
dù hệ thống hoàn hảo.

### Nguyên nhân

Prompt enrich tổng quát bắt model **vừa đọc chữ vừa định vị bbox** trong một
JSON. Ép hai việc cùng lúc thì nó bỏ hẳn phần chữ. Cùng ảnh, prompt chỉ đòi
CHỮ thì tìm thấy ở **6/6** frame "không có chữ", gồm cả tiêu đề bản tin đầy đủ.

Nên: **ưu tiên chữ, hy sinh bbox**. Điền khung toàn ảnh, ghi
`model_name: "<model>:ocr-textonly"` để không ai đọc nhầm là toạ độ thật.
Không nhánh nào đọc bbox (`bm25_ocr`/`ocr_fuzzy` chỉ dùng `text`).

### Chọn model bằng đo, không theo catalog

Catalog chỉ ghi một VLM, nhưng dò thực tế: `gemma-3-27b-it` và
`gemma-4-31B-it` **đều nhận ảnh**; `GLM-5.2` và `gpt-oss-120b` thì không.
Chọn `gemma-4-31B-it` vì đọc chính xác hơn trên cùng frame (`"HTV9 HD"` liền,
`"nhiều lĩnh vực"` đúng chính tả thay vì `"nghiêu lĩnh vực"`). Lợi thêm: hạn
mức RPM riêng, không đụng `Qwen2.5-VL` đang bận.

### Kết quả

| | trước | sau |
|---|---|---|
| độ phủ OCR (cả 3 video) | 84% / 80% / 79% | **100% / 100% / 100%** |
| truy vấn không thể ăn điểm | 6/120 | **0/120** |

### Ba phép kiểm chống bịa

177/177 frame đều tìm thấy chữ — con số hoàn hảo là thứ phải nghi:

1. **0/181 chuỗi OCR trùng nhau.** Model bịa thường lặp cùng một khuôn.
2. **Bắt được chữ chạy bị cắt dở** (`"g thẳng leo thang giữa Israel..."`).
   Model bịa sẽ viết câu hoàn chỉnh.
3. **Đồng hồ tăng đơn điệu 65/65, 52/52, 59/61** qua các lệnh gọi ĐỘC LẬP.
   Không cách nào bịa ra một chiếc đồng hồ nhất quán theo thứ tự frame.

Đồng hồ trôi nhanh hơn thời lượng video (1763s so với 1217s) là tính chất của
NGUỒN — bản tin đã dựng cắt nên đồng hồ chạy theo giờ phát sóng gốc.

---

## Bộ gold ba video — 120 truy vấn

| | khớp TB | Easy/Medium/Hard | thiếu modality |
|---|---|---|---|
| V001 | 0.436 | 0.49 / 0.44 / 0.38 | 0 |
| V002 | 0.349 | 0.47 / 0.31 / 0.28 | 0 |
| V003 | 0.404 | 0.53 / 0.39 / 0.29 | 0 |

Cả ba sạch, mỗi bộ 40 truy vấn (12 KIS / 12 VQA / 8 TRAKE / 8 AVS). Thang độ
khó của V002/V003 **đơn điệu hơn V001** (V001 có Medium 0.44 và Hard 0.38 sát
nhau nên nhãn ít ý nghĩa).

**Holdout theo video giờ làm được lần đầu.** `eval_tasks` tự tách chỉ số theo
`target_video` khi gold có nhiều video.

Một cảnh báo khi đọc: chênh lệch khớp TB giữa các video **không** đo được độ
khó của truy vấn — nó lẫn cả việc caption/OCR của video đó giàu hay nghèo chữ.

---

## Việc tiếp theo

**BM25-01 — concept coverage.** Đây là kết luận chung của cả hai thí nghiệm
trên: bệnh nằm ở token-overlap có nền nhiễu cao, không ở chỗ chọn modality.
Query `"cột nước phun lên từ lòng đất"` phải bị tách thành 3 nhóm khái niệm
bắt buộc (nước / phun lên / từ mặt đất); candidate chỉ khớp `đất` phải bị phạt
coverage.

**TRAKE — đã chẩn đoán xong, xem mục T1–T4 ở trên.** Chết ở T3: ứng viên
đúng có sẵn cách gold 1.1s cho 34/35 bước, nhưng chuỗi được chọn lệch trung
bình 254s. Việc tiếp theo là tắt thử `ordering_weight`/`gap_penalty_per_sec`
để xem ràng buộc hình thức có đang lấn át độ liên quan không — thay đổi ở
tầng online, KHÔNG phải trích lại dữ liệu.

**Ba thí nghiệm ranking đã chạy đều DROP.** Điểm chung: mỗi lần số liệu tổng
lại chỉ về một chỗ khác với chỗ giả thuyết chỉ. Đó là lý do quy trình bắt
buộc error analysis trước khi quyết giữ/bỏ.
