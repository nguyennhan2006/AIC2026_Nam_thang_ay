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
