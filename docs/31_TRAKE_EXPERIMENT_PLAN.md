# 31 — Kế hoạch thí nghiệm TRAKE

Lập 2026-08-09. Nguồn phương pháp: bản tổng hợp paper của người dùng (DANTE,
MADTempo, temporal-decay beam, ABTS/neighbor, Q2E, Vortex, Drop-DTW, EA-VTR).

Thứ tự phase: **A → D → B → C** (§2.3). Tôi từng đề xuất đổi thành C trước A dựa
trên lập luận về trần retrieval, rồi chính thí nghiệm `TRK-C07` bác bỏ nó — xem
§2 để khỏi lặp lại lập luận đã sai đó.

---

## 1. Chẩn đoán làm cơ sở — đọc trước khi chạy bất kỳ thí nghiệm nào

### 1.1 Có HAI trần khác nhau, đừng lẫn

| Chỉ số | Trả lời câu hỏi | Giá trị |
|---|---|---:|
| `frame_oracle_coverage` | **Corpus** có keyframe gần mốc gold không? | 0.845 |
| `gold_region_recall@100` | **Retrieval** có đưa nó vào pool 100 candidate không? | **0.773** |

Cái đầu là thuộc tính dữ liệu, độc lập hoàn toàn với retrieval. Cái sau mới là
trần mà sequence solver được phép làm việc. Lẫn hai cái là gốc của việc xếp nhầm
thứ tự phase.

Đo bằng `scripts/diagnose_trake_stage_b.py` trên đúng đường Stage B thật (không
dedup, không `KisProcessor.rank`, không cắt `top_k`).

### 1.2 `complete_chain_rate = 0` KHÔNG phải lỗi solver

```
buoc co vung gold trong pool  : 85/110  = 77.3%
truy van co DU moi buoc       :  9/24   = 37.5%   <- TRAN CUNG
```

Chỉ **9/24** truy vấn có đủ mọi event trong pool. Không thuật toán ghép chuỗi nào
— DANTE, MADTempo, beam — dựng được chuỗi từ ứng viên không tồn tại.

> **Hệ quả bắt buộc nhớ:** Phase A chạy hoàn hảo cũng chỉ đưa `complete_chain_rate`
> lên **tối đa 0.375**. Dùng chỉ số này để chấm solver sẽ kết luận nhầm là solver
> kém. **Chấm solver bằng `mean_r_score`.**

Số bước thiếu tập trung ở vài truy vấn, không rải đều:

```
V003_TRAKE_H03  thieu 3/5     V002_TRAKE_H01  thieu 2/5
V002_TRAKE_M03  thieu 3/5     V001_TRAKE_M03  thieu 2/5
V001_TRAKE_H01  thieu 3/4     V001_TRAKE_M01  thieu 2/5
```

### 1.3 Dư địa thật của solver

```
mean_r_score hien tai   0.254
tran do retrieval       0.773
con lay duoc            0.519
```

Đây mới là con số để đánh giá Phase A.

### 1.4 Lỗi nằm ở chọn SCENE, không phải chọn frame trong scene

Trên 90 bước có video đúng:

```
du doan roi DUNG scene chua frame gold :  35  (38.9%)
du doan roi SCENE KHAC                 :  55  (61.1%)

chon dung keyframe GAN NHAT gold       :  27/90 = 30.0%
khoang cach toi gold  p50=104  p90=590 frame
keyframe gan nhat CO THE  p50=33  p90=70 frame
```

91% scene chỉ có **1 keyframe**, nên "chọn frame trong scene" gần như không tồn
tại như một bài toán trên corpus này.

### 1.5 Tầm với của tinh chỉnh cục bộ (T4/ABTS) bị chặn bởi độ dài scene

Scene p50 = 120 frame. Cửa sổ cục bộ bán kính R chỉ cứu được bước có `|lệch| ≤ R`:

```
R=+/-4   frame  ->  3.3%      R=+/-64   frame  -> 32.2%
R=+/-8   frame  ->  6.7%      R=+/-128  frame  -> 53.3%   (vuot bien scene)
R=+/-16  frame  -> 13.3%      R=+/-256  frame  -> 74.4%   (het la "cuc bo")
R=+/-32  frame  -> 20.0%
```

T4 chỉ có nghĩa **sau khi** solver đã chọn đúng scene. Chạy nó trước là đo trần
20%.

### 1.6 Bằng chứng sẵn có cho T8 (fusion) — không cần thí nghiệm để biết RRF hỏng

Đo trực tiếp: với truy vấn `"Ông NGUYỄN TRANG SƯ THÀNH PHỐ HỒNG NGỰ TỈNH ĐỒNG THÁP"`,
nhánh `bm25_ocr` chấm scene đúng **27.06** so với **6.5** của kẻ đứng sau — chắc
chắn tuyệt đối. Scene đó **không vào nổi top-20**, vì RRF chỉ tính thứ hạng nên
nhánh đó vẫn chỉ được `1/(60+1)` như một nhánh đoán mò. Nâng trọng số nhánh lên
5.0 → **hạng 1**.

Cùng cơ chế làm `color_search` vô dụng: tắt hẳn nhánh cho kết quả **giống hệt** bật.

Đây khớp với bằng chứng độc lập của Q2E (RRF 62.91 R@10 so với InvEntropy 75.76).

### 1.7 T6 đã được cài một phần

[search.py:346](../online/services/search.py#L346) đã gọi
`compute_modality_weights(event.text, ...)` **riêng cho từng event**, và
`plan.search_options.temporal.step_modality_weights` cho phép ghi đè theo bước.

Nên T6-B ("cùng trọng số cho mọi event") **không phải hiện trạng**. T6-C là mở
rộng cái đã có chứ không phải xây mới.

---

## 2. Thứ tự phase — đã chạy `TRK-C07` và nó ĐẢO NGƯỢC đề xuất hiệu chỉnh

### 2.1 Hai lần sửa của chính tài liệu này

**Sửa lần 1 (số sai).** Bảng §1.1–1.2 ban đầu ghi `gold_region_recall@100 = 0.718`
và `6/24`. Sai: `scripts/diagnose_trake_stage_b.py` mặc định dùng
`build_service` của `eval_kis`, mà nó dựng **bộ nhánh khác** hệ đang chạy — có
`bm25_ocr` + `ocr_fuzzy` (production đã tắt cả hai) và thiếu hẳn
`bm25_object`/`bm25_action`/`color_search`. Script đã được sửa để dùng
`build_container`; số đúng ở §1.1–1.2.

**Sửa lần 2 (kết luận sai).** Đề xuất "C trước A" dựa trên lập luận trần. Chạy
thật thì lập luận đó sai — xem `TRK-C07` dưới đây.

### 2.2 `TRK-C07` — nâng `candidate_limit` · **DROP**, và nó quyết định thứ tự phase

| `candidate_limit` | truy vấn đủ mọi bước (TRẦN) | `mean_r` | `frame_sel` | `chain` |
|---:|---:|---:|---:|---:|
| 100 (BASE) | 9/24 = 0.375 | **0.254** | **0.361** | 0.000 |
| 300 | 14/24 = 0.583 | 0.221 | 0.299 | 0.000 |
| 500 | **18/24 = 0.750** | 0.231 | 0.313 | 0.000 |

`video_recall@1` có nhích 0.833 → 0.875, nhưng đó là chỉ số duy nhất tăng.

**Trần tăng gấp đôi mà điểm thật GIẢM, và `complete_chain_rate` vẫn đúng 0.000.**
Kết luận trực tiếp: solver **chưa với tới cả 9/24 nó vốn có**, nên cho thêm ứng
viên chỉ là cho thêm nhiễu. Nút thắt không phải candidate availability.

### 2.3 `TRK-A02/A03` — DP kiểu DANTE · **ĐÃ CHẠY, DROP**

Cài ở [temporal_dp.py](../online/services/temporal_dp.py), tách biến để chạy
O(N·M). Xác minh cờ có tác dụng thật bằng cách đếm lời gọi (DP=1, beam=0).

**Ở CÙNG λ=0.002, DP và beam giống hệt nhau từng số**, kể cả theo từng video:

```
                    vid@1   mean_r   frame_sel
beam lam=0.002      0.833    0.254      0.361
DP   lam=0.002      0.833    0.254      0.361    <- khong lech mot chu so
```

**`beam_size=100` đã tìm ra tối ưu toàn cục.** Không có gì để DP giành lại.

Quét λ (DP), tune = V001+V002, holdout = V003:

| λ | vid@1 | mean_r | V001 | V002 | **V003\*** |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.792 | 0.201 | 0.338 | 0.183 | 0.081 |
| 0.0005 | **0.875** | **0.296** | 0.344 | **0.431** | 0.113 |
| 0.001 | 0.833 | 0.260 | 0.344 | 0.356 | 0.081 |
| **0.002** (hiện tại) | 0.833 | 0.254 | 0.281 | 0.306 | **0.175** |
| 0.003 | 0.792 | 0.223 | 0.281 | 0.212 | **0.175** |

λ=0.0005 đẹp nhất trên tổng nhưng holdout tụt 0.175 → 0.113. Tune và holdout mâu
thuẫn → giữ 0.002, **không promote**.

λ=0 kém hẳn ở cả hai solver, nên **phạt khoảng cách có ích thật** — giả thuyết
"phạt đang lấn át tín hiệu" ở §6 là sai.

### 2.4 Hệ quả: loại cả một họ phương pháp

> Beam đã cực đại hoá đúng hàm mục tiêu `Σ score − λ·gap`. Nên **T1 (DANTE) và
> phần lớn T3 (biến thể beam) không thể giúp** — không phải vì cài sai, mà vì
> không còn gì để tìm thêm. Muốn TRAKE khá lên phải đổi **cái được chấm**, chứ
> không phải **cách đi tìm**.

### 2.5 Thứ tự chốt lại sau khi chạy

| Thứ tự | Phase | Trạng thái |
|:-:|---|---|
| ~~A — Sequence solver~~ | | ✅ **ĐÃ CHẠY — không promote.** Solver đã tối ưu |
| 1 | **D — Fusion** | Điểm mỗi scene là RRF, vứt độ chắc chắn của nhánh. Solver tối ưu trên hàm mục tiêu nhiễu thì vẫn ra chuỗi nhiễu |
| 2 | **B — Frame refinement** | Trần ≤20% (§1.5), nhưng giờ là hướng còn lại rẻ nhất |
| 3 | **C — CHẤT LƯỢNG candidate** | Không phải số lượng — `TRK-C07` đã bác. Viết lại truy vấn / định tuyến modality |

Ba lần xếp thứ tự, ba lần bị thí nghiệm sửa. Ghi lại đầy đủ ở §2.1–2.3 để không
ai dựng lại lập luận đã bị bác.

---

## 3. Quy tắc chung cho mọi thí nghiệm

1. **Mỗi thí nghiệm đổi ĐÚNG MỘT khâu.** Đổi hai thứ cùng lúc là mất khả năng
   quy công — đã vấp hôm 08/08: bù dữ liệu + sửa `bm25_ocr` cùng lúc, phải chạy
   thêm 4 ablation mới tách được nguyên nhân.

2. **Luôn log CẢ HAI oracle** (§1.1). Bảng chỉ có một cột "oracle" là bảng dẫn
   đến kết luận sai.

3. **Chấm solver bằng `mean_r_score`, không bằng `complete_chain_rate`** cho tới
   khi Phase C xong.

4. **`PYTHONHASHSEED=0`** ở mọi lần chạy. Không cố định thì chênh ±1 truy vấn là
   nhiễu chứ không phải tín hiệu.

5. **Giữ holdout.** Tune trên V001+V002, xác nhận trên V003. Với 8 truy vấn
   TRAKE/video, chênh 1 truy vấn = 0.125 — đừng kết luận từ delta nhỏ hơn thế.

6. **Ghi cả kết quả ÂM** vào [docs/20](20_EXPERIMENT_LOG.md). Bốn thí nghiệm đã bị
   lặp lại vì kết quả âm không được ghi.

### Lệnh nền

```powershell
$env:PYTHONHASHSEED="0"
$env:AIC_ENV_FILE=".env.fpt.local"
$env:AIC_METADATA_JSONL="storage/exports_multivideo/scenes.jsonl"
$env:AIC_DATA_ROOT="storage"

python -m scripts.eval_tasks `
  --gold examples/gold_all3.jsonl `
  --metadata storage/exports_multivideo/scenes.jsonl `
  --pipeline container --disable-branch bm25_ocr --max-per-video 40 `
  --tasks TRAKE --json-out outputs/evaluation/trake/<EXP_ID>.json

# tran retrieval — chay lai moi khi doi Phase C
python -m scripts.diagnose_trake_stage_b `
  --gold examples/gold_all3.jsonl `
  --metadata storage/exports_multivideo/scenes.jsonl `
  --out outputs/evaluation/trake/<EXP_ID>_stageb.json
```

### Bảng log bắt buộc

| Exp | Đổi gì | oracle<br>corpus | oracle<br>**pool** | full-chain<br>khả thi | mean_r | frame_sel | chain | ms/query | KEEP? |
|---|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| BASE | — | .845 | **.773** | **9/24** | .254 | .361 | .000 | — | — |

Ba cột `oracle pool`, `full-chain khả thi` và `mean_r` là bộ ba quyết định. Đọc:

```
pool khong doi + mean_r tang   -> solver/refiner tot len       KEEP
pool tang + mean_r khong doi   -> retrieval tot hon, solver hong
pool giam + mean_r tang        -> NGHI NGO metric, kiem tra lai
```

---

## 4. Phase D — Fusion (chạy trước)

**Câu hỏi:** RRF có đang vứt bỏ độ chắc chắn của nhánh không?

Sửa ở [online/services/fusion.py](../online/services/fusion.py). Cảnh báo trong
docstring hiện có: `weighted_sum`/`max_score` đang dùng lại contribution **suy từ
rank**, không phải điểm đã chuẩn hoá — nên chúng KHÔNG phải hai biến thể cần đo.
Phải chuẩn hoá điểm thật trước.

| ID | Biến thể | Ghi chú |
|---|---|---|
| `TRK-D00` | RRF (hiện tại) | BASE |
| `TRK-D01` | z-score mỗi nhánh rồi cộng có trọng số | Chuẩn hoá dùng `ScoreNormalizers.from_pool()` đã có, tính TRƯỚC dedup (EVAL-01) |
| `TRK-D02` | min-max mỗi nhánh rồi cộng | |
| `TRK-D03` | max qua các nhánh (điểm đã chuẩn hoá) | |
| `TRK-D04` | trọng số theo margin: `w_b ∝ (top1_b − top2_b)` | Nhánh tách bạch rõ được nói to hơn |
| `TRK-D05` | inverse-entropy trên phân bố điểm của nhánh | Biến thể Q2E báo tốt nhất |

**Chạy trên cả 4 task**, không chỉ TRAKE — fusion là tầng dùng chung. Chỉ số
quyết định: KIS MRR + QA joint + TRAKE mean_r cùng lúc.

**Tiêu chí KEEP:** không task nào tụt quá 0.02, và ít nhất một task tăng ≥ 0.05.

**Kiểm tra phụ bắt buộc:** với biến thể thắng, chạy lại truy vấn
`"Ông NGUYỄN TRANG SƯ THÀNH PHỐ HỒNG NGỰ TỈNH ĐỒNG THÁP"` **có bật `bm25_ocr`**.
Nếu fusion đã đúng thì scene gold phải vào top-5 mà **không cần** nâng trọng số
thủ công. Đây là bài kiểm tra trực tiếp nhất cho §1.6.

---

## 5. Phase C — Candidate recall (nút thắt cứng)

**Câu hỏi:** đưa được bao nhiêu trong 31/110 bước còn thiếu vào pool?

Chỉ số chính là **`gold_region_recall@100`** và **số truy vấn có đủ mọi bước**.
`mean_r_score` ở phase này là chỉ số phụ.

| ID | Biến thể | Nguồn |
|---|---|---|
| `TRK-C00` | event text nguyên bản | BASE |
| `TRK-C01` | dịch VI→EN cho từng event trước khi vào CLIP | đã có `FptQueryTranslator` |
| `TRK-C02` | LLM viết lại theo hướng thị giác (1 biến thể) | Q2E |
| `TRK-C03` | 3 biến thể viết lại, hợp nhất candidate | Q2E |
| `TRK-C04` | `C00 ∪ C03` | |
| `TRK-C05` | định tuyến modality theo luật cho từng event | mở rộng §1.7 |
| `TRK-C06` | LLM định tuyến modality | T6-D |
| `TRK-C07` | nâng `candidate_limit` 100 → 300 cho riêng TRAKE | rẻ nhất, thử trước |

~~**Chạy `TRK-C07` đầu tiên**~~ — đã chạy, DROP (§2.2). Giữ đoạn dưới làm ghi
chép: ý tưởng là nếu pool 300 đưa
`gold_region_recall` lên đáng kể thì phần lớn "thiếu candidate" chỉ là cắt pool
quá sớm chứ không phải retrieval không tìm ra.

> **ĐÃ CHẠY 09/08 — DROP.** Trần lên 18/24 nhưng `mean_r` xuống 0.231 và
> `complete_chain_rate` vẫn 0.000. Xem §2.2. Đừng chạy lại.

**Bẫy phải canh:** query drift. Mọi biến thể viết lại phải kèm
`query_drift_rate` — tỉ lệ bản viết lại thêm thực thể/số/vật thể **không có**
trong event gốc. Có sẵn drift guard ở [docs/15](15_RESEARCH_AGENDA.md); tái dùng.

**Tiêu chí KEEP:** `gold_region_recall@100` tăng ≥ 0.05 **và** `query_drift_rate`
≤ 0.05 **và** KIS/QA không tụt.

---

## 6. Phase A — Sequence solver

**Chấm bằng `mean_r_score`.** Trần 0.773 (§1.3).

Sửa ở [online/services/temporal.py](../online/services/temporal.py). Hàm hiện tại
`link_event_hits` là beam (`beam_size=100`) + **hard** increasing `scene_idx` +
`gap_penalty=0.002` tuyến tính trên giây.

| ID | Solver | Ràng buộc thời gian | Ghi chú |
|---|---|---|---|
| `TRK-A00` | beam hiện tại | cứng tăng dần + phạt tuyến tính | BASE |
| `TRK-A01` | greedy theo thứ tự | cứng | Đối chứng dưới — beam có hơn greedy không? |
| `TRK-A02` | DP (DANTE) | không phạt, chỉ ràng thứ tự | λ = 0 |
| `TRK-A03` | DP (DANTE) | phạt tuyến tính | λ ∈ {1e-4, 5e-4, 1e-3, 3e-3, 1e-2} |
| `TRK-A04` | beam | phạt giảm mũ `e^(−α·Δt)` | α ∈ {0.005, 0.01, 0.05}; beam ∈ {2,4,8,16} |
| ~~`TRK-A05`~~ | ~~MADTempo hai đầu~~ | — | **DROP**, điều kiện tiên quyết không thoả — xem dưới |
| `TRK-A05b` | MADTempo một đầu | neo E₁ → beam tiến về sau | thay thế; chạy sau `A04` |
| `TRK-A06` | Drop-DTW | cho phép bỏ frame nhiễu, **không** bỏ event | ưu tiên thấp — xem dưới |

### Chuẩn hoá khoảng cách thời gian trước khi so λ

`λ` của paper không mang sang được nếu đơn vị khác. Chuẩn hoá về **giây** và ghi
rõ trong log. Corpus này: scene p50 = 4.0s, khoảng cách giữa hai bước gold p50
cần đo trước khi chọn dải λ — **đo trước, đừng chép từ paper**.

### `TRK-A05` MADTempo — điều kiện tiên quyết ĐÃ ĐO, và nó KHÔNG thoả

Neo hai đầu chỉ có nghĩa nếu E₁ và Eₙ dễ tìm hơn các bước giữa. Đo trên chính dữ
liệu Stage B (09/08):

| Vị trí bước | n | có trong pool | rank p50 | trong top-5 |
|---|---:|---:|---:|---:|
| **đầu** | 24 | 75.0% | 5.0 | **45.8%** |
| giữa | 62 | 71.0% | 9.0 | 24.2% |
| **cuối** | 24 | 70.8% | 8.0 | **20.8%** |

**Bước đầu đúng là mạnh hơn** (top-5 45.8% so với 24.2%). Nhưng **bước cuối thì
không** — nó là yếu nhất trên top-5 (20.8%), kém cả bước giữa. Giả định "hai đầu
đều là neo rõ" sai với corpus này.

Và MADTempo cần **cả hai** neo cùng lúc:

```
truy van co CA HAI dau trong pool: 13/24 = 54.2%   <- tran cua TRK-A05
```

**Quyết định: bỏ `TRK-A05` bản hai đầu.** Thay bằng biến thể một đầu, khớp với
dữ liệu:

| ID | Biến thể | |
|---|---|---|
| ~~`TRK-A05`~~ | ~~neo hai đầu → đoạn ứng viên → beam ở giữa~~ | **DROP** — bước cuối không phải neo |
| `TRK-A05b` | neo **chỉ E₁** (mạnh nhất), rồi beam tiến về sau trong cửa sổ thời gian | Giữ được lợi thế duy nhất đo được |

Ghi chú cho `TRK-A05b`: nó gần như là `TRK-A04` (beam + phạt mũ) cộng thêm ràng
buộc "bắt đầu từ top-K của E₁". Nếu `TRK-A04` đã thắng thì delta của `A05b`
nhiều khả năng nhỏ — chạy sau `A04` và chỉ chạy nếu `A04` cho thấy bước đầu hay
bị chọn sai.

### `TRK-A06` Drop-DTW — vì sao xếp cuối

Keyframe pool hiện rất thưa (91% scene có đúng 1 keyframe). DP có ràng buộc thứ
tự đã ngầm "bỏ qua" gần hết frame trung gian rồi, nên Drop-DTW nhiều khả năng
trùng lặp với `TRK-A03`. Chỉ đáng làm khi đã có timeline dày (xem
[docs/30](30_SYSTEM_DIAGNOSIS.md) §2.3 về OCR dày).

**Tiêu chí KEEP:** `mean_r_score` tăng ≥ 0.05 trên tune **và không giảm** trên
holdout V003.

---

## 7. Phase B — Frame refinement

Chỉ chạy sau khi Phase A chốt. Trần 20% với cửa sổ cục bộ (§1.5), nên **đặt kỳ
vọng đúng ngay từ đầu**.

| ID | Biến thể | Ghi chú |
|---|---|---|
| `TRK-B00` | `scene.best_frame_idx` | BASE |
| `TRK-B01` | + trung bình điểm lân cận | `S'(f) = S(f) + β·mean(S(f±1..r))` |
| `TRK-B02` | + max điểm lân cận | |
| `TRK-B03` | + độ ổn định thời gian (ABTS) | `c = λs·s + λt·t`, `t` = nghịch đảo phương sai lân cận |
| `TRK-B04` | nhiều cửa sổ + độ ổn định | |

Quét `r ∈ {4, 8, 16, 32}`, `β ∈ {0.1, 0.25, 0.5, 1.0}`,
`λs:λt ∈ {1:0, 0.9:0.1, 0.75:0.25, 0.5:0.5}`.

✅ **Đã hết chặn (10/08).** `TRK-B03`/`B04` cần frame thô quanh mốc, và trước đây
`L21_V002.mp4`/`L21_V003.mp4` không có trên đĩa nên chỉ chạy đủ trên V001 (8 truy
vấn — quá ít để kết luận). Hai file đã được tìm thấy ở `D:\` và chép vào
`storage/raw/videos/`; xác minh khớp dataset trước khi chép (5/6 keyframe mẫu của
V002 khớp ở offset 0 với sai lệch mức nhiễu JPEG, V003 khớp 3/3). Phase B giờ
chạy được trên cả 24 truy vấn.

---

## 8. Không đưa vào kế hoạch này

| | Vì sao |
|---|---|
| **T9 Vortex** | Chỉ kiểm "cùng video", mà `video_recall@3` đã 1.000 — không còn gì để cải thiện ở tầng đó |
| **T11 EA-VTR** | Huấn luyện lại representation. Chỉ đáng khi oracle **pool** vẫn thấp sau Phase C, và cần corpus lớn hơn 3 video |
| **Nới ràng buộc thứ tự** | 24/24 gold đều tăng dần → gold **không đo được** lợi ích, mà nới sẽ chiếm chỗ trong top-20. Xem [docs/30](30_SYSTEM_DIAGNOSIS.md) §4.3; khuyến nghị vẫn là cảnh báo trên UI |

---

## 9. Điều kiện tiên quyết trước khi tin bất kỳ số nào

- [ ] **Thêm video distractor.** Mọi số ở đây đo trên **3 video**, `video_recall@3`
      đã đạt 1.000 — trần đó gần như chắc chắn do corpus quá nhỏ. Solver tối ưu
      trên 3 video có thể vô nghĩa ở 876. Đây là ưu tiên #1 trong
      [docs/30](30_SYSTEM_DIAGNOSIS.md) §8 và nó đứng **trên** toàn bộ kế hoạch này.
- [x] ~~Phục hồi `L21_V002.mp4`, `L21_V003.mp4`~~ — xong 10/08, Phase B hết chặn.
- [ ] `PYTHONHASHSEED=0` trong mọi lần chạy.

---

## 10. Tài liệu liên quan

- [docs/30](30_SYSTEM_DIAGNOSIS.md) — chẩn đoán toàn hệ thống, §4.3 phần TRAKE
- [docs/22](22_TRAKE_CHAIN_SCORING.md) — cách chấm điểm chuỗi hiện tại
- [docs/20](20_EXPERIMENT_LOG.md) — nhật ký thí nghiệm, **ghi cả kết quả âm vào đây**
- [from_sequences.py](../online/services/trake/from_sequences.py) — vì sao đường cũ thắng `TrakeProcessor`
