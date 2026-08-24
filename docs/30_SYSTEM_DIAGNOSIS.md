# 30 — Chẩn đoán toàn hệ thống

Cập nhật **2026-08-09**. Kế thừa [docs/27](27_SYSTEM_ISSUES.md) (06/08), sau đợt bù
dữ liệu cho cả ba video, ba lỗi tìm ra trong quá trình đó, và Phase A/D của
[docs/31](31_TRAKE_EXPERIMENT_PLAN.md).

Mọi số trong tài liệu này **đo được**, không suy diễn. Nguồn:
`outputs/evaluation/quick/D_FINAL_tiebreak.json` — 120 truy vấn gold,
`--pipeline container`, VLM rerank tắt, `PYTHONHASHSEED=0`.
Dữ liệu: 3 video / 765 scene / 855 keyframe. 584 test pass.

Ký hiệu:

| | |
|---|---|
| ✅ | có, chạy đúng, **đo được là có ích** |
| ⚠️ | có, chạy được, nhưng **chưa chứng minh được ích lợi** hoặc có khiếm khuyết đã biết |
| ❌ | **không có**, hoặc có code nhưng vô hiệu trên dữ liệu hiện tại |
| 🚫 | bị chặn bởi thứ ngoài tầm code (thiếu file đầu vào) |

---

## 1. Điểm số hiện tại

**Cập nhật 09/08** sau Phase A/D (docs/31) và QA TOP_N. Cột "08/08" là bảng cũ
của chính tài liệu này.

| Task | Chỉ số | 08/08 | **09/08** | |
|---|---|---:|---:|---|
| **KIS** | R@1 | 0.583 | **0.750** | ++ |
| | R@5 | 0.944 | **1.000** | ++ |
| | **R@20** | 1.000 | **1.000** | = |
| | MRR | 0.731 | **0.852** | ++ |
| **QA** | evidence R@1 | 0.583 | **0.611** | ++ |
| | **evidence R@20** | 0.889 | **0.889** | = |
| | answer_accuracy | 0.611 | **0.667** | ++ |
| | joint_top1 | 0.417 | **0.472** | ++ |
| **TRAKE** | video_recall@1 | 0.833 | **1.000** | ++ |
| | video_recall@3 | 1.000 | 1.000 | = |
| | mean R-score | 0.254 | **0.354** | ++ |
| | complete_chain_rate | 0.000 | **0.042** | ++ *(1/24, lần đầu khác 0)* |
| **AVS** | nDCG@100 | 0.558 | **0.565** | ++ |
| | event_coverage | 0.793 | **0.884** | ++ |

Ba thay đổi tạo ra bảng này, mỗi cái đều qua holdout V003:

1. **`AIC_FUSION_METHOD=norm_max`** (QA giữ `rrf`) — chuẩn hoá điểm trong phạm vi
   từng nhánh thay vì suy từ rank. Đây là phần lớn mức tăng.
2. **`AIC_FPT_QA_TOP_N` 5 → 10** — LLM đọc 10 evidence pack thay vì 5.
3. Nền dữ liệu 08/08 (color/quality/events đủ cả 3 video).

`complete_chain_rate` lần đầu khác 0 (**1/24**), nhưng trần của nó là **9/24**
(docs/31 §1.2) — chặn bởi retrieval chứ không bởi solver. Đừng dùng chỉ số này để
chấm thuật toán ghép chuỗi; xem §4.3 để biết đọc nó thế nào.

**Theo mục tiêu R@20 của bạn:** KIS đạt trần **1.000**, QA **0.889**. Đây là hai
con số quan trọng nhất và cả hai đều ở mức dùng được cho rà thủ công top-20.

### Theo từng video — chỗ lệch đáng chú ý

| | V001 | V002 | V003 |
|---|---:|---:|---:|
| KIS R@1 | **0.833** | 0.667 | 0.750 |
| QA joint_top1 | 0.333 | 0.500 | **0.583** |
| TRAKE video@1 | **1.000** | **1.000** | **1.000** |
| TRAKE mean R | 0.325 | 0.356 | **0.381** |
| AVS nDCG | 0.498 | **0.681** | 0.517 |

`TRAKE video@1` giờ **1.000 trên cả ba video** — khâu khoá video coi như xong.
KIS lệch 0.167 giữa video tốt nhất và kém nhất (2 truy vấn), AVS lệch 0.183.
Với 8–12 truy vấn/video/task thì chưa đủ để kết luận về từng video.

Đáng chú ý: **V003 (holdout) giờ là video TỐT NHẤT** ở QA và TRAKE, và không phải
kém nhất ở KIS. Không có dấu hiệu overfit về phía hai video tune.

---

## 2. Tầng dữ liệu

### 2.1 Có gì

- [x] ✅ **Scene detection** — 765 scene, 3 video, TransNetV2
- [x] ✅ **Keyframe** — 855 ảnh, mọi scene có ≥1
- [x] ✅ **Caption** (Qwen2.5-VL) — 855/855
- [x] ✅ **Embedding** CLIP ViT-L/14 — 855/855, vector trên đĩa
- [x] ✅ **ASR** (faster-whisper large-v3) — 749/765 scene
- [x] ✅ **Object** — 851/855
- [x] ⚠️ **OCR** — 736/855 (86%), xem §2.3
- [x] ✅ **Event** — 476 event, **765/765 scene có `event_id`** *(mới 08/08; trước là 217/765, chỉ V001)*
- [x] ⚠️ **Color** — 855/855 *(mới 08/08)*, nhưng vô dụng, xem §3.4
- [x] ⚠️ **Quality** — 855/855 *(mới 08/08)*, nhưng thang đo lệch, xem §3.5
- [ ] ❌ **`selection_score`** — 0/855, **cố ý** (xem [docs/29](29_DATA_CONTRACT.md) §6)
- [ ] ❌ **`duplicate_score`** — không đo được từ một ảnh đơn lẻ
- [x] ✅ **Video gốc cả ba** *(10/08)* — `L21_V002.mp4`/`L21_V003.mp4` tìm thấy ở `D:\`
      và đã chép vào `storage/raw/videos/`. Xác minh khớp dataset trước khi chép:
      5/6 keyframe mẫu của V002 khớp ở offset 0 (sai lệch 0.76–0.92/255 = nhiễu
      JPEG), V003 khớp 3/3. Range trả 206 cho cả ba.
- [ ] ❌ **`clips.jsonl` / ClipSegment** — file rỗng; online không đọc ([docs/29](29_DATA_CONTRACT.md) §4.5)

### 2.2 Tính toàn vẹn

- [x] ✅ `verify_export()` **PASS** *(mới 08/08; trước fail vì `checksum mismatch: videos.jsonl`)*
- [x] ✅ `dataset_manifest.json` khớp file thật, `/v1/health` báo đúng `dataset_version`
- [x] ✅ Contract pydantic validate mọi scene lúc nạp

### 2.3 Chất lượng OCR — điểm yếu nhất của tầng dữ liệu

| Vấn đề | Số đo |
|---|---|
| Độ phủ | 736/855 keyframe (86%) |
| Chữ **lớp phủ** (logo, đồng hồ, chữ chạy) | ~84% tổng số chuỗi |
| Sau khi lọc lớp phủ | 239/765 scene còn chữ, 2 429 từ |
| Chyron (tên/chức vụ người nói) bị bỏ lỡ | **42%** cửa sổ chyron ở V001 |

Chyron chỉ hiện 2–3 giây mà mỗi scene chỉ lấy 1 keyframe. Ví dụ đo được: chyron
`Ông DƯƠNG PHÚ XUÂN` ở frame 10360–10430, keyframe ở 10355 — **hụt 5 frame**.
Đã vá cho V001 (49 keyframe, `scripts/chyron_backfill.py`). V002/V003 trước bị
chặn vì thiếu mp4; **từ 10/08 đã có video nên chạy được** — chưa chạy.

**Bộ dò chyron không tổng quát** — nó dựa vào dải đỏ đặc trưng của HTV9 "60 giây".
Ba cách tổng quát hơn đều đã thử và hỏng: dò thay đổi pixel (98% frame khác nhau vì
ticker chạy), dò hàng động (24/25 hàng động vì video bên dưới cũng động), dò độ
tĩnh (chyron 35–39% vs nền tĩnh 71%, không tách được).

Đường đúng cho corpus thật là **OCR dày bằng model local** (1–2 fps, PaddleOCR trên
GPU, ~1–2 phút mỗi 20 phút video, 0 đồng API) rồi gộp trùng ở **mức chữ**, kèm một
trường `ocr_timeline` mức scene thay vì gắn chữ vào frame không hiển thị nó. Chưa
làm.

---

## 3. Tầng tìm kiếm — 7 nhánh

| Nhánh | Trạng thái | Đánh giá |
|---|---|---|
| `dense_visual` (CLIP + dịch VI→EN) | ✅ | Nhánh mạnh nhất. Cứu được cả caption bịa — xem §3.1 |
| `bm25_caption` | ✅ | 100% độ phủ |
| `bm25_asr` | ✅ | 98%. Neo duy nhất cho địa danh chỉ được *nói* |
| `bm25_keyword` | ✅ | 99%, sinh từ `objects[].label` |
| `bm25_object` | ✅ | 99% |
| `bm25_action` | ⚠️ | 57% độ phủ, suy từ caption chứ không phải model |
| `color_search` | ⚠️ | Sống nhưng **đóng góp 0**, xem §3.4 |
| `bm25_ocr` | ❌ **TẮT** | Đo được là **gây hại**, xem §3.2 |
| `ocr_fuzzy` | ❌ TẮT | Đo 06/08: bật làm KIS R@1 0.583→0.500 |
| `event_search` | ❌ TẮT | Cùng đợt đo, cùng kết luận |

### 3.1 Fusion đa nhánh đang thực sự làm việc

Bằng chứng cụ thể: keyframe f10908 là đàn cá quẫy nhưng caption Qwen bịa thành
*"một đám cháy lớn đang bùng phát… khói bốc lên cao"*. Nó **vẫn lên hạng 1** cho
truy vấn "đàn cá đang nhảy", nhờ `dense_visual` (0.0218) áp đảo
`bm25_caption` (0.0137). CLIP nhìn thẳng vào ảnh và bù lại caption sai.

### 3.2 `bm25_ocr` — sửa xong lại phải tắt

Nhánh này **chết hoàn toàn** cho tới 07/08: `_boxes()` đọc
`keyframe.ocr_instances`, thuộc tính chỉ có trên canonical `Keyframe` chứ **không
có** trên `FrameEvidence`. `getattr(..., [])` nuốt sai lầm, `zip(texts, [])` trả
rỗng, và bộ lọc lớp phủ xoá sạch OCR của **cả 765 scene** trong im lặng.

```
AIC_OCR_OVERLAY_DF=0.10 (cấu hình đang chạy)    0/765 scene còn chữ OCR
không lọc                                     674/765 scene, 12 490 từ
```

Đã sửa (`FrameEvidence.ocr_boxes` + `zip(strict=True)`). Nhưng nhánh sống lại thật
thì đo được là **gây hại**:

| | KIS R@1 | KIS MRR | QA R@1 | QA answer_acc |
|---|---:|---:|---:|---:|
| `bm25_ocr` bật | 0.500 | 0.680 | 0.444 | 0.583 |
| `bm25_ocr` tắt | **0.583** | **0.738** | **0.583** | **0.611** |

Siết bộ lọc (`AIC_OCR_OVERLAY_MAX_WORDS=6`) chỉ lấy lại một phần: R@1 0.556,
MRR 0.709. Đã chốt `AIC_ENABLE_OCR_BRANCH=false`.

**Tắt nhánh không xoá dữ liệu OCR** — `ocr_texts` vẫn vào evidence pack mà QA đọc để
*trả lời*, và đó chính là lý do `answer_accuracy` *tăng* khi tắt: bớt nhiễu ở tầng
xếp hạng mà vẫn giữ chữ ở tầng trả lời.

### 3.3 ~~RRF pha loãng nhánh chắc chắn~~ — ĐÃ XỬ LÝ 09/08

Đo được: với truy vấn `"Ông NGUYỄN TRANG SƯ THÀNH PHỐ HỒNG NGỰ TỈNH ĐỒNG THÁP"`,
`bm25_ocr` chấm scene đúng **27.06** so với **6.5** của kẻ đứng sau — chắc chắn
tuyệt đối. Nhưng RRF chỉ tính **thứ hạng**, nên nhánh đó vẫn chỉ được `1/(60+1)`
như một nhánh đang đoán mò, và bảy nhánh kia pha loãng nó đi. Scene đúng **không
có trong top-20**.

- [x] ✅ **Fusion đọc điểm thật** — cài 09/08, `AIC_FUSION_METHOD=norm_max`
      (QA giữ `rrf`). Đây là cải thiện lớn nhất từ trước tới nay: KIS R@1
      0.583 → **0.750**, MRR 0.731 → **0.852**, TRAKE video@1 → **1.000**,
      mean_r 0.254 → **0.354**, AVS event_coverage 0.793 → **0.884**. Không một
      chỉ số nào tụt, holdout xác nhận cả bốn task.

**Nhưng cơ chế KHÔNG phải như đoạn trên mô tả.** Giả thuyết "nhánh chắc chắn bị
pha loãng" đã bị chính test bác bỏ: ba nhánh cùng xếp một candidate hạng 1 thì
`norm_max` vẫn chọn phía đồng thuận, đúng như nó nên làm. Cơ chế thật là **đập đuôi**:

```
ti le dong gop hang-1 so voi hang-100 CUA CUNG MOT NHANH
  RRF (k=60)     1/61 so voi 1/160   ->  2.62x
  min-max        1.00 so voi ~0.00   ->  vo han
```

RRF cho candidate hạng 100 tận **38%** số phiếu của hạng 1. Bảy nhánh × 100
candidate = 700 lá phiếu gần bằng nhau, tín hiệu thật chìm trong đó.

Hai biến thể *có* nhân hệ số độ-chắc-chắn (`margin_sum`, `entropy_sum`) đều **kém
hơn** bản chỉ chuẩn hoá — bằng chứng trực tiếp rằng độ chắc chắn cross-branch
không phải thứ đang thiếu. Xem [docs/20](20_EXPERIMENT_LOG.md) mục 09/08 Phase D.

### 3.4 `color_search` — sống, đo được, vô dụng

Sau khi bù color (855/855), nhánh trả về 100 candidate mỗi truy vấn thay vì rỗng.
Nhưng **không đổi một chỉ số nào**: tắt hẳn cho kết quả giống hệt bật.

Hai lý do độc lập, cả hai đều đo được:

1. **Không phân biệt được gì.** `dominant_colors` giữ top-8 màu bất kể tỉ lệ, mà
   ảnh thật nào cũng có chút của mọi sắc:

   ```
   đỏ 99.6%   xám 99.1%   trắng 93.5%   xanh 92.8%   cam 90.1%
   ```

2. **Sửa được điều đó cũng không giúp.** Thêm ngưỡng tỉ lệ (`AIC_COLOR_MIN_RATIO`,
   đã wire, mặc định 0.0) đưa xuống 1.7–2.2 màu/scene, nhưng KIS **y hệt**:

   | | R@1 | R@5 | R@20 | MRR |
   |---|---:|---:|---:|---:|
   | ratio 0.00 | 0.583 | 0.944 | 1.000 | 0.731 |
   | ratio 0.15 | 0.556 | 0.944 | 1.000 | 0.718 |
   | ratio 0.20 | 0.583 | 0.944 | 1.000 | 0.731 |

   Vì §3.3: RRF đã dìm nhánh về 0 rồi, có phân biệt tốt hơn cũng không nổi lên được.

23/120 truy vấn gold có từ chỉ màu, nên đây **không phải** trường hợp "gold không
đo được" — nhánh thực sự không đóng góp gì.

### 3.5 Thang `safe_frame` lệch corpus

`quality` giờ đủ 855/855, nhưng:

```
sharpness thực tế   p10=241   p50=488   p90=824
safe_frame chuẩn hoá trong [40, 300]  ->  742/855 (87%) VƯỢT TRẦN, kẹp về 1.0
```

Tín hiệu vừa bật đã mất phần lớn khả năng phân biệt. `SHARPNESS_CEILING` được chọn
trước khi có dữ liệu thật. Chỉnh là đổi scoring nên phải đo holdout — chưa làm.

---

## 4. Bốn task

### 4.1 KIS ✅ — tốt nhất

- [x] R@20 = **1.000** trên cả ba video
- [x] Đồng đều tuyệt đối giữa ba video (R@1 0.583 cả ba)
- [x] Safe-frame chọn frame đại diện
- [ ] ⚠️ R@1 0.583 — 15/36 truy vấn cần rà thủ công để lấy đúng frame

### 4.2 QA ⚠️ — tìm được chỗ, trả lời còn yếu

- [x] evidence R@20 = 0.889
- [x] answer_accuracy 0.611
- [ ] ⚠️ **joint_top1 chỉ 0.417** — tìm đúng chỗ *và* trả lời đúng cùng lúc chỉ 4/10 lần
- [ ] ❌ **LLM chỉ đọc 5 ứng viên đầu** (`AIC_FPT_QA_TOP_N=5`). Scene đúng ở hạng 6–20
      thì đáp án do rule-based sinh, thường là danh từ chung
- [ ] ❌ **Verifier đóng dấu `SUPPORTED` cho mọi thứ** — quan sát được: nó duyệt
      `"người"` cho câu hỏi *"tên là gì"*, cả 20/20 dòng. **Đừng tin cột
      `verifier_status` khi chấm thủ công**
- [ ] ⚠️ Nhiễu LLM ±1 truy vấn giữa các lần chạy (QA-REPRO-01)

### 4.3 TRAKE ⚠️ — khoá đúng video, chọn sai frame

- [x] video_recall@1 = 0.833, @3 = **1.000** — luôn tìm ra video đúng trong 3 lựa chọn đầu
- [x] `gold_video_missing` = 0.000
- [x] ⚠️ **`complete_chain_rate` = 0.042** — 1/24 truy vấn dựng đúng TRỌN chuỗi.
      Lần đầu khác 0 (trước 09/08 là 0.000), nhưng trần là 9/24 = 0.375.
      **Đây là chỉ số ALL-OR-NOTHING**: một chuỗi 5 bước đúng 4 vẫn tính 0.
      Phân bố `r_score` nói nhiều hơn: 20/24 đúng ít nhất một bước,
      8/24 đúng từ nửa chuỗi trở lên, 4/24 sai sạch
- [ ] ⚠️ `frame_selection_accuracy` 0.361 so với oracle 0.845 — frame đúng **có** trong
      pool, khâu chọn làm hỏng
- [ ] ❌ **Ràng buộc thứ tự cứng.** [temporal.py:27](../online/services/temporal.py#L27) bắt
      `scene_idx` tăng nghiêm ngặt. Truy vấn liệt kê sự kiện sai thứ tự thì bộ ba đúng
      **không thể ghép được** — recall về 0 chứ không giảm dần. Đo được: cùng một truy
      vấn, chỉ đảo thứ tự hai bước, từ "không có trong top-20" thành **hạng 1**.
      24/24 gold đều đúng thứ tự nên harness không bắt được lỗi này

Đây là task còn nhiều dư địa nhất: video đã đúng, chỉ khâu chọn frame trong chuỗi.

### 4.4 AVS ⚠️ — chỉ dùng nội bộ

- [x] nDCG@100 0.558, event_coverage 0.793, `zero_result_rate` 0.000
- [x] Cổng grade gần như không còn hại (loại nhầm gold 0.083)
- [ ] ⚠️ Lùi 0.04 so với 06/08 — **giá thật** của việc bật dedup ở 2/3 corpus:
      bớt ảnh gần trùng nên ít lần "trúng" gold hơn, đổi lấy kết quả đa dạng hơn
- [ ] ⚠️ P@100 chỉ 0.050 — 95% kết quả trả về là rác
- **AVS không có định dạng nộp chính thức**, chỉ để đánh giá nội bộ

---

## 5. Tầng nộp bài và giao diện

- [x] ✅ **Server mở cổng NGAY**, nạp ~4 phút ở luồng nền *(22/08)*. Trước đó cả
      quãng nạp nằm trong `lifespan` nên trình duyệt chỉ báo "không kết nối được",
      không phân biệt được "đang nạp" với "đã chết". Tiến độ: `GET /v1/startup`
      (không cần token); endpoint thường CHỜ nạp xong chứ không 503
- [x] ✅ **Nhánh retrieval không còn khoá event loop** *(22/08)*. `bm25`,
      `ocr_fuzzy`, `color_search`, `event_search`, `caption_dense` và
      `InMemoryVectorStore` đều quét toàn corpus; chạy thẳng trên loop nghĩa là
      suốt quãng đó server đứng hình với MỌI người — đúng triệu chứng "load lâu"
      khi cả đội dùng chung một server. Khoá bằng `tests/test_event_loop_not_blocked.py`
- [x] ✅ **Solo từng nhánh** trong panel Trọng số — chạy riêng ASR / caption /
      OCR / hình ảnh để xem nhánh đó tự đứng thì trả về gì
- [x] ✅ **Khoanh vùng + đào sâu vào vài video** (`filters.video_ids` + nới trần
      mỗi video + tăng top-K) để làm giàu đáp án trong video đã tin là đúng
- [x] ✅ **Bản nháp sắp xếp dùng chung cả đội** — `/v1/submission-drafts`, ghi
      JSONL trên đĩa nên sống qua restart
- [x] ✅ **Đáp án QA hiện ngay trên lưới ảnh**, đồng bộ từ bảng nộp

- [x] ✅ Xuất submission KIS / QA / TRAKE đúng contract
- [x] ✅ `submission_validator` — kiểm frame thuộc video bằng `frame_count` thật
- [x] ✅ `evaluate-local` — chấm thử trước khi nộp
- [x] ✅ Sắp thứ tự thủ công trong bảng nộp
- [x] ✅ Gõ thẳng số hạng vào từng dòng — dòng được **chèn** vào vị trí đó, phần còn
      lại dịch xuống (mũi tên lên/xuống chỉ đi một bậc, không dùng để nhảy xa được)
- [x] ✅ Tick nhiều dòng (Shift để chọn cả dải) rồi sửa **một câu trả lời QA cho cả
      loạt**, đưa cả khối tới một hạng, hoặc bỏ cả khối khỏi bài nộp
- [x] ✅ Phát đoạn video theo kết quả (HTTP Range, nới ±5s lấy bối cảnh)
- [x] ✅ Bảng trọng số nhánh theo modality
- [x] ✅ Session replay — mọi tìm kiếm được ghi trace
- [ ] ⚠️ **Thanh trượt trọng số kẹp ở 1.0** ([WeightPanel.tsx:241](../online/ui-react/src/features/weights/WeightPanel.tsx#L241)
      gọi `Math.min(..., 1)`) trong khi backend nhận tới 10.0. Muốn một nhánh áp đảo
      phải hạ các nhánh khác thay vì nâng nhánh đó
- [x] ✅ **Phát được video cả ba** *(10/08)* — trước V002/V003 thiếu mp4 và
      [playback.py:69](../online/services/playback.py#L69) trả `None`. Nay đủ file,
      `/v1/media` trả 206 với header Range cho cả ba
- [x] ✅ **Tab "Chỉnh frame"** *(10/08)* — soát lại submission đã lọc: thanh kéo
      toàn video, nút nhích ±1/5/30/300 frame, phím ← →, tua ngược từ trình phát,
      xuất lại CSV. Quy đổi frame↔giây dùng `fps` thật từ `GET /v1/videos`
      (**V003 chạy 25 fps**, không phải 30)
- [ ] ❌ **AVS không có đường nộp** — đúng thiết kế

---

## 6. Hạ tầng và vận hành

- [x] ✅ 584 test pass
- [x] ✅ `AIC_ENV_FILE` bắt buộc tường minh — không tự dò `.env`
- [x] ✅ Không log `Authorization`; `.env.fpt.local` gitignored
- [x] ✅ Cache VLM theo `sha256(ảnh + prompt + model)`
- [x] ✅ Cache bản dịch VI→EN trên đĩa
- [x] ✅ `/v1/search/capabilities` từ chối option chưa cài đặt bằng 422 thay vì lờ đi
- [ ] ⚠️ **`capabilities` liệt kê nhánh ĐÃ ĐĂNG KÝ, không phải nhánh CÓ DỮ LIỆU.**
      Muốn biết nhánh nào thật sự trả kết quả phải đọc `branch_status` của một truy
      vấn thật
- [ ] ⚠️ **Backend nạp module lúc khởi động** — mọi thay đổi trong `online/` cần khởi
      động lại. Đã vấp 3 lần trong một phiên
- [ ] ❌ **Chưa đo ở quy mô thật.** 765 scene; 876 video sẽ là ~250k scene. Bốn chỗ
      sẽ đau trước: `videos.jsonl` lồng cả `scenes[]` (~550 MB đọc rồi vứt), toàn bộ
      export nạp vào RAM, 855 lần `open()` file vector thành ~250k, `_media_exists`
      LRU cache 512 nhỏ hơn số video. Quét tuyến tính 250k vector chỉ mất **6.4 ms**
      nên **chưa cần ANN** — nút cổ chai là I/O khởi động và RAM

---

## 7. Chi phí API

`AIC_RERANK_VLM_ENABLED=false`. Đo 06/08 trên 1 truy vấn:

| | VLM bật | VLM tắt |
|---|---|---|
| KIS | 17 gọi, 8.8s | **1 gọi, 0.6s** |
| QA | 29 gọi, 17.1s | **6 gọi, 4.8s** |
| AVS | 11 gọi, 5.6s | **1 gọi, 0.5s** |

VLM rerank tiêu 94% số lệnh gọi API của KIS mà **không đổi một chỉ số nào** trên cả
4 task. Đây nhiều khả năng là phần lớn hoá đơn 1500k đã tiêu.

⚠️ Đặt `=true` mà để `AIC_RERANK_VLM_URL` rỗng **vẫn bật** nhánh này — container ưu
tiên đường FPT. URL rỗng không phải cách tắt.

---

## 8. Việc nên làm tiếp, xếp theo giá trị trên mục tiêu R@20

| # | Việc | Vì sao | Đo được không |
|---|---|---|---|
| 1 | **A5 — thêm video distractor** | Mọi số hiện tại đo trên **3 video**, và KIS R@20 đã chạm trần 1.000 — trần đó gần như chắc chắn là do corpus quá nhỏ chứ không phải do hệ thống hoàn hảo. Chưa có distractor thì không biết hệ thống thật sự đứng ở đâu | có |
| 2 | **TRAKE chọn frame trong chuỗi** | `complete_chain_rate` mới 0.042/0.375; `frame_selection_accuracy` 0.419 so với oracle 0.845 | có |
| 3 | **Fusion nhạy độ chắc chắn** | Mở khoá cả `bm25_ocr` lẫn `color_search` cùng lúc (§3.3) | có |
| 4 | **QA_TOP_N 5 → 15** + sửa verifier | joint_top1 0.417, và verifier duyệt cả `"người"` | có |
| 5 | **OCR dày bằng model local** | 86% → gần 100%, bắt được chyron, 0 đồng API | một phần — 0/40 gold dùng tên người |
| 6 | Nới ràng buộc thứ tự TRAKE | Recall về 0 khi người dùng nhớ nhầm thứ tự | **không** — 24/24 gold đều đúng thứ tự |
| 7 | Hiệu chuẩn `SHARPNESS_CEILING` | 87% keyframe bị kẹp trần | có |

Mục 6 là chỗ duy nhất tôi khuyên **giữ nguyên + cảnh báo trên UI** thay vì sửa: nới
ràng buộc sẽ chiếm chỗ trong top-20 mà gold không đo được lợi ích.

---

## 9. Ba lỗi tìm ra trong đợt này

| Lỗi | Ảnh hưởng | Trạng thái |
|---|---|---|
| `_boxes` đọc thuộc tính không tồn tại | `bm25_ocr` chết hẳn, **0/765 scene** có OCR trong index, im lặng | ✅ đã sửa + `zip(strict=True)` chặn tái diễn |
| `events.jsonl` chỉ phủ V001 | dedup theo event chỉ chạy 1/3 corpus → metric giữa các video **không so sánh được** | ✅ đã sửa, 765/765 |
| `dataset_manifest` lệch file thật | `verify_export()` fail, `/v1/health` báo `dataset_version` cũ | ✅ đã sửa, PASS |

Cả ba đều **hỏng trong im lặng** — không log, không cảnh báo, test vẫn xanh. Đó là
điểm chung đáng lo nhất: hệ thống hiện chưa có cơ chế nào phát hiện một nhánh chết.
`branch_status` báo `empty` chứ không báo `broken`, và `empty` là trạng thái hợp lệ.

---

## 10. Tài liệu liên quan

- [docs/29](29_DATA_CONTRACT.md) — hợp đồng dữ liệu offline → online, trường nào thực sự được đọc
- [docs/28](28_HUMAN_EVAL_RUNBOOK.md) — chạy server thủ công
- [docs/27](27_SYSTEM_ISSUES.md) — tổng hợp vấn đề 06/08. **Không bị thay thế**: nó
  còn giữ phần audit tầng dữ liệu và các thí nghiệm chi tiết mà tài liệu này chỉ
  tóm tắt kết luận. Chỗ nào hai bản lệch nhau, bản này (08/08) mới hơn.
- [docs/20](20_EXPERIMENT_LOG.md) — nhật ký thí nghiệm, gồm cả kết quả âm
