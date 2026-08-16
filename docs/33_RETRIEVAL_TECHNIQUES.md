# 33 — Kỹ thuật nền tảng: so khớp từ hay ngữ nghĩa?

Lập **2026-08-13**. Trả lời một câu hỏi duy nhất: *mỗi thứ đang chạy trong hệ
thống dựa trên kỹ thuật nền tảng nào?*

Mọi khẳng định ở đây đọc thẳng từ code ngày 13/08, không từ tên nhánh và không
từ thiết kế mong muốn. Chỗ nào tài liệu cũ mô tả một cơ chế mà container **không
bật**, tài liệu này nói rõ là không bật.

---

## 0. Trả lời ngắn

Hệ hiện tại **chủ yếu là so khớp từ**. Trong 8 nhánh retrieval đang đăng ký, có
**đúng một** nhánh hiểu ngữ nghĩa thật — `dense_visual` (CLIP). Bảy nhánh còn
lại đều là biến thể của so khớp chuỗi ký tự / token.

Ngữ nghĩa xuất hiện thêm ở **ba chỗ nằm NGOÀI retrieval**: dịch truy vấn VI→EN
trước khi vào CLIP, rerank bằng cross-encoder, và LLM đọc bằng chứng để trả lời
QA.

```
   TRUY VẤN
      │
      ├─ [NGỮ NGHĨA]  dịch VI→EN bằng LLM ─→ CLIP text tower ─→ cosine ─→ dense_visual
      │
      └─ [SO KHỚP TỪ] Okapi BM25 ×5 · fuzzy ký tự ×1 · từ điển màu ×1 · BM25 event ×1
                              │
                              ▼
                    FUSION (norm_max / rrf)      ← số học trên điểm, không hiểu nghĩa
                              │
                     [NGỮ NGHĨA] cross-encoder bge-reranker-v2-m3   100 → 50
                              │
                     [NGỮ NGHĨA] LLM đọc 10 evidence pack (chỉ QA)
```

---

## 1. Từng nhánh dùng kỹ thuật gì

| Nhánh | Kỹ thuật nền tảng | Cụ thể trong code | Hiểu nghĩa? |
|---|---|---|:-:|
| `dense_visual` | **Embedding đa phương thức + cosine** | CLIP ViT-L/14, 768 chiều, vector ảnh dựng offline; quét tuyến tính trong RAM ([vector_stores.py:67](../online/adapters/vector_stores.py#L67)) | ✅ |
| `bm25_caption` | **Okapi BM25** | `k1=1.5`, `b=0.75`, `idf = log(1 + (N−df+0.5)/(df+0.5))` ([bm25.py:144](../online/adapters/bm25.py#L144)) | ❌ |
| `bm25_asr` | Okapi BM25 | cùng index, field `asr` | ❌ |
| `bm25_keyword` | Okapi BM25 | field `keyword` | ❌ |
| `bm25_object` | Okapi BM25 | field `object` — nhãn object dạng chữ | ❌ |
| `bm25_ocr` | Okapi BM25 + lọc lớp phủ | thêm `_strip_overlay`: bỏ chuỗi có `df` > 0.10 trong cùng video, bỏ đồng hồ/ngày tháng theo regex | ❌ |
| `ocr_fuzzy` | **So khớp xấp xỉ mức KÝ TỰ** | NFD bỏ dấu + `đ→d`, prefilter character-trigram, rồi `difflib.SequenceMatcher`: `0.55×token-containment + 0.45×partial-phrase` ([ocr_fuzzy.py](../online/adapters/ocr_fuzzy.py)) | ❌ |
| `color_search` | **Khớp từ vựng đóng** | từ điển 11 tên màu VI/EN, chấm bằng tỉ lệ tag màu của query có trong scene | ❌ |
| `event_search` | Okapi BM25 | dùng lại chính `BM25Index` trên text gộp của event; event khớp thì fan-out ra mọi scene thành viên | ❌ |
| `lexical_hash_fallback` | hash text | **không phải dense**; chỉ xuất hiện khi export thiếu vector — container đổi tên nhánh để `/capabilities` không quảng cáo nhầm | ❌ |

**Điểm mấu chốt:** BM25 chấm điểm bằng *tần suất token hiếm*, không bằng nghĩa.
`"đàn cá quẫy"` và `"bầy cá nhảy"` với BM25 là hai truy vấn không liên quan nếu
caption chỉ chứa một trong hai cách nói.

---

## 2. Ba chỗ ngữ nghĩa nằm ngoài retrieval

### 2.1 Dịch VI→EN trước khi vào CLIP — bắt buộc, không phải tuỳ chọn

Text tower của CLIP chỉ biết tiếng Anh, mà truy vấn thi đấu là tiếng Việt. Không
dịch thì nhánh mạnh nhất **vẫn trả số nhưng số đó gần như vô nghĩa**.
`TranslatingTextEncoder` bọc encoder ([container.py:238](../online/api/container.py#L238)),
gọi LLM nhanh (`gemma-4-31B-it`), có cache trên đĩa nên mỗi câu chỉ dịch một lần.

**Số đo cho câu "gần như vô nghĩa" ở trên** (VISUAL-01, 16/08 — 10 câu khác
nghĩa hẳn nhau, đo cosine GIỮA CÁC CÂU; thấp = phân biệt được):

| model | tiếng Việt | tiếng Anh | khớp vi↔en cùng nghĩa |
|---|---:|---:|---:|
| CLIP ViT-L/14 | **0.912** (max 0.969) | 0.448 | 0.421 |
| jina-clip-v2 | 0.260 | 0.262 | 0.820 |

CLIP dồn mọi câu tiếng Việt về gần MỘT điểm, và một câu tiếng Việt còn gần một
câu tiếng Việt vô quan hơn là gần bản dịch tiếng Anh của chính nó (0.912 so với
0.421). Trên tiếng Anh nó bình thường — nên đây là hỏng vì NGÔN NGỮ, không phải
anisotropy (E5 cũng có mean 0.824 mà vẫn hoạt động tốt).

Đây là lý do `AIC_ENABLE_QUERY_TRANSLATION=true` được [docs/27](27_SYSTEM_ISSUES.md)
xếp là "cải thiện lớn nhất từng đo". Nó cũng là lý do **mọi số `dense_visual` đo
bằng `scripts/eval_kis.py --pipeline build_service` đều dưới sức thật**:
`build_service` không bọc `TranslatingTextEncoder` (cảnh báo tại
[eval_kis.py:313](../scripts/eval_kis.py#L313)).

**Đường thoát khỏi phụ thuộc mạng:** `jina-clip-v2` xử lý tiếng Việt ở cùng mức
phân biệt với tiếng Anh (0.260 / 0.262) và ghép đúng cặp dịch (0.820), tức
không cần bước dịch nào. Đó là cách thứ hai giải quyết đúng vấn đề mà
[docs/32](32_ROUTE2_INPUT_GAP.md) nêu (index sẵn caption EN là cách thứ nhất).
CHƯA có cơ sở để thay CLIP — phép so cần thiết là jina so với **CLIP + dịch**,
và nó chưa được đo; xem [docs/20](20_EXPERIMENT_LOG.md) § VISUAL-01.

### 2.2 Cross-encoder rerank — `bge-reranker-v2-m3`, 100 → 50

Khác BM25 và khác cả CLIP: nó đọc **cặp (truy vấn, tài liệu) cùng lúc** nên bắt
được quan hệ giữa hai vế mà bi-encoder không thấy. Gọi qua FPT `/rerank`
([rerank.py:164](../online/adapters/rerank.py#L164)). Đang bật.

### 2.3 LLM đọc bằng chứng — chỉ ở QA

`Qwen3.6-27B` đọc 10 evidence pack đầu (`AIC_FPT_QA_TOP_N=10`) rồi sinh đáp án.
Đây là chỗ duy nhất trong hệ có "đọc hiểu" theo nghĩa đầy đủ.

> **VLM rerank (`Qwen2.5-VL-7B`) đang TẮT.** Nó là tầng ngữ nghĩa thứ tư về mặt
> thiết kế, nhưng đo được: tiêu 94% số lệnh gọi API của KIS mà **không đổi một
> chỉ số nào** trên cả 4 task.

---

## 3. Những thứ trông như ngữ nghĩa nhưng KHÔNG phải

Đây là phần dễ nhầm nhất khi đọc tên biến.

| Thứ | Trông như | Thực tế |
|---|---|---|
| `AIC_ENABLE_EXPANSION=true` | mở rộng truy vấn bằng LLM | Đang là **từ điển đồng nghĩa cứng**. `AIC_ENABLE_LLM_EXPANSION=false` nên `expander=None`, chỉ còn lexicon tĩnh, và chỉ bọc `caption`/`keyword` — OCR/ASR giữ nguyên văn |
| `action_tags[]` | nhãn hành động do model sinh | Suy bằng **rule từ chuỗi caption** (`extract_action_tags`), không phải model. Đó là lý do độ phủ dừng ở 56.9% |
| `keywords[]` mức scene | từ khoá ngữ nghĩa | `assemble` sinh từ `objects[].label`, tức nhãn object viết lại |
| gating theo modality | hệ hiểu truy vấn hỏi gì | **Khớp chuỗi với danh sách cue cứng**: `TEXT_HINTS` (chữ, biển, logo…) đẩy OCR lên 2.0; `SPEECH_HINTS` (nói, phát biểu, phỏng vấn…) đẩy ASR lên 1.7 ([query_planner.py:66](../online/services/query_planner.py#L66)) |
| `color.dominant_colors` | hiểu màu trong cảnh | Bucket hue thành 11 tên rồi so chuỗi. **Không phân biệt được "áo đỏ" với "nền đỏ"** |
| `event_search` | gom theo sự kiện có ngữ nghĩa | BM25 trên text gộp của event |

### Coverage (BM25-01) — có code, **không chạy**

[lexical_coverage.py](../online/services/lexical_coverage.py) cài phép chấm "truy
vấn được bao phủ bao nhiêu nhóm khái niệm", để phạt candidate chỉ khớp một mảnh.
Nhưng `LexicalRetriever.build()` ([bm25.py:226](../online/adapters/bm25.py#L226))
**không có tham số `coverage`**, nên đường server luôn chạy `CoverageConfig`
mặc định = no-op, tức **BM25 nguyên bản**. Module này chỉ sống trong ablation.

Ghi lại vì bài toán nó định giải là có thật và đã đo được: một scene **ngẫu
nhiên** đã trùng sẵn trung bình **2.25 token OCR** và **6.75 token ASR** với truy
vấn. Nền nhiễu của so khớp token trên corpus này rất cao.

---

## 4. Trọng số mặc định theo modality — bản đồ ưu tiên hiện tại

Từ [query_planner.py:83](../online/services/query_planner.py#L83). Đây là "hệ
đang tin nhánh nào" trước khi người dùng đụng vào:

```
VISUAL  1.00     CAPTION 1.00     KEYWORD 0.65
OBJECT  0.50     ACTION  0.50     COLOR   0.40     EVENT 0.30
OCR     0.35  →  2.00  khi truy vấn có cue chữ
ASR     0.25  →  1.70  khi truy vấn có cue lời nói
```

Ngữ nghĩa (VISUAL) và caption đứng ngang nhau ở đỉnh; mọi thứ còn lại là phụ trợ.

---

## 5. Vì sao tỉ lệ "nghiêng về so khớp từ" lại chạy được trên corpus này

Không phải vì lexical mạnh, mà vì **dữ liệu tình cờ hợp với nó** — và điều đó sẽ
đổi khi mở rộng corpus.

1. **OCR ở đây là lower-third bản tin, mô tả CHÍNH cảnh đang chiếu.** Đo được:
   11/12 gold KIS có OCR trùng từ khoá truy vấn. Nên chữ trên màn hình hoạt động
   như một caption thứ hai, không như chữ ngẫu nhiên. Đây là lý do zero-gating
   của ROUTE-01 bị tắt mặc định: bỏ OCR khi truy vấn không nhắc tới chữ thì mất
   nhiều hơn được.

2. **Truy vấn gold viết bằng đúng từ vựng của caption.** Người viết gold nhìn
   video rồi mô tả, nên trùng từ cao một cách không tự nhiên so với thi đấu thật.

3. **Ngữ nghĩa vẫn cứu đúng chỗ lexical mù.** Ca cụ thể: keyframe f10908 là đàn
   cá quẫy nhưng caption Qwen bịa thành *"một đám cháy lớn đang bùng phát"*. Nó
   **vẫn lên hạng 1** cho truy vấn "đàn cá đang nhảy" — `dense_visual` (0.0218)
   áp đảo `bm25_caption` (0.0137). CLIP nhìn thẳng vào ảnh nên không bị caption
   sai kéo theo.

Nói cách khác: lexical gánh phần lớn recall, **ngữ nghĩa gánh phần lexical không
thể có**.

---

## 6. Fusion — nơi hai họ kỹ thuật gặp nhau, và nó KHÔNG hiểu nghĩa

Chốt hiện tại: `norm_max` cho KIS/TRAKE/AVS, `rrf` cho QA.

| Họ method | Cách tính | Đọc điểm thật? |
|---|---|:-:|
| `rrf`, `weighted_sum`, `max_score`, `intersection`, `union` | `weight / (rrf_k + rank)` — chỉ dùng **thứ hạng** | ❌ |
| `norm_sum`, `norm_max`, `margin_sum`, `entropy_sum` | chuẩn hoá min-max **trong từng nhánh** rồi mới gộp | ✅ |

Đây là cải thiện lớn nhất từ trước tới nay (KIS R@1 0.583 → 0.750, MRR 0.731 →
0.852). Cơ chế đo được là **đập đuôi**, không phải "nhánh chắc chắn thắng":

```
ti le dong gop hang-1 so voi hang-100 CUA CUNG MOT NHANH
  RRF (k=60)     1/61 so voi 1/160   ->  2.62x
  min-max        1.00 so voi ~0.00   ->  vo han
```

RRF cho candidate hạng 100 tận 38% số phiếu của hạng 1; 7 nhánh × 100 candidate
= 700 lá phiếu gần bằng nhau, tín hiệu thật chìm trong đó.

QA đi ngược (giữ `rrf`) vì nó hỏi *"scene nào là BẰNG CHỨNG tốt"* — đồng thuận
nhiều nhánh đáng tin hơn một nhánh rất chắc chắn.

---

## 7. Ngữ nghĩa cho VĂN BẢN — đã cắm vào container, **chưa đo lại**

**Cập nhật 13/08.** `caption_dense` giờ là một nhánh thật của container, bật
bằng `AIC_CAPTION_DENSE_INDEX`. Trước đó code nhánh đã có nhưng chỉ hai script
thí nghiệm dùng được, nên không đo được bằng `--pipeline container` — tức không
so sánh được với bất kỳ baseline nào đã chốt. Xem §7.1 để chạy.

Kết quả DENSE-TEXT-01 cũ (index caption + object + action + keyword, dim 1024,
đo trên Stage B của TRAKE, corpus 1 video):

| | R@20 | R@50 | R@100 | median rank |
|---|---:|---:|---:|---:|
| baseline (BM25) | 15/35 | 18/35 | 21/35 | 13 |
| dense text riêng | 17/35 | 18/35 | 19/35 | **3** |
| baseline + dense | **17/35** | **20/35** | **22/35** | 6.0 |

Đọc đúng bảng này: dense text cải thiện mạnh **thứ hạng** (13 → 3) nhưng gần như
không cải thiện **độ phủ** (21 → 22). Nghĩa là **nút thắt không nằm ở retriever
mà ở offline** — 14 bước không tìm được là vì caption không mô tả nội dung đó,
đổi cách tìm không sinh ra thông tin chưa có.

Hạ tầng nhiều index dense (`AIC_DENSE_INDEXES`) đã cài và test 13 ca, cho phép
chạy song song CLIP + SigLIP + Jina; **chưa có vector nào ngoài CLIP** nên chưa
biết nhiều index có hơn một index không.

### 7.1 Chạy thí nghiệm — hai lệnh

Máy đã đủ điều kiện: `torch`, `transformers`, và `storage/models/multilingual-e5-large`
đều có sẵn trong venv hiện tại.

**Bước 1 — dựng index cho corpus ĐANG PHỤC VỤ** (765 scene, 3 video). Chỉ cần
chạy lại khi export đổi:

```powershell
$env:PYTHONIOENCODING = "utf-8"
python -m scripts.build_caption_dense_index `
    --metadata storage/exports_multivideo/scenes.jsonl `
    --out storage/indexes_multivideo/caption_dense
```

Thêm `--device cuda` nếu muốn nhanh hơn; E5-large trên CPU vẫn chạy được cho 765
document. Index cũ ở `storage/indexes_l21/caption_dense` **không dùng lại được** —
nó chỉ có 216 scene của L21_V001.

Chọn encoder bằng `--encoder {e5,jina_v3}` (mặc định `e5`), và phía online phải
khai KHỚP qua `AIC_CAPTION_DENSE_ENCODER`. Hai họ này khác nhau ở cách bất đối
xứng query-vs-passage: E5 dùng prefix chuỗi `query: `/`passage: `, jina-v3 dùng
LoRA adapter `retrieval.query`/`retrieval.passage` (và tự chèn instruction của
nó, nên KHÔNG nối thêm prefix).

⚠️ Cả hai đều ra vector **1024 chiều**, nên `assert_dimension` KHÔNG bắt được ca
khai lệch model — chạy jina trên index E5 vẫn trả đủ candidate, `branch_status`
vẫn `success`, mọi điểm cosine đều là rác. Chốt bắt ca đó là
`CaptionDenseRetriever.assert_encoder_kind`, đối chiếu `encoder_kind` ghi trong
manifest của index. Kết quả đo hai encoder: [docs/20](20_EXPERIMENT_LOG.md)
§ DENSE-TEXT-03 (hoà, giữ E5).

**Bước 2 — đo, đúng harness đã chốt mọi baseline khác:**

```powershell
$env:PYTHONHASHSEED = "0"
$env:AIC_ENV_FILE = ".env.fpt.local"
$env:AIC_METADATA_JSONL = "storage/exports_multivideo/scenes.jsonl"
$env:AIC_CAPTION_DENSE_INDEX = "storage/indexes_multivideo/caption_dense"

python -m scripts.eval_tasks --pipeline container `
    --gold examples/gold_all3.jsonl `
    --metadata storage/exports_multivideo/scenes.jsonl `
    --max-per-video 0 --disable-branch event_search --disable-branch ocr_fuzzy `
    --json-out outputs/evaluation/quick/dense_text_container.json
```

⚠️ **Giữ nguyên `--disable-branch ocr_fuzzy`.** Bỏ cờ đó ra là lặp lại lần chạy
13/08: `ocr_fuzzy` chạy 8.5s làm `dense_visual` timeout ở 40/84 truy vấn, và
mọi con số sau đó không so sánh được với gì cả.

### 7.2 Kết quả — đo 13/08, **DROP mặc định**

36 truy vấn KIS, tái lập đúng cấu hình `D_FINAL_tiebreak`. Phân tích đầy đủ ở
[docs/20 § DENSE-TEXT-02](20_EXPERIMENT_LOG.md).

| trọng số | R@1 | R@5 | MRR | pairwise | **V003\*** |
|---|---:|---:|---:|---:|---:|
| 0 (baseline) | **0.750** | 1.000 | **0.852** | **0.844** | 0.750 |
| 0.25 | 0.722 | 1.000 | 0.840 | 0.812 | 0.750 |
| 0.5 | 0.722 | 1.000 | 0.840 | 0.812 | 0.750 |
| 1.0 | 0.667 | 0.972 | 0.806 | 0.750 | 0.667 |

Đơn điệu, holdout cùng chiều → giữ `AIC_CAPTION_DENSE_INDEX=` rỗng.

**Nhưng gold này không đo được thứ nhánh đó sinh ra để giải** — truy vấn gold
viết bằng đúng từ vựng của caption (§5). Trên 4 cặp paraphrase tự viết, overlap
top-5 trung bình là **BM25 2.0/5 vs caption_dense 3.75/5**; ca rõ nhất
(*"cán bộ phát biểu trong cuộc họp"* vs *"một người đang trình bày tại hội nghị"*)
BM25 được **0/5** còn dense 4/5 và cùng top-1. Phép thử đó chỉ đo tính nhất
quán, không đo đúng/sai. Quyết dứt điểm cần bộ paraphrase có gold.

### 7.2b `norm_max` biến trọng số thành công tắc — ảnh hưởng mọi sweep sau này

0.5 và 0.25 cho kết quả **trùng từng chữ số** (34/36 truy vấn cùng hạng gold).
Vì `contribution = weight × normalized` với `normalized ∈ [0,1]`, còn
`totals = MAX qua các nhánh`, một nhánh `weight < 1.0` **không bao giờ đặt được
max** cho candidate mà nhánh weight-1.0 cũng chấm mạnh. Nó chỉ còn ảnh hưởng
qua candidate *chỉ mình nó* tìm ra (xếp dưới hết) và qua khoá phụ `sums`.

Điều này sửa lại một quy kết sai ở [docs/30 §3.4](30_SYSTEM_DIAGNOSIS.md):
`color_search` đóng góp 0 **không phải vì RRF pha loãng** — fusion đã là
`norm_max` từ 09/08 — mà vì modality weight 0.4 < 1.0. Cùng cơ chế áp cho
OBJECT 0.5, ACTION 0.5, COLOR 0.4, EVENT 0.3.

⚠️ **Quét trọng số dưới 1.0 gần như vô nghĩa.** Muốn biết một nhánh có giá trị
thật thì quét ≥ 1.0, hoặc hạ trọng số các nhánh khác xuống.

### 7.3 Ba chốt an toàn — vì sao có chúng

Cả ba ca dưới đây **vẫn chạy được** nếu để lọt: nhánh trả đủ candidate, điểm
cosine hợp lệ, `branch_status` báo `success`. Nên chúng chặn ở lúc khởi động.

| Chốt | Chặn ca gì |
|---|---|
| `assert_covers` | Index dựng từ export KHÁC. Dùng index `exports_l21` (216 scene, chỉ V001) cho corpus 765 scene thì nhánh **không bao giờ đề xuất nổi một scene nào của V002/V003** |
| `assert_dimension` | Khai sai model — encoder 768 chiều gặp index 1024 chiều thì cosine vẫn ra số |
| `encoder.warmup()` | Nạp model trong request đầu tiên sẽ vượt `AIC_BRANCH_TIMEOUT_MS` và nhánh biến mất trong im lặng — đúng lỗi đã cắn với text tower CLIP |

Thêm một điểm không phải chốt nhưng cùng loại: `search()` chạy encode trong
`asyncio.to_thread`. Để đồng bộ thì E5 chặn event loop và làm **mọi nhánh khác**
trượt deadline — đúng cơ chế `ocr_fuzzy` đang gây ra.

Test: [tests/test_caption_dense_branch.py](../tests/test_caption_dense_branch.py) (15 ca).

---

## 8. Khi nào dùng gì — quy đổi ra thao tác

| Bạn đang tìm | Kỹ thuật trúng | Thao tác |
|---|---|---|
| Một **cảnh** ("đàn cá quẫy", "người đội mũ bảo hiểm") | ngữ nghĩa hình ảnh | preset *Tìm theo hình ảnh* — `dense_visual` weight 3 |
| **Chữ trên màn hình** (tên người, biển hiệu, con số) | so khớp từ trên OCR | preset *Tìm chữ hiện trên màn hình* — `bm25_ocr` weight 5 |
| Thứ chỉ được **nói ra** | so khớp từ trên ASR | preset *Tìm theo lời nói* — `bm25_asr` weight 3 |
| Chữ OCR **sai chính tả / mất dấu** | so khớp ký tự | `ocr_fuzzy` — nhưng xem cảnh báo ở §9 |
| Câu **hỏi** cần đáp án | LLM đọc bằng chứng | task QA, giữ fusion `rrf` |

---

## 9. Điểm mù của bộ kỹ thuật hiện tại

1. **Không có so khớp CẤU TRÚC.** Không nhánh nào biểu diễn quan hệ hay thuộc
   tính, nên *"người mặc áo đỏ cầm chai"* bị rã thành các token rời — hệ không
   phân biệt được với *"chai đỏ cạnh người mặc áo trắng"*. Route2 sinh sẵn
   `entities[].attributes` và `relations[]`, nhưng contract online chưa có chỗ
   chứa (xem [docs/32 §4](32_ROUTE2_INPUT_GAP.md)).

2. **BM25 không tách từ tiếng Việt.** Tokenizer là `\w+` trên chuỗi đã casefold
   ([bm25.py:17](../online/adapters/bm25.py#L17)), tức cắt theo **âm tiết**:
   `"xanh lá"` thành hai token độc lập, `"xanh"` khớp luôn cả `"xanh dương"`.
   Giữ nguyên dấu, không stemming.

3. **`ocr_fuzzy` đang phá nhánh mạnh nhất.** Đo 13/08: `ocr_fuzzy` chạy p50
   **8.5s**, kéo `dense_visual` từ p50 224ms lên 8 682ms và làm nó **timeout ở
   40/84 truy vấn** (deadline 8 000ms). KIS R@1 tụt 0.750 → 0.611. Đây là lỗi
   vận hành, không phải lỗi thuật toán — nhưng hậu quả là mất hẳn tầng ngữ nghĩa.

4. **`color_search` sống, dữ liệu 100%, đóng góp đúng 0.** Từ vựng đóng 11 màu +
   không có vùng ảnh ⇒ không phân biệt được gì (đỏ xuất hiện ở 99.6% scene).

5. **`bm25_action` trả rỗng ở 84/84 và 120/120 truy vấn đã đo.** Độ phủ 56.9%
   trên giấy nhưng thực tế không đóng góp.

6. **Chưa cần ANN.** Quét tuyến tính 250k vector chỉ mất **6.4 ms** — nút cổ chai
   là I/O lúc khởi động và RAM, không phải tìm kiếm.

7. **Tầng ngữ nghĩa duy nhất phụ thuộc một lời gọi LLM ở đường request.** CLIP
   chỉ hoạt động sau khi `TranslatingTextEncoder` dịch (§2.1). Cache đĩa che
   được chi phí cho truy vấn lặp, nhưng truy vấn MỚI — tức mọi truy vấn lúc thi
   — vẫn phải chờ mạng, và provider hỏng là mất nhánh mạnh nhất. Encoder đa ngữ
   sẵn (jina-clip-v2) bỏ được phụ thuộc này, nhưng chưa đo so với CLIP + dịch.

8. **Không có harness nào đo `dense_visual` ở cấu hình thật.** Cả
   `eval_kis.build_service` lẫn `eval_tasks` đều dựng encoder trần
   ([eval_kis.py:313](../scripts/eval_kis.py#L313),
   [eval_tasks.py:414](../scripts/eval_tasks.py#L414)); chỉ
   `--pipeline container` mới đi qua `TranslatingTextEncoder`. Nghĩa là mọi
   ablation nhánh visual đã ghi trong docs/20 đều đo một CLIP không dịch.

---

## 10. Tài liệu liên quan

- [docs/29](29_DATA_CONTRACT.md) — hợp đồng dữ liệu, trường nào online thực sự đọc
- [docs/30](30_SYSTEM_DIAGNOSIS.md) — chẩn đoán toàn hệ thống + số đo từng task
- [docs/32](32_ROUTE2_INPUT_GAP.md) — §1 vì sao online chỉ đọc `.text`/`.label`
- [docs/20](20_EXPERIMENT_LOG.md) — ROUTE-01, BM25-01, DENSE-TEXT-01, Phase D fusion,
  và **VISUAL-01 (16/08)**: vì sao mọi số `dense_visual` đã ghi đều đo một CLIP
  không dịch, cùng phép đo mức sụp đổ của text tower trên tiếng Việt
- [docs/27](27_SYSTEM_ISSUES.md) — bảng cấu hình đã đo, gồm `AIC_ENABLE_QUERY_TRANSLATION`
- [KAGGLE_OFFLINE_GUIDE](KAGGLE_OFFLINE_GUIDE.md) — notebook rời theo stage,
  gồm `embed-jina-clip-v2.ipynb` cho vector ảnh đa ngữ
