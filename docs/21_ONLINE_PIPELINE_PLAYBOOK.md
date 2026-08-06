# 21 — Online pipeline: kỹ thuật đã dùng, kết quả đo được, bài học

Bản chốt những gì **đã thí nghiệm và đã đo**, không phải danh sách ý tưởng.
Mỗi mục nêu: làm gì, số liệu, và giữ hay bỏ. Nhật ký chi tiết từng thí nghiệm
ở `docs/20_EXPERIMENT_LOG.md`; file này là bản đọng lại.

Điều kiện đo, áp dụng cho MỌI con số dưới đây:

```
dataset      storage/exports_l21_enriched/   (1 video L21_V001, 217 scene, 307 vector ảnh thật)
gold         examples/AIC2026_L21_V001_queries_4tasks.jsonl  (12 KIS + 12 QA + 8 TRAKE + 8 AVS)
lệnh         python -m scripts.eval_tasks --pipeline container --max-per-video 0
môi trường   PYTHONHASHSEED=0, AIC_ENV_FILE=.env.fpt.local
```

> **Bộ gold rất nhỏ và chỉ một video.** Chênh 1 query trên KIS là 0.083 —
> lớn hơn phần lớn khác biệt từng tranh cãi. Đừng kết luận từ chênh 1–2 query.

---

## 1. Cấu hình tốt nhất hiện tại

| Task | Chỉ số | Giá trị |
|---|---|---|
| KIS | R@1 / R@20 / MRR | **0.500 / 1.000 / 0.720** |
| KIS | `top1_pairwise_accuracy` | 0.545 (6/11) |
| QA | answer_accuracy / joint_top1 / MRR | **0.583** / **0.333** / **0.421** |
| QA | `evidence_recall` / `pairing_accuracy` | 0.833 / 0.875 |
| TRAKE | correct_video / mean_r_score | **1.000** / 0.263 |
| TRAKE | `frame_oracle_coverage` / `frame_selection_accuracy` | 0.800 / 0.328 |
| AVS | nDCG@100 / P@100 / event_coverage | **0.453** / **0.500** / **0.308** |
| AVS | `zero_result_rate` | **0.000** (trước: 0.375) |

Cấu hình sinh ra bộ số này:

```ini
AIC_ENABLE_QUERY_TRANSLATION=true    # dịch VI→EN trước CLIP text tower
AIC_ENABLE_LLM_EXPANSION=false       # mở rộng đồng nghĩa: đo được là có hại
AIC_RERANK_VLM_ENABLED=false         # VLM rerank: đo được là không đổi gì
AIC_FPT_ENABLED=true
AIC_FPT_LLM_MODEL=Qwen3.6-27B        # QA (model reasoning)
AIC_FPT_FAST_LLM_MODEL=gemma-4-31B-it # dịch/mở rộng (model trả thẳng)
AIC_FPT_RERANK_MODEL=bge-reranker-v2-m3
AIC_VISUAL_EMBEDDING_MODEL=storage/models/clip-vit-large-patch14
AIC_QA_LLM_RANK_MODE=boost              # PR-1: joint_top1 0.083 -> 0.333
AIC_AVS_GRADE_MODE=semantic_or_lexical  # PR-2: zero_result_rate 0.375 -> 0
```

---

## 2. Kỹ thuật GIỮ — có bằng chứng cải thiện

### 2.1 Dịch truy vấn VI→EN trước text tower của CLIP

Thay đổi có tác động lớn nhất từ trước tới nay.

| | trước | sau |
|---|---|---|
| KIS R@1 | 0.333 | **0.500** |
| KIS R@5 | 0.750 | **0.917** |
| KIS R@20 | 0.917 | **1.000** |
| KIS MRR | 0.547 | **0.720** |
| TRAKE mean_r | 0.225 | 0.263 |
| AVS nDCG | 0.238 | 0.299 |

Lý do không phải may: vector ảnh sinh bằng `openai/clip-vit-large-patch14`,
mà text tower của CLIP **chỉ được huấn luyện trên tiếng Anh**. Đưa thẳng truy
vấn tiếng Việt vào đó là so một câu model chưa từng học với vector ảnh — vẫn
ra số, vẫn xếp hạng, nhưng gần như vô nghĩa.

Cài đặt: bọc encoder (`TranslatingTextEncoder`), không sửa `DenseRetriever` —
vector store và fusion phía sau không cần biết có bước dịch. Dịch hỏng thì để
nhánh `failed`, **không** lặng lẽ encode bản tiếng Việt: "fallback" kiểu đó
chính là tái lập trạng thái hỏng mà không ai thấy.

### 2.2 Weighted RRF theo THỨ HẠNG, không cộng điểm thô

`weight / (rrf_k + rank)`. Điểm của BM25 và của cosine similarity không cùng
thang; cộng thẳng là để nhánh có thang rộng hơn nuốt các nhánh còn lại.

### 2.3 Chuẩn hoá điểm TRƯỚC dedup

`ScoreNormalizers.from_pool()` phải chốt trên pool đầy đủ. Tính sau dedup thì
mẫu số đổi theo `max_results_per_video`, và chỉ nới cap hiển thị cũng làm xáo
trộn thứ hạng đã có (EVAL-01, prefix invariance).

### 2.4 Nạp model NGOÀI request path, và nạp nguyên tử

Lần `encode()` đầu nạp CLIP (~3s), vượt deadline nhánh 3000ms nên bị
`asyncio.wait_for` huỷ; huỷ giữa `from_pretrained` để lại model kẹt trên
`meta` device nên lần sau hỏng hẳn. Hệ quả: **1–2 truy vấn đầu mỗi tiến trình
chạy không có nhánh dense, im lặng.** Sửa: `warmup()` gọi lúc dựng container,
`_load()` có khoá và chỉ gán vào `self` sau khi dựng xong.

### 2.5 Fail-fast với lỗi CẤU HÌNH, degrade-có-cảnh-báo với lỗi RUNTIME

Ranh giới này quan trọng và từng bị làm sai theo cả hai chiều:

- Model không nạp được vì trỏ sai đường dẫn → **chặn khởi động**. Đây là lỗi
  người vận hành, và server vẫn lên được nghĩa là mọi phép đo sau đó đều sai.
- FPT trả 502 giữa chừng → **giữ thứ hạng stage trước + warning**. Đây là lỗi
  nhất thời, giết cả kết quả tìm kiếm vì nó là đánh đổi tệ.

### 2.6 QA answer bằng LLM

`answer_accuracy` 0.333 → 0.500 (chỉ riêng việc sửa cho nó chạy được) → 0.583
(khi cộng thêm bước dịch). Kèm rule-based `ANSWER_TOOLS` làm nền, vì luật chấm
tính bất kỳ dòng nào đúng cả ba (video/frame/answer), không riêng dòng đầu.

### 2.7 Cửa sổ chấm TRAKE thích ứng theo độ dài scene

`clamp(scene_duration_sec * 0.5, 2.0, 7.0)`. Bản đầu dùng cửa sổ ±4 frame và
dẫn tới một chẩn đoán sai hoàn toàn.

### 2.8 Lọc bằng chứng bằng LLM ở bước cuối

`EvidencePack.rerank_text()` gộp máy móc caption + OCR + ASR + object, nên lớp
phủ đồ hoạ của đài đứng ngang hàng với nội dung cảnh. Quan sát thật: bằng
chứng trả về là `HTV9 HD` và `06:33:29` cho truy vấn không hỏi kênh nào cũng
chẳng hỏi mấy giờ. Những chuỗi đó có ở **mọi** khung hình nên không chứng minh
được gì.

Sau khi lọc, giữ `Một cột nước cao vút phun lên từ giếng...`, loại `HTV9 HD` /
`06:33:29` / `60 GIÂY`. Truy vấn không khớp thì trả `supports: false` thay vì
dựng bằng chứng giả.

*Đã kiểm chứng định tính, chưa đo trên metric.*

### 2.9 LLM đề xuất trọng số nhánh

Với `bảng hiệu có chữ "Gừng cay muối mặn"`: `bm25_ocr=3.0`, `ocr_fuzzy=2.5`,
tắt hẳn 7 nhánh còn lại kể cả `dense_visual=0.0`.

**Chỉ đề xuất, không tự áp.** Trọng số đổi ngầm giữa hai lần tìm là kiểu thay
đổi khiến không ai tái lập được kết quả.

*Đã kiểm chứng định tính, chưa đo trên metric.*

---

## 3. Kỹ thuật ĐÃ THỬ và BỎ — kèm lý do

| Kỹ thuật | Kết quả đo | Vì sao bỏ |
|---|---|---|
| **VLM rerank** (Qwen2.5-VL) | trùng baseline **đúng từng chữ số** trên cả 10 chỉ số, sau ~1400 lệnh gọi | Cho điểm rất phân cực (0.80 / 0.00) nên đa số candidate hoà điểm, sort ổn định giữ nguyên thứ tự. Giá trị của nó là *bác bỏ dương tính giả*, mà bộ gold này không ép được — `correct_video_rate` đã 1.000 từ đầu |
| **Mở rộng query bằng LLM** (đồng nghĩa VI) | KIS MRR 0.547→**0.512**, AVS nDCG 0.238→**0.201** | Query drift: term đồng nghĩa kéo theo candidate chỉ khớp term phụ. Chỉ có ích cho QA (MRR 0.261→0.291) |
| **Lexicon VI→EN cứng** cho BM25 | 3/4 truy vấn thử ra kết quả y hệt | Viết khi caption còn tiếng Anh. Caption nay là tiếng Việt nên term tiếng Anh khớp 0 token |
| **ROUTE-01** — ép OCR/ASR về 0 | trượt tiêu chí giữ | Giữ cơ chế sau cờ `allow_zero_modality` |
| **BM25-01** — concept coverage | hỏng đúng trên case mục tiêu | Giữ cơ chế sau `CoverageConfig` |
| **DENSE-TEXT-01** — caption dense (E5) | trượt 2/3 tiêu chí | Giữ code: sẽ có giá trị ngay khi caption tốt lên |
| **CAPTION-ENRICH-01** | caption tốt lên thật (6/10 scene) nhưng **không chỉ số nào đổi** | Đo nhầm tầng — retrieval scene đã 35/35 từ sớm |
| **FRAME-REFINE-01** | ba chiến lược chọn frame khác nhau đều cho 19/23 | Nút thắt không nằm ở cách chọn |

---

## 4. Trần của dữ liệu — và chỗ KHÔNG phải trần

Đây là các phép **đếm**, không phụ thuộc thứ hạng, nên không bị ảnh hưởng bởi
những sai sót đo đạc ở mục 6:

```
scene retrieval                          35/35   đã đạt trần
frame_oracle_coverage                    23/35
frame_selection_accuracy_given_oracle    19/23
```

> **Đính chính.** Bản trước của mục này viết rằng 7/35 bước TRAKE có candidate
> "không tồn tại trong corpus" và gọi đó là trần cứng. **Sai.** Đo lại khoảng
> cách từ mỗi frame gold tới keyframe gần nhất:
>
> | nửa cửa sổ | bước không có keyframe | trần R-score |
> |---|---|---|
> | ±2s (đang dùng) | 7/35 | 0.800 |
> | ±3s | 1/35 | 0.971 |
> | **±4s** | **0/35** | **1.000** |
>
> Khoảng cách xa nhất là **92 frame = 3.07s** (fps=30), còn keyframe cách nhau
> trung vị 120 frame = 4s. Mọi bước TRAKE **đều có** keyframe trong bán kính
> 4s. Con số 7/35 là hệ quả của **cận dưới 2s** trong
> `clamp(scene_duration * 0.5, 2.0, 7.0)`, không phải của dữ liệu.

Việc trích keyframe theo stride cố định ~120 frame (không theo scene) vẫn là
một khiếm khuyết offline có thật — nó làm 119 scene ngắn rơi qua lưới và bị
schema loại. Nhưng nó **không** chặn TRAKE: mọi bước gold đều nằm trong tầm
với của một keyframe.

Nút thắt thật của TRAKE là **việc CHỌN frame**, không phải sự tồn tại của
candidate và cũng không phải cửa sổ chấm — đã đo: nới cửa sổ 2s→4s đẩy trần
lên +0.200 nhưng điểm thật chỉ được +0.056. Xem `docs/22_TRAKE_CHAIN_SCORING.md`.

---

## 5. Vì sao điểm không tăng — chẩn đoán theo từng tầng

Điểm chung của cả bốn task: **hệ tìm ĐƯỢC tài liệu đúng rồi hỏng ở bước chấm
hoặc lọc cuối cùng.** Không task nào đang thiếu recall.

### KIS — sai lệch chỉ ở hạng 1 vs hạng 2

`first_frame_hit_rank` của 12 query:

```
hạng 1: 6 query      hạng 2: 5 query      hạng 7: 1 query
```

**11/12 query có đáp án ở hạng 1 hoặc 2**, và `R@20 = 1.000`. Đẩy được 5 query
từ hạng 2 lên hạng 1 thì `R@1` đi từ 0.500 lên 0.917.

Đây là bài toán **phân biệt tinh ở đỉnh danh sách**, không phải bài toán tìm
kiếm. Retrieval mạnh hơn không giúp gì — nó đã đưa đáp án vào top-2 rồi.

### QA — trả lời đúng nhưng ghép sai dòng

`answer_accuracy = 0.583` (7/12 câu có đáp án đúng trong danh sách) nhưng
`joint_top1 = 0.083` (1/12 câu có dòng hạng 1 đúng cả video + frame + answer).

Nguyên nhân đã định vị: `QaProcessor._enhance_with_llm` thay `answer` nhưng
không đụng `joint_score`, nên `confidence` của LLM bị vứt và thứ hạng vẫn do
điểm rule-based quyết định. Đây là khoảng cách lớn nhất của toàn hệ thống, và
là lỗi rõ ràng chứ không phải giới hạn năng lực.

### AVS — bị một cổng TỪ VỰNG ở cuối vứt hết kết quả

`result_count` của 8 query: `0, 0, 0, 1, 2, 3, 3, 3`. Ba query trả về **không
kết quả nào** dù top-100 được phép.

`AvsCriteria.grade()` chấm 0–3 bằng **khớp token nguyên văn** trên các nhóm
inclusion, và `min_grade = 1` loại thẳng mọi candidate không khớp chữ nào.
Toàn bộ retrieval ngữ nghĩa phía trên — CLIP, dense, rerank — bị một bộ lọc
từ vựng ở bước cuối vứt đi.

Cùng loại lỗi với chuyện lexicon VI→EN: một tầng rule-based được viết theo giả
định cũ, đứng chắn trước tầng ngữ nghĩa.

### TRAKE — đã loại ba giả thuyết, còn một

Ràng buộc hình thức, cửa sổ chấm, và thuật toán ghép chuỗi đều đã bị loại bằng
đo đạc (xem `docs/22`). Còn lại: `s_i` là điểm **tương đối** trong khi R-score
hỏi câu **tuyệt đối**.

### Hai rủi ro của chính phép đo

**Bộ gold quá nhỏ.** 12 query KIS nên 1 query = 0.083; 8 query AVS/TRAKE nên
1 query = 0.125. Không kết luận được gì từ thay đổi dưới 2 query.

**Chỉ có MỘT video.** `correct_video_rate` luôn bằng 1.000 vì không có video
nhiễu nào để nhầm. Cuộc thi thật có ~800k keyframe. Mọi kết luận về xếp hạng ở
đây đều chưa được thử thách bởi phần khó nhất của bài toán — phân biệt giữa
các video khác nhau.

---

## 5b. Chỗ còn hở đã xác định

**LLM confidence bị vứt khi xếp hạng QA.** `QaProcessor._enhance_with_llm`
thay `answer`/`answer_type`/`verifier_status` nhưng không đụng `joint_score` —
mà `joint_score` mới quyết định thứ hạng. Quan sát trên submission thật, câu
hỏi "Cột nước phun lên từ đâu?":

```
rank 1  L21_V001,6933,nguồn nước gần người đàn ông
rank 2  L21_V001,6933,người
rank 3  L21_V001,6099,giếng        ← LLM chấm 0.95, đây là đáp án đúng
```

`joint_top1 = 0.083` phần lớn đến từ đây. Là thay đổi ranking nên phải đo
trước khi giữ.

---

## 6. Bài học

### 6.1 Đo nhầm hệ thống là nguồn sai lớn nhất, hơn mọi lỗi thuật toán

`scripts/eval_kis.py::build_service` là **định nghĩa pipeline thứ hai**, không
đi qua `online/api/container.py`. Nó thiếu nhánh object/action/color/event,
thiếu VLM rerank, không bọc encoder. Mọi số trong nhật ký trước FPT-WIRE-01
đều là số của bản dựng đó — trông hoàn toàn hợp lệ, và không phải số của hệ
thống mà server chạy.

Chỉ riêng việc đo đúng pipeline đã đổi kết quả nhiều hơn bất kỳ thí nghiệm
ranking nào: KIS MRR 0.442→0.547, TRAKE mean_r 0.075→0.225.

> **Quy tắc rút ra: harness phải dựng hệ thống qua ĐÚNG composition root mà
> production dùng.** Nay có `--pipeline container`; bản `legacy` chỉ còn dùng
> cho ablation từng nhánh và **không được dùng để báo cáo điểm**.

### 6.2 HTTP 200 không chứng minh gì — hệ này hạ cấp chứ không sập

Một nhánh retrieval `failed` vẫn cho ra bảng kết quả đầy đủ và mã 200. Tín
hiệu nghiệm thu thật là `branch_status.dense_visual.state == "success"`.

Ba lần hạ cấp âm thầm đã xảy ra và không ai phát hiện trong nhiều đợt đo:

1. `.env.fpt.local` **chưa từng được nạp** — không có dotenv loader nào trong
   `online/`. `uvicorn` khởi động với `fpt_enabled=False`, không một dòng cảnh
   báo. Một buổi đo tưởng là "có FPT" thực ra chạy hoàn toàn không FPT.
2. `warmup()` nuốt exception → `dense_visual` chết ở mọi request.
3. **QA LLM chưa từng chạy được lần nào** (xem 6.3).

### 6.3 Model reasoning phá vỡ giả định về `content`

`Qwen3.6-27B` trên FPT trả `reasoning_content` và tiêu ngân sách token cho nó
**trước** rồi mới sinh câu trả lời. Hai chế độ hỏng khác hẳn nhau:

- `max_tokens` hẹp → `content=None`, `finish_reason="length"`. Cần ~1650 token
  chỉ để dịch một câu; `max_tokens=200` hard-code nên hỏng 100%.
- `response_format={"type":"json_object"}` → câu trả lời **hoàn chỉnh nằm
  trong `reasoning_content`**, `content=None`, `finish_reason="stop"`, chỉ tốn
  22 token. Đọc sai field làm hỏng toàn bộ QA.

Đường json_object rẻ hơn 50 lần (22 vs 1091 token) nên đáng giữ — chỉ cần đọc
đúng field.

**Vai model quan trọng hơn tên model.** Cùng một câu dịch:

| model | token | `content` |
|---|---|---|
| gemma-4-31B-it | 9 | OK |
| gpt-oss-20b | 165 | OK |
| DeepSeek-V4-Flash | >200 | None (reasoning) |
| Qwen3.6-27B | 1652 | OK (reasoning) |

`DeepSeek-V4-Flash` được ghi sẵn ở vai "LLM nhanh" nhưng chính nó là model
reasoning — sai vai.

### 6.4 File cấu hình có thể nói dối

`.env.fpt.local` khai báo 9 model. Thực tế `Settings` chỉ đọc 3 biến, và một
trong ba (QA) hỏng ở mọi lệnh gọi → **chỉ 2 model thật sự sống**. Cả một mục
"provider-neutral routing" không có một dòng code nào đọc tới.

> Biến môi trường không có consumer phải được **đánh dấu là chưa wire** ngay
> trong file, nếu không nó là tài liệu sai nằm cạnh cấu hình thật.

### 6.5 Tương quan không phải nhân quả

Nhiễu kết quả từng được chẩn đoán là "`PYTHONHASHSEED` rò vào ranking" vì cố
định seed thì hết dao động. Nguyên nhân thật là cold-start nạp CLIP: cố định
seed chỉ đổi *thời điểm/thứ tự*, không đụng gì tới hash order. Cách truy ra:
đổ dấu vết **từng tầng** rồi tìm điểm phân kỳ đầu tiên, thay vì chỉ so metric
cuối — metric cuối chỉ cho biết "có khác", không cho biết "khác từ đâu".

### 6.6 Một metric trộn hai tầng sẽ giấu nút thắt

Ba thí nghiệm liên tiếp (DENSE-TEXT-01, SCENE-COVERAGE-01, CAPTION-ENRICH-01)
đều tối ưu tầng retrieval, trong khi tầng đó **đã đạt 35/35 từ sớm**. Không ai
tách được cho tới khi gặp mâu thuẫn "đầu vào tốt lên, đầu ra đứng yên" —
chính mâu thuẫn đó mới ép phải tách metric thành `scene_recall` /
`frame_oracle_coverage` / `frame_selection_accuracy_given_oracle`.

> **Khoá metric và oracle TRƯỚC khi sửa logic**, nếu không sẽ tối ưu nhầm tầng
> và không biết mình đang nhầm.

### 6.7 Ngân sách token là thuộc tính của PROMPT, không phải của chỗ gọi

`max_tokens=200` hard-code trong `qa_llm.py` là cách QA hỏng 100%. Nay mọi
prompt nằm ở `online/prompts/registry.py`, mỗi cái khai **vai model**
(`fast`/`reasoning`/`vlm`) và ngân sách của chính nó. Khai vai chứ không khai
tên model: tên model là chuyện môi trường, còn "việc này có cần suy luận nhiều
bước không" là thuộc tính của chính việc đó.

### 6.8 Mở rộng query dễ hại hơn lợi

Cả hai lần thử đều làm giảm chỉ số chính. Cơ chế giống nhau: thêm term làm
tăng recall của những candidate chỉ khớp phần phụ, và BM25 không phân biệt
được term nào là bắt buộc.

---

## 7. Việc tiếp theo, theo thứ tự

1. **Sửa `joint_score` cho QA** (mục 5) rồi đo — chỗ hở đã xác định rõ nhất.
2. **Đo lại prompt expansion v2** (đã siết chỉ nhận thứ nhìn thấy được, hạ trần
   term 6→3); v1 gây drift.
3. **Sửa bước trích keyframe offline** để mỗi scene có ít nhất một frame — mở
   trần cứng 7/35 của TRAKE. Đây là việc có trần cao nhất, và không làm được ở
   tầng online.
4. Chạy lại toàn bộ benchmark qua `--pipeline container` để thay thế các con
   số cũ trong `docs/20_EXPERIMENT_LOG.md`.
