# 25 — Baseline P2 và quyết định hướng rerank

> **Đã bị `docs/26` sửa ở hai chỗ. Đọc kèm.**
>
> 1. **Con số AVS ở §1 (nDCG 0.522) không dùng được.** Chỉ số nDCG khi đó tính
>    sai và có thể vượt 1 (2/24 truy vấn ở chính lượt này, max 1.429). Nó cũng
>    chạy ở `AIC_AVS_MAX_RESULTS_PER_VIDEO=3` trong khi `.env.fpt.local` hiện
>    là 20. Mọi kết luận AVS phải đo lại.
> 2. **Kết luận §2 "tầng chấm điểm task gánh toàn bộ" chỉ đúng một nửa.** Tính
>    theo từng truy vấn, `KisProcessor` đẩy gold XUỐNG ở 15 truy vấn và lên ở
>    10. Xem `docs/26` §1.
>
> Các con số KIS/QA/TRAKE ở §1 vẫn dùng được làm mốc so sánh.

Baseline sạch đầu tiên: sau khi sửa `branch_ceiling`, lọc lớp phủ OCR, và tắt
hai nhánh có hại. 120 truy vấn, 3 video, `PYTHONHASHSEED=0`.

```
dataset   storage/exports_multivideo   765 scene, 3 video, dữ liệu đối xứng
gold      examples/gold_all3.jsonl     120 truy vấn (40 mỗi video)
tắt       event_search, ocr_fuzzy
```

---

## 1. Con số baseline

| Task | Chỉ số | Giá trị |
|---|---|---|
| KIS | R@1 / MRR / `top1_pairwise` | **0.500 / 0.671 / 0.692** |
| QA | R@1 / MRR / answer_acc / `joint_top1` | 0.611 / 0.653 / 0.583 / 0.389 |
| TRAKE | `video_recall@1` / `mean_r` | 0.542 / 0.183 |
| AVS | nDCG@100 / P@100 | 0.522 / 0.198 |

So với trước đợt kiểm định (cùng gold, cùng dữ liệu):

| | trước | sau | nguồn cải thiện |
|---|---|---|---|
| KIS R@1 | 0.361 | **0.500** | sửa `branch_ceiling`, lọc OCR, tắt 2 nhánh |
| KIS `top1_pairwise` | 0.481 | **0.692** | ↑ |
| QA R@1 | 0.472 | **0.611** | ↑ |
| AVS nDCG | 0.485 | **0.522** | ↑ |

Không có cải thiện nào đến từ việc thêm model. Tất cả là **gỡ bỏ thứ đang sai**.

---

## 2. Phát hiện quyết định: tầng nào đang làm việc

Đây là lý do P2 tồn tại — tách thứ hạng của candidate đúng theo TỪNG TẦNG:

| tầng | gold ở hạng 1 |
|---|---|
| có mặt trong pool | 32/36 |
| **sau fusion (RRF)** | **2/32** |
| **sau rerank (FPT `bge-reranker-v2-m3`)** | **1/27** |
| **sau task-scoring (`KisProcessor`)** | **18/36** |

Hạng của gold trong pool: p50 = **6**, max 95.

Ba điều đọc ra, mỗi điều đổi một quyết định:

**Fusion gần như không xếp hạng được.** RRF đặt đáp án đúng lên đầu 2/32 lần
(6%). Nó làm tốt việc GOM candidate (32/36 có mặt) nhưng không phân định được
thứ tự.

**Reranker API hiện KHÔNG giúp.** `bge-reranker-v2-m3` qua FPT đang chạy thật
(không có warning) mà hạng 1 vẫn là 1/27 — không hơn fusion. Đây là bằng chứng
trực tiếp chống lại phương án "dùng API reranker": nó đã được dùng rồi, và
không tạo ra khác biệt ở đỉnh.

**Tầng chấm điểm task đang gánh toàn bộ.** `KisProcessor` (signature coverage,
rare cue, safe-frame) kéo từ 2/32 lên 18/36. Mọi giá trị xếp hạng của hệ thống
hiện nằm ở đây, không nằm ở retrieval.

---

## 3. `agreement` là tín hiệu yếu, kể cả sau khi sửa

Số nhánh khớp candidate ĐÚNG: `[4, 5, 5, 5, 5, 4, 5, 5, 5, 2, 4, 2]`
Số nhánh khớp cao nhất trong pool: `[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]`

Candidate đúng thường có 4–5 nhánh, mà pool cũng luôn có ai đó đạt 5. Nghĩa là
**đếm số nhánh không phân biệt được đúng/sai**. Sau khi sửa `branch_ceiling`,
số hạng này đóng góp tối đa 0.15 — đúng vai phụ, và dữ liệu cho thấy nó xứng
đáng ở vai đó.

---

## 4. Latency: vấn đề timeout đã hết

| nhánh | p50 | max |
|---|---|---|
| `dense_visual.raw` | 181ms | 248ms |
| `bm25_caption.expanded` | 6ms | 13ms |
| các nhánh BM25 khác | 1–4ms | ≤9ms |

Trước khi tắt `ocr_fuzzy`, `dense_visual` mất 1.6–4.6s vì tranh CPU. Nay 248ms
so với ngân sách 8000ms — dư 32 lần. Có thể hạ deadline lại nếu muốn, nhưng
không có lý do gấp.

---

## 5. Keyword extraction — DROP

Chạy lại trên nền đã sửa, để kiểm giả thuyết "nó từng bị `agreement` lấn át":

| | tắt | bật |
|---|---|---|
| KIS R@1 | **0.500** | 0.500 |
| KIS `top1_pairwise` | 0.692 | **0.720** |
| QA R@1 | **0.611** | 0.556 |
| TRAKE `mean_r` | **0.183** | 0.149 |

Giả thuyết SAI. Nó nhích 1 query ở `top1_pairwise` nhưng mất 2 query QA và tụt
TRAKE. Giữ code (có chỉ số riêng đo được: token_recall 0.374 -> 0.524), mặc
định TẮT.

---

## 6. Quyết định hướng rerank

Dữ liệu ở mục 2 loại bớt hai lựa chọn và chỉ rõ một:

**Bỏ: "dùng API reranker".** Đã dùng rồi. `bge-reranker-v2-m3` không cải thiện
hạng 1 (1/27 so với 2/32 của fusion). Đổi sang một API reranker khác là đánh
cược không có cơ sở.

**Bỏ: "finetune model retrieval".** Recall candidate đã là 32/36. Vấn đề không
phải tìm không ra, mà là xếp không đúng. Finetune encoder cải thiện recall —
thứ đang không thiếu.

**Nên làm trước: hiểu vì sao `KisProcessor` thắng cả fusion lẫn rerank.** Nó
dùng signature coverage + rare cue + safe-frame trên văn bản scene. Nếu tín
hiệu đó mạnh tới vậy, hai câu hỏi tự nhiên:

1. Đưa nó SỚM hơn — vào chính fusion — có tốt hơn không? Hiện nó chỉ chạy sau,
   trên top-k đã bị fusion cắt.
2. `top1_pairwise = 0.692` nghĩa là còn 8/26 cặp top-2 chọn sai. Xem 8 cặp đó
   khác nhau ở đâu mới biết cần thêm tín hiệu gì — đó là việc phân tích lỗi,
   rẻ và không cần model.

**Distillation: chưa đủ cơ sở.** Tiền đề của nó là "có một model đắt phán đúng
để học theo". Hiện chưa có: VLM rerank đo được là không đổi gì (FPT-WIRE-01),
và confidence của LLM không mang thông tin xếp hạng (QA-JOINT-01). Cần PR-6
(thăm dò VLM verifier trên đúng 8 cặp sai) chứng minh có tín hiệu trước.

---

## 7. Việc tiếp theo, theo thứ tự

1. ~~**Phân tích 8 cặp top-2 chọn sai**~~ — ĐÃ LÀM, xem `docs/26`. Làm được
   hoàn toàn từ `outputs/evaluation/p2/kw_false.json`, không cần chạy lại
   pipeline. Nó tìm ra 5 lỗi, trong đó 2 lỗi đã đưa KIS R@1 từ 0.500 lên
   **0.583** và 1 lỗi làm hỏng toàn bộ chỉ số AVS.
2. **Thử đưa tín hiệu của `KisProcessor` vào fusion** — hoãn. Trong eval,
   `--max-per-video 0` cho toàn bộ pool đi tới `KisProcessor` rồi, nên đổi thứ
   tự tầng không đổi được gì ở đây; nó chỉ có nghĩa khi chạy production có cap.
3. **PR-6 thăm dò VLM verifier** — hoãn tới khi hết lỗi rẻ. Bốn lỗi tìm được ở
   `docs/26` đều không cần model nào.
4. TRAKE Stage A vẫn là nút thắt riêng (`video_recall@1` 0.542, và 1.000 trên
   V001 so với 0.38 trên holdout) — xem `docs/22`.
5. **Đo lại AVS từ đầu** với metric đã sửa, và chốt `avs_max_per_video`.
