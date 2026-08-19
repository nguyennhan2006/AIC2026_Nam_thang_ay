# 37 — Tận dụng hệ thống khi đã chạy được: tính năng và độ tự tin

Lập **2026-08-19**. Dành cho lúc hệ đã lên (`docs/36`), trả lời câu hỏi tiếp
theo: **tin nhánh nào, dùng tính năng nào, và đừng tin cái gì.**

Số đo chính trong tài liệu này lấy từ `outputs/eval_all3_873video_rerank.json`
và `outputs/eval_all3_873video_norerank.json` — **120 truy vấn gold chạy trên
đúng corpus thi đấu 873 video / 87.742 scene**, không phải bản cũ 765
scene/3 video. Số cũ trong `docs/27` vẫn được trích để so sánh, nhưng luôn ghi
rõ là "sân nhà" (không distractor).

---

## 1. Độ tự tin theo task — nhìn thẳng vào số, đừng lạc quan hộ hệ thống

| Task | Chỉ số chính | Có rerank | Không rerank | Đọc thế nào |
|---|---|---:|---:|---|
| **KIS** | R@1 / R@5 / R@20 / MRR | 0.50 / 0.83 / 0.94 / 0.647 | *(rerank không đụng KIS)* | **Đáng tin nhất.** 94% truy vấn có đáp án trong 20 dòng đầu — duyệt tay 20 dòng là khả thi |
| **QA** | R@1 / answer_accuracy / joint_top1 | 0.25 / 0.28 / 0.17 | 0.11 / 0.11 / 0.00 | **Tầm trung, PHỤ THUỘC rerank.** Rerank tắt thì joint_top1 về **0** — tức dòng đầu không bao giờ vừa đúng evidence vừa đúng đáp án |
| **AVS** | nDCG@100 / P@100 | 0.169 / 0.014 | như trên | **Yếu — nhiễu cao.** P@100=0.014 nghĩa là trong 100 kết quả đầu chỉ ~1-2 cái thật sự liên quan. Phải duyệt tay, đừng nộp nguyên rổ top-100 |
| **TRAKE** | video_recall@1 / complete_chain_rate | 0.25 / **0.00** | 0.25 / 0.00 | **Yếu nhất.** 75% truy vấn hệ **chọn sai cả video**, và **0/24 truy vấn có chuỗi hoàn toàn đúng** ở cả hai cấu hình |

So với "sân nhà" 3 video không distractor (`docs/27`, đo 2026-08-06):
KIS R@1 0.583→0.50, TRAKE video@1 0.833→0.25. Thêm 870 video nhiễu **kéo TRAKE
sập gần 4 lần** — dự đoán đúng của `docs/27` §D3 ("KIS R@20 vỡ dần quanh 30-60
video") còn nhẹ hơn thực tế đo được ở TRAKE.

**Kết luận thao tác:** thứ tự đầu tư thời gian duyệt tay nên **ngược với độ
tin cậy** — KIS ít cần soi kỹ, TRAKE/AVS cần soi từng dòng trước khi nộp.

---

## 2. Rerank: luôn phải đang BẬT, và đây là lý do

`.env.fpt.local` đặt `AIC_FPT_RERANK_MODEL=bge-reranker-v2-m3` (cross-encoder,
100→50) và `AIC_FPT_LLM_MODEL=Qwen3.6-27B` (đọc evidence sinh câu trả lời QA).
Cả hai chỉ chạy khi `AIC_FPT_ENABLED=true` **và** mạng/quota FPT còn sống.

Lợi ích đo được, đúng corpus 873 video:

| Chỉ số | Tắt | Bật | Chênh |
|---|---:|---:|---:|
| QA R@1 | 0.111 | 0.250 | **×2.25** |
| QA joint_top1 | 0.000 | 0.167 | từ không gì lên có |
| TRAKE `frame_oracle_coverage` | 0.000 | 0.845 | frame đúng có nằm trong ứng viên không |
| TRAKE `frame_selection_accuracy` | 0.000 | 0.404 | có ứng viên rồi có chọn đúng không |

Dòng TRAKE đáng chú ý nhất: **không rerank thì hệ có ứng viên đúng trong tay
mà 0% chọn được nó.** Đây khớp với phân tích ở `docs/22` §3 — điểm ứng viên
mặc định là *tương đối* (đứng nhất trong đám sai vẫn được điểm cao), còn rerank
(VLM/cross-encoder) là nguồn *tuyệt đối* duy nhất trả lời "keyframe này có
đúng khớp hành động không".

**Rủi ro vận hành:** rerank là một cuộc gọi mạng ra FPT. Hệ **không sập khi nó
lỗi** — `branch_status`/nguồn câu trả lời QA tụt về `ocr_exact`/rule-based
trong im lặng, API vẫn 200. Nếu giữa buổi thi thấy QA/TRAKE đột nhiên tệ hẳn,
**kiểm tra rerank trước khi nghi ngờ retrieval**:

```powershell
curl.exe -s -H "Authorization: Bearer $key" localhost:8000/v1/search/capabilities | python -m json.tool
# tìm "rerank": {"text": true, ...}
```

Xem field `source` trong `qa[]` của response — `fpt_llm` là rerank thật đang
chạy, `ocr_exact`/rule khác là đã rơi về im lặng (docs/12 §7).

---

## 3. Nhánh nào SỐNG trên corpus thi đấu — đừng tin vào tên nhánh, tin vào số đo

Đo trên 873 video / 120 truy vấn gold (`docs/36` §5). Đây là bảng quan trọng
nhất của tài liệu này: **5/10 nhánh đăng ký đã chết trên đúng dữ liệu sẽ thi.**

| Nhánh | success | p50 | Ghi chú |
|---|---:|---:|---|
| `dense_visual` | 120/120 | 2 382 ms | Sống, mạnh nhất. Cần `AIC_BRANCH_TIMEOUT_MS=30000` (mặc định 8000 sẽ cắt nó) |
| `bm25_caption` | 120/120 | 787 ms | Sống, phủ tốt (caption VI 95,3%) |
| `bm25_asr` | 96/120 | 241 ms | Sống một phần — 24 truy vấn bị cổng modality bỏ qua vì không có cue "nói/phát biểu" |
| `bm25_object` | 88/120 | 17 ms | Sống, mới sửa lệch ngôn ngữ VI/EN (commit `a7cbf07`) |
| `color_search` | 15/120 | 0 ms | Gần như chết — từ điển 11 màu đóng, đỏ xuất hiện ở 99,6% scene nên không phân biệt được gì |
| `bm25_ocr`, `ocr_fuzzy` | **0/120** | — | **Chết.** Pack thi đấu OCR = 0,00% trên cả 873 video |
| `bm25_keyword` | **0/120** | — | Chết — pack để `keywords: []` |
| `bm25_action` | **0/120** | — | Chết — pack để `action_tags: []` |
| `event_search` | **0/120** | — | Chết — 11.079 event nhưng `event_caption: null` |

**Hệ quả trực tiếp lên preset UI:** preset "Tìm chữ hiện trên màn hình"
(`bm25_ocr` weight 5, theo `docs/33` §8) **vô dụng trên corpus thi đấu** dù nó
từng là chiến lược tốt trên dữ liệu L21 gốc (11/12 gold KIS ở đó có OCR trùng
từ khoá). Đừng dùng preset này khi thi trừ khi đã tự vá OCR.

Suy ra chiến lược đặt trọng số thực tế cho corpus này:

| Bạn đang tìm | Dùng | Đừng dùng |
|---|---|---|
| Một cảnh nhìn thấy được | `dense_visual` (mạnh nhất, luôn bật) | — |
| Từ khoá xuất hiện trong caption | `bm25_caption` | — |
| Thứ được NÓI ra | `bm25_asr` (nhớ: chỉ chạy nếu câu hỏi có cue "nói/phỏng vấn/phát biểu") | — |
| Tên vật thể cụ thể | `bm25_object` (mới sống) | — |
| Chữ trên màn hình | *(không có đường nào — OCR chết)* | `bm25_ocr`, `ocr_fuzzy`, preset "Tìm chữ" |
| Màu sắc | *(gần như vô dụng)* | đừng đặt trọng số cao cho `color_search` |
| Chuỗi sự kiện / hành động | *(không có đường nào — action/event chết)* | `bm25_action`, `event_search` |

`AIC_ENABLE_OCR_BRANCH=true`/`AIC_ENABLE_OCR_FUZZY=true` vẫn phải **giữ bật**
dù rỗng — tắt làm `/v1/search/capabilities` từ chối 422 mọi `search_options`
đã lưu có nhắc tới hai nhánh này (`docs/36` §9). Rỗng thì tốn 0ms, không hại gì.

---

## 4. Từng workspace trong UI — dùng để làm gì

Bốn workspace theo task (`online/ui-react/src/components/TaskWorkspaces.tsx`),
cộng các panel hỗ trợ dùng chung:

| Workspace | Việc nó làm | Khi nào mở |
|---|---|---|
| **KIS Safe Frame** | Mỗi dòng là `(video_id, frame_idx)` kèm `safe_frame_score` + `must_match_coverage` — điểm "frame này có an toàn để nộp không" | Trước khi nộp KIS, ưu tiên dòng safe_frame cao nếu điểm gần nhau |
| **QA Evidence Studio** | Bộ ba `(video, frame, answer)` **cộng verifier chạy độc lập** khỏi tool sinh câu trả lời | Luôn mở để xem verifier có đồng ý với answer không trước khi nộp |
| **TRAKE Alignment Studio** | 3 tầng: Stage A chọn video ứng viên, Stage B "Best Sequence" xếp ngang, Stage C timeline tinh chỉnh từng frame | **Bắt buộc mở** cho TRAKE — §1 cho thấy complete_chain_rate=0%, tức hệ hiếm khi tự đúng cả chuỗi, phải tinh chỉnh tay ở Stage C |
| **AVS Relevance/Diversity** | `relevance_grade` 0-3 mỗi kết quả + `cluster_id` (đã gom bớt trùng lặp bằng MMR) | P@100 chỉ 0.014 (§1) — lọc theo `relevance_grade≥2` trước khi chọn, đừng nộp nguyên danh sách |

Panel dùng chung, không theo task:

| Panel | Việc nó làm |
|---|---|
| **BranchStatusPanel** | Thấy trực tiếp nhánh nào `success/timeout/failed` — đừng đoán qua số kết quả tụt |
| **FrameTuner** | Tua **chính xác theo frame** trên toàn video (không chỉ trong cửa sổ scene). Dùng `fps` thật từ `GET /v1/videos` — **V003 chạy 25fps trong khi V001/V002 chạy 30fps**, hard-code 30 sẽ tua lệch 20% |
| **SubmissionBoard** | Bảng nộp cuối, đổi frame→giây bằng chính scene chứa nó (không giả định fps); dòng TRAKE bấm được từng bước để xem lại cả chuỗi trước khi nộp |
| **CompareLab** | So cấu hình + trạng thái nhánh + thời gian giữa 2 session (thường: gốc vs. replay) — không diff kết quả, đó là việc của Results Explorer |
| **EvidenceInspector** | `GET /v1/evidence/{id}` — mở lazy, request thật mỗi lần, không phải dữ liệu có sẵn |
| **StreamLog** | Log SSE thô — xác nhận stream đang chạy thật theo tiến độ backend, không phải progress bar giả |

---

## 5. Điểm mù đã biết — đừng tự tin quá vào những cái này

Từ `docs/33` §9 và `docs/27`, còn nguyên giá trị trên corpus mới:

1. **Không có so khớp cấu trúc.** *"người áo đỏ cầm chai"* và *"chai đỏ cạnh
   người áo trắng"* rã thành cùng một túi token — hệ không phân biệt được ai
   mang thuộc tính nào.
2. **BM25 không tách từ tiếng Việt đúng nghĩa.** Cắt theo `\w+`, tức theo âm
   tiết: `"xanh"` khớp luôn cả `"xanh dương"` lẫn `"xanh lá"`.
3. **`ocr_fuzzy` từng kéo sập `dense_visual`** (p50 8.5s làm dense_visual
   timeout 40/84 truy vấn ở corpus cũ). Đã né bằng
   `AIC_BRANCH_TIMEOUT_MS=30000`, nhưng nếu tự đổi số này xuống thấp hơn thì
   lỗi cũ có thể quay lại — **và nó im lặng**, không báo gì ngoài
   `branch_status=timeout`.
4. **Dịch VI→EN cho CLIP là một lời gọi LLM trên đường request.** Provider
   FPT hỏng → nhánh mạnh nhất mất theo. Không có cache nào cứu được truy vấn
   MỚI (tức mọi truy vấn lúc thi).
5. **VLM rerank đang tắt, và nên giữ tắt.** Đã đo: tốn 94% lời gọi API của KIS
   mà không đổi một chỉ số nào (`docs/27` D0). Đừng bật lại giữa lúc thi để
   "thử cho chắc" — vừa tốn quota vừa tốn thời gian, không có gì đổi.
6. **Container hạ cấp trong im lặng, không báo lỗi.** Nguyên tắc xuyên suốt hệ
   thống: HTTP 200 không có nghĩa là nhánh mạnh nhất đang chạy. Luôn nhìn
   `branch_status` (BranchStatusPanel) và `/v1/search/capabilities` trước khi
   tin một kết quả tụt hạng là do dữ liệu chứ không phải do một nhánh vừa chết.

---

## 6. TRAKE — vì sao yếu nhất, và cách bù bằng tay

`complete_chain_rate = 0.000` ở cả hai cấu hình rerank/không-rerank (§1).
Phân rã theo `docs/27` B1b (số cũ, corpus nhỏ, nhưng cơ chế vẫn đúng):

```
R-score ≈ P(chọn đúng video) × P(chọn đúng frame | đã có ứng viên đúng) × trần_oracle
```

`gold_video_missing = 0.75` ở corpus 873 video nghĩa là **3/4 truy vấn TRAKE
hệ chọn sai video ngay từ đầu** — không phải vấn đề chọn frame, mà vấn đề xếp
hạng video. `docs/22` đã đo và loại hai giả thuyết (ràng buộc hình thức,
cửa sổ chấm), giả thuyết sống duy nhất là **điểm ứng viên đang tương đối,
không tuyệt đối** — tức cần rerank tuyệt đối (đã bật, §2) trên **cả bước chọn
video** chứ không chỉ bước chọn frame trong video đã chọn.

**Thao tác tay khi thi:** đừng tin `video_recall@1`. Luôn mở TRAKE Alignment
Studio, kiểm video ứng viên hạng 2-3 (`video_recall@3` cao hơn hẳn hạng 1 —
đúng video thường NẰM TRONG top-3 dù không đứng đầu), rồi mới tinh chỉnh chuỗi
ở Stage C bằng FrameTuner.

---

## 7. Thứ tự tin cậy tóm tắt — dùng khi phải quyết nhanh lúc thi

```
Tin ngay, ít cần soát          KIS (R@20 = 0.944)
Soát nhanh trước khi nộp       QA (giữ rerank bật, đọc verifier)
Phải lọc bằng tay              AVS (relevance_grade ≥ 2, đừng nộp nguyên rổ)
Phải tự dựng lại bằng tay      TRAKE (kiểm cả top-3 video, tinh chỉnh Stage C)
```

Và trước khi thi, chạy đúng một lệnh để biết mình đang đứng ở đâu trên đường
kẻ này — không đoán:

```powershell
python -m scripts.eval_tasks --pipeline container `
    --gold examples/gold_all3.jsonl `
    --metadata storage/exports_competition/scenes.jsonl `
    --json-out outputs/eval_warmup.json
```

---

## 8. Đọc tiếp

| Tài liệu | Nội dung |
|---|---|
| `docs/33_RETRIEVAL_TECHNIQUES.md` | mỗi nhánh dùng kỹ thuật gì (so khớp từ vs ngữ nghĩa), điểm mù chi tiết |
| `docs/22_TRAKE_CHAIN_SCORING.md` | vì sao TRAKE yếu, ba giả thuyết đã loại |
| `docs/27_SYSTEM_ISSUES.md` | bảng cấu hình đã chốt + lý do từng dòng, số đo trên corpus nhỏ để đối chiếu |
| `docs/12_USER_GUIDE.md` | layout UI đầy đủ, cách gọi API trực tiếp, bảng sự cố thường gặp |
| `docs/34_COMPETITION_PACK_IMPORT.md` | pack thi đấu thiếu gì và vì sao (OCR/object/event) |
| `outputs/eval_all3_873video_rerank.json` | số thô đầy đủ, gồm `per_query` để soi từng truy vấn |
