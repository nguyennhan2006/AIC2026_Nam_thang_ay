# Kết quả đã hết hiệu lực — KHÔNG dùng để ra quyết định

Chuyển vào đây ngày 2026-08-06. Giữ lại thay vì xoá vì `docs/25`–`docs/27`
trích dẫn trực tiếp một số file, và xoá chúng là phá dấu vết kiểm chứng của
chính những kết luận đang dùng.

**Không con số nào trong thư mục này được dùng để so cấu hình nữa.** Dưới đây
là lý do theo từng nhóm.

---

## 1. Mọi chỉ số AVS — metric nDCG khi đó tính sai

`ideal` chỉ dựng từ gold interval trong khi `grades` chạy trên tới 100 kết
quả, mà một interval gold trải qua 15–64 scene. Tử số vì thế cộng trên nhiều
vị trí hơn mẫu số và nDCG **vượt 1** — tức không còn là quan hệ thứ tự, nên
không xếp hạng được cấu hình nào với cấu hình nào.

Đo được: `sig01a.json` có 14/24 truy vấn nDCG > 1 (max 2.357);
`p2/kw_false.json` có 2/24 (max 1.429). Những file KHÔNG vượt 1 cũng không
dùng được — mẫu số vẫn sai, chỉ là chưa đủ trùng lặp để lộ ra.

Xem `docs/26` §6. Đã sửa bằng `dedup_grades_by_event()`.

### Hệ quả nặng nhất: `avs_grade/`

Sáu file trong `avs_grade/` là thứ đã CHỌN `AIC_AVS_GRADE_MODE`:

```
hard_gate 0.299 · no_gate 0.453 · semantic_or_lexical 0.453 · soft 0.488
```

Chúng vừa dùng metric hỏng, vừa chỉ chạy trên **8 truy vấn** (1 truy vấn =
0.125). Lựa chọn `semantic_or_lexical` hiện tại vì thế không có cơ sở đo —
phải đo lại trên metric đã sửa.

## 2. Mọi chỉ số KIS trước 2026-08-06 08:00 — trước SIG-01a/01b

Hai lỗi trong `build_signature` (`PROPER_NOUN_RE` nuốt chữ thường có dấu;
`must_match` lấy ba từ đầu câu) làm KIS R@1 thấp hơn thật 0.083. Xem `docs/26`
§2–§3.

Gồm: `audit/`, `ocr/`, `multivideo/`, `keyword/`, `wiring/`, `split/`,
`best_config.json`, `p2/kw_true.json`.

## 3. `trake/` — tinh chỉnh trên 8 truy vấn của MỘT video

`T0`–`T3`, `W2`–`W5`, `S_*`, `sa_*`, `fair_*`, `stageA_baseline` đều chạy trên
8 truy vấn TRAKE của V001. Sau đó đo được `video_recall@1` = 1.000 trên V001
nhưng 0.375 và 0.250 trên V002/V003 — tức chúng khớp vào đặc thù một video.

Cùng kiểu hỏng với `WEIGHT-01` (xem `docs/26` §10): cấu hình tốt nhất trên
V001 mất 2 truy vấn khi đo trên hai video còn lại.

## 4. `qa_joint/` — 12 truy vấn, và QA có nhiễu LLM

Đo trên 12 truy vấn (1 truy vấn = 0.083) trong khi QA-REPRO-01 đo được nhiễu
LLM khoảng ±1 truy vấn giữa các lượt chạy giống hệt nhau. Không phân biệt được
gì ở quy mô đó.

---

## Những file CÒN hiệu lực (nằm ở thư mục cha)

| file | vai trò |
|---|---|
| `quick/avs_criteria01.json` | **baseline 4 task hiện tại** |
| `quick/new_base_cap20.json`, `quick/avs_cap3.json` | đối chứng chốt cap AVS = 20 |
| `quick/sig01b_kis.json` | bằng chứng SIG-01b (R@1 0.500 → 0.583) |
| `quick/kis_server_config.json` | bằng chứng confound nhánh (0.583 → 0.500 khi bật 2 nhánh) |
| `quick/qa_rep{1,2,3}.json` | bằng chứng QA-REPRO-01 (nhiễu LLM) |
| `p2/kw_false.json` | baseline P2 — `docs/25`+`26` trích dẫn trực tiếp |
| `kis_features.json`, `avs_candidates.json` | đầu vào cho hai harness replay offline |
| `quick/avs_variants.json` | ablation A–E chọn cách chấm tiêu chí AVS |

`p2/kw_false.json` giữ lại **dù chỉ số AVS trong đó sai** — nó là nguồn của
phân tích đẩy-lên/đẩy-xuống ở `docs/26` §1, và phần KIS/QA/TRAKE vẫn đúng.
