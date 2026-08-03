# 17. Hướng dẫn tự kiểm tra thủ công — L21_V001 (PR-12/13/14A)

Ghi lại đúng trạng thái tại commit `6ce6584` (branch `server_implementation`)
để bạn tự chạy lại và test UI mà không cần dò lại từ đầu.

## 0. Trạng thái tóm tắt (đọc trước khi test)

Đã có, đã đo bằng dữ liệu THẬT (L21_V001 — video/scene/keyframe/ASR thật, caption/
OCR/object từ FPT VLM, embedding từ CLIP ViT-L/14 local):

| Task  | Đã dùng được? | Ghi chú |
|---|---|---|
| KIS   | Có | R@1 ~0.33-0.42, MRR ~0.5-0.6 |
| QA    | Có (retrieval), yếu (answer) | evidence R@100 ~0.92, answer_accuracy ~0.33 |
| TRAKE | Có (Gate A), chưa (Gate B) | 8/8 đúng video, sequence luôn khác rỗng; r_score vẫn thấp — xem §5 |
| AVS   | Chưa | phần lớn query trả `result_count=0` — chưa sửa (để dành PR riêng) |

**Đừng hoảng nếu**: AVS gần như luôn rỗng, TRAKE `mean_r_score` thấp/0, 2/307
frame FPT enrichment lỗi JSON, câu lệnh **đầu tiên** gọi CLIP trong một tiến
trình mới luôn chậm hơn hẳn (~4s, các câu sau ~30-40ms) — tất cả đều đã biết
nguyên nhân, không phải bug mới.

---

## 1. Chuẩn bị môi trường

```bash
cd "d:/Sinh viên CNhan/AIC/Data/AIC2026_Nam_thang_ay"

# Các biến BẮT BUỘC để trỏ vào dữ liệu L21_V001 thật thay vì fixture demo nhỏ:
export AIC_METADATA_JSONL=storage/exports_l21/scenes.jsonl
export AIC_DATA_ROOT=storage
export AIC_VISUAL_EMBEDDING_MODEL=storage/models/clip-vit-large-patch14
```

Trên PowerShell dùng `$env:TÊN = "giá trị"` thay vì `export`.

Nếu `storage/exports_l21/` hoặc `storage/models/clip-vit-large-patch14/` không
còn tồn tại (đã bị dọn/máy khác), dựng lại bằng:

```bash
python -m scripts.build_l21_stage_packs          # video/scene/asr/keyframe pack
python -m scripts.embed_keyframes_local --model-path storage/models/clip-vit-large-patch14
python -m scripts.enrich_keyframes_fpt --env-file .env.fpt.local   # cần AIC_FPT_API_KEY thật
python -m offline assemble --packs storage/packs --dataset-id l21-v001-real
# AIC_EXPORT_DIR mặc định storage/exports — đổi thành storage/exports_l21 nếu muốn giữ tách biệt:
AIC_EXPORT_DIR=storage/exports_l21 python -m offline assemble --packs storage/packs --dataset-id l21-v001-real
python -m offline index --frames --scenes storage/exports_l21/scenes.jsonl --output storage/indexes_l21/frames
```

`storage/models/clip-vit-large-patch14/` tải bằng `curl` (không phải qua
Python/`transformers.from_pretrained`) vì máy này có lỗi SSL "Basic Constraints
of CA cert not marked critical" khi Python gọi HTTPS — `online/adapters/
fpt_client.py` đã tự vá lỗi này (bỏ cờ `VERIFY_X509_STRICT`), nhưng việc TẢI
model CLIP ban đầu bằng `transformers` vẫn cần né bằng `curl` nếu chạy trên
đúng máy có lỗi này. Xem lệnh tải trong lịch sử hội thoại nếu cần làm lại.

---

## 2. Chạy test tự động trước

```bash
python -m pytest tests/ -q
```

Kỳ vọng: `442 passed`. Nếu ít hơn, có regression — dừng lại kiểm tra trước khi
test tay.

Kiểm tra không lộ secret trước khi commit gì:

```bash
python -m scripts.check_secret_leak
```

---

## 3. Đo lại số liệu (không cần UI/server)

```bash
python -m scripts.eval_tasks \
  --metadata storage/exports_l21/scenes.jsonl \
  --tasks all --use-query-prep --verbose
```

`--use-query-prep` bật `PreparedQueryPlanner` (tách target/ocr/context cho
KIS/QA/AVS) — không bắt buộc cho TRAKE (TRAKE tự tách step đánh số `(1)...(2)...`
ngay trong `RuleBasedQueryPlanner` mặc định, chạy được kể cả không có cờ này).

So sánh với kết quả đã lưu để biết máy/dữ liệu của bạn khớp: `outputs/
evaluation/pr14a_trake_gates/eval_full.txt` (bản mới nhất) hoặc `outputs/
evaluation/pr13_fpt_enriched/eval_full.txt` (bản trước khi sửa TRAKE).

Số dao động nhẹ giữa các lần chạy là bình thường (xem §5 — CLIP cold-start).

---

## 4. Khởi động backend

```bash
python -m uvicorn online.api.app:app --reload --port 8000
```

Kiểm tra nhanh:

```bash
curl -s http://127.0.0.1:8000/v1/health | python -m json.tool
```

Kỳ vọng thấy `"status": "ok"`, `dataset_version` khác `null` (build_id của
lần `offline assemble` gần nhất).

Nếu bạn KHÔNG set `AIC_METADATA_JSONL` như §1 trước khi chạy uvicorn, server
sẽ tự nạp fixture demo nhỏ (`storage/exports/scenes.jsonl`, 3 scene) — vẫn
chạy được nhưng không có gì để test thật.

---

## 5. Test qua API trực tiếp (curl)

### KIS

```bash
curl -s -X POST http://127.0.0.1:8000/v1/search/kis \
  -H "Content-Type: application/json" \
  -d '{"query": "người dẫn chương trình đứng trước bản đồ thời tiết", "top_k": 10, "debug": true}' \
  | python -m json.tool
```

Xem `results[].scene_id`, `results[].best_frame_idx`, `component_scores`
(cần `debug: true`).

### QA

```bash
curl -s -X POST http://127.0.0.1:8000/v1/search/qa \
  -H "Content-Type: application/json" \
  -d '{"query": "có bao nhiêu người dẫn chương trình xuất hiện trong đoạn tin?", "top_k": 10}' \
  | python -m json.tool
```

Xem `response.qa[].answer`, `.evidence_refs`.

### TRAKE

```bash
curl -s -X POST http://127.0.0.1:8000/v1/search/trake \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tìm video về giếng nước bất ngờ phun cao và căn chỉnh bốn khoảnh khắc: (1) cột nước được quay từ xa; (2) một người đàn ông tiến sát cột nước; (3) người này cầm chai hoặc vật chứa cạnh dòng nước; (4) nhiều người cùng chỉ về phía cột nước.",
    "top_k": 20
  }' | python -m json.tool
```

Xem `response.trake[0].frame_ids` (phải tăng dần), `.missing_steps` (rỗng =
đủ 4 step), `.degraded` (true = video được giữ dù dưới ngưỡng coverage —
hiếm khi xảy ra với video độc nhất L21_V001).

**Query PHẢI dùng đúng format `(1) ...; (2) ...`** — câu văn xuôi kiểu "A, sau
đó B, cuối cùng C" cũng được nhận (marker "sau đó"/"cuối cùng"/"rồi") nhưng
gold thật của bộ L21_V001 luôn dùng format đánh số.

### AVS

```bash
curl -s -X POST http://127.0.0.1:8000/v1/search/avs \
  -H "Content-Type: application/json" \
  -d '{"query": "Tìm các đoạn dưới nước có thợ lặn, cá lớn hoặc rùa biển.", "top_k": 20}' \
  | python -m json.tool
```

**Kỳ vọng thực tế**: `response.avs` rất có thể rỗng hoặc chỉ 0-1 kết quả — đây
là hạn chế đã biết (criteria matching kiểu từ khoá cứng trong `online/services/
avs.py`, xem báo cáo trước), không phải lỗi cấu hình của bạn.

### Bộ gold query đầy đủ (để thử nhiều câu cùng lúc)

`examples/AIC2026_L21_V001_queries_4tasks.jsonl` — mỗi dòng một query, có
`task`/`query_vi`/`query_en`/gold answer. Dùng để copy query mẫu thay vì tự
nghĩ câu hỏi.

---

## 6. Khởi động UI React

```bash
cd online/ui-react
npm install   # chỉ cần lần đầu
npm run dev
```

Mở `http://localhost:5173`. Vào tab **Health** (hoặc HealthDrawer) trước,
nhập API base `http://localhost:8000` (mặc định đã đúng) — xác nhận health
xanh trước khi thử các tab khác.

CORS đã mở sẵn cho `localhost:5173`/`127.0.0.1:5173` trong `.env.example`
(`AIC_CORS_ORIGINS`) — nếu bạn đổi port Vite, nhớ cập nhật biến này và khởi
động lại backend.

### Các tab nên thử theo thứ tự

1. **Query Studio** — nhập query, chọn task, `top_k`, bật `debug` — xem có
   chạy được không, thời gian phản hồi, có warning gì không (bản thân
   `warnings` trong response là chỗ lộ rõ nhất các hạn chế đã biết ở §0).
2. **Results Explorer** — kiểm tra hiển thị kết quả, ảnh keyframe có load
   được không (`image_path` trỏ vào `storage/processed/keyframes/L21_V001/`
   — endpoint `/v1/media/{path}` phải serve được file thật, thử mở trực tiếp
   `http://localhost:8000/v1/media/processed/keyframes/L21_V001/frame_000150.jpg`
   trên trình duyệt nếu ảnh không hiện).
3. **KIS / QA / TRAKE workspaces** — mỗi tab có UI chuyên biệt cho task đó,
   test đúng loại câu hỏi tương ứng.
4. **Evidence Inspector** — bấm vào một candidate_id để xem `/v1/evidence/
   {candidate_id}` — caption/OCR/object thật từ FPT có hiển thị đúng không.
5. **Mixing Console** — chỉnh trọng số nhánh (dense/BM25/OCR...), xem kết quả
   đổi theo thời gian thực có hợp lý không.
6. **Stream Log** — bật streaming, xem các sự kiện SSE (`branch_started`,
   `branch_completed`, `fusion_completed`...) có tới đúng thứ tự không.
7. **Submission Board** — thử build/validate submission từ kết quả tìm được.
8. **Compare Lab** — so 2 lần search cạnh nhau (vd bật/tắt `use_query_prep`
   qua Mixing Console nếu UI có expose, hoặc so 2 query khác nhau).

### Những điểm UI nên soi kỹ (theo đúng yêu cầu "đã thuận tiện dùng chưa")

- Trạng thái loading/error có rõ ràng không khi backend chậm (CLIP cold-start
  lần gọi đầu ~4s) hay khi một branch timeout.
- `warnings` từ response có được hiển thị dễ thấy không, hay bị ẩn trong debug
  panel không ai bấm vào.
- Ảnh keyframe 1280×720 thật (không phải placeholder SVG như bộ demo cũ) có
  hiển thị đúng tỉ lệ, không bị vỡ layout không.
- TRAKE workspace có hiển thị được `missing_steps`/`degraded` (field mới thêm
  ở PR-14A) hay UI cần cập nhật thêm để show hai field này? (Nếu UI chưa có
  chỗ hiển thị, đây là việc cần làm thêm — không phải lỗi.)

---

## 7. Dọn dẹp sau khi test (nếu cần)

Các thư mục sau **không nằm trong git** (đã `.gitignore`), xoá thoải mái nếu
muốn dựng lại từ đầu để kiểm tra tính lặp lại được:

```
storage/packs/            (stage pack thô)
storage/exports_l21/      (canonical export)
storage/indexes_l21/      (FAISS frame index)
storage/models/           (weights CLIP)
storage/cache/            (cache enrichment FPT)
storage/processed/keyframes/L21_V001/
storage/processed/embeddings/
storage/processed/clip_embeddings/
```

`.env.fpt.local` (chứa `AIC_FPT_API_KEY` thật) cũng không nằm trong git —
đừng xoá nếu vẫn cần gọi FPT.

---

## 8. Nếu muốn test nhanh không cần dữ liệu thật

```bash
python -m scripts.seed_demo
AIC_METADATA_JSONL=storage/exports/scenes.jsonl python -m uvicorn online.api.app:app --reload
```

Bộ demo 3 scene này luôn dùng `lexical_hash_fallback` (không CLIP, không FPT)
— chỉ hợp để kiểm tra UI/API còn hoạt động cơ bản, không phản ánh chất lượng
tìm kiếm thật.
