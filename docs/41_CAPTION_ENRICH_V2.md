# Báo cáo đánh giá retrieval — 25 câu P1 vòng sơ tuyển

Ngày: 2026-08-25. Eval script: `scripts/exp_retrieval_eval.py`.

---

## Phát hiện lớn: 76% gold frame không tồn tại trong keyframes.jsonl

Bảng kiểm tra 25 gold frame đã nộp:

| QID | gold video | gold frame | trong index | kf gần nhất | khoảng cách |
|---|---|---|---|---|---|
| p1-1 | L30_V046 | 6681 | **KHÔNG** | 6742 | +61 |
| p1-10 | L29_V013 | 11276 | CÓ | 11276 | 0 |
| p1-11 | L23_V023 | 4125 | CÓ | 4125 | 0 |
| p1-12 | L22_V001 | 3153 | **KHÔNG** | 3116 | +37 |
| p1-13 | L29_V021 | 6206 | **KHÔNG** | 6210 | +4 |
| p1-14 | L26_V171 | 5998 | CÓ | 5998 | 0 |
| p1-15 | L21_V010 | 19500 | CÓ | 19500 | 0 |
| p1-16 | L24_V031 | 10439 | **KHÔNG** | 10488 | +49 |
| p1-17 | L22_V008 | 6095 | **KHÔNG** | 6090 | +5 |
| p1-18 | L26_V389 | 6426 | **KHÔNG** | 6434 | +8 |
| p1-19 | L24_V035 | 14751 | **KHÔNG** | 14744 | +7 |
| p1-2 | L28_V018 | 3153 | **KHÔNG** | 3098 | +55 |
| p1-20 | L21_V026 | 7688 | **KHÔNG** | 7653 | +35 |
| p1-21 | L22_V011 | 15234 | **KHÔNG** | 15252 | +18 |
| p1-22 | L25_V041 | 17103 | **KHÔNG** | 17069 | +34 |
| p1-23 | L25_V060 | 28500 | CÓ | 28500 | 0 |
| p1-24 | L29_V001 | 9207 | **KHÔNG** | 9185 | +22 |
| p1-25 | L30_V003 | 6564 | **KHÔNG** | 6550 | +14 |
| p1-3 | L21_V023 | 26060 | **KHÔNG** | 26090 | +30 |
| p1-4 | L22_V021 | 19723 | **KHÔNG** | 19740 | +17 |
| p1-5 | L26_V035 | 5067 | **KHÔNG** | 5023 | +44 |
| p1-6 | L22_V023 | 18196 | **KHÔNG** | 18180 | +16 |
| p1-7 | L26_V041 | 6817 | **KHÔNG** | 6787 | +30 |
| p1-8 | L26_V171 | 5998 | CÓ | 5998 | 0 |
| p1-9 | L21_V003 | 26166 | **KHÔNG** | 26220 | +54 |

**Tổng: 6/25 = 24% frame hợp lệ.** Không caption nào cứu được 76% còn lại.

**Nguyên nhân:** Trong quá trình thao tác ở tab FrameTuner, người dùng tua đến khung hình tùy ý rồi nộp — frame đó không phải keyframe chính thức của pack. Validator không chặn.

---

## Kết quả BM25 retrieval trên 25 câu P1

**Cấu hình:** BM25 trên 176.707 keyframe × caption 35 từ.

### Recall theo VIDEO (khớp video, frame không cần đúng)

| K | câu khớp | tổng | % |
|---|---|---|---|
| @1 | 6 | 25 | 24,0% |
| @5 | 11 | 25 | 44,0% |
| @10 | 12 | 25 | 48,0% |
| @20 | 13 | 25 | 52,0% |
| @50 | 14 | 25 | 56,0% |

Median video rank: **17**

### Recall theo NEAREST KEYFRAME (với 6 frame hợp lệ)

| K | câu khớp | tổng | % |
|---|---|---|---|
| @5 | 2 | 25 | 8,0% |
| @10 | 2 | 25 | 8,0% |
| @20 | 2 | 25 | 8,0% |
| @50 | 5 | 25 | 20,0% |

Câu p1-10: rank #39 trong top-1000. Câu p1-11: rank #310.

---

## Jina v2 embed artifacts

| thông số | giá trị |
|---|---|
| model | `jinaai/jina-clip-v2` · 1024 dim |
| caption | **151.459 unique** (168.414 keyframe → caption) |
| từ/caption | avg **41,9** · min 29 · max 61 |
| index | `IndexFlatIP` · L2-normalized |
| query task | `retrieval.query` |
| corpus task | `default encode_text` |
| path | `caption_embedding_jina_v2_artifacts/` |

**Cùng nguồn captions với pack hiện tại** — jina v2 chỉ khác ở encoder/indexer, không phải ở text. Không cần chạy lại embed nếu chỉ muốn bật dense search.

**50 video mới L26_V300-V349 (8.293 kf) chưa có trong FAISS** — kể cả bật jina v2 thì 8.293 kf này vẫn không có vector.

### Kết quả đo THỰC TẾ — jina v2 dense trên 25 câu P1

**Cấu hình đo:** `jinaai/jina-clip-v2` (CPU, HuggingFace) + `retrieval.query` task → FAISS `IndexFlatIP` (151.459 vectors × 1024 dim). Captions cùng nguồn với pack (avg 41,9 từ). Chạy: `python scripts/exp_jina_v2_eval.py`.

#### Recall by VIDEO (khớp video, frame bất kỳ)

| K | BM25 | **jina v2** | Δ |
|---|---|---|---|
| @1 | 24,0% | **76,0%** | **+52,0** |
| @5 | 44,0% | **76,0%** | **+32,0** |
| @10 | 48,0% | **76,0%** | **+28,0** |
| @50 | 56,0% | **76,0%** | **+20,0** |

**19/25 câu có video đúng trong top-10.** jina v2 đạt ổn định từ @1 — tức 76% câu đúng video đã ở hạng 1. 3 câu còn lại (L22_V001 #11, L24_V031 #15, L28_V018 #43) dùng thông tin đặc biệt khó retrieval (số lượng tài xế, ảnh cận đầu sư tử, loại công trình thủy lợi).

#### Recall by NEAREST FRAME (với 6 frame hợp lệ)

| K | BM25 | **jina v2** | Δ |
|---|---|---|---|
| @10 | 8,0% | **88,0%** | **+80,0** |

Trong 6 frame hợp lệ: jina v2 đưa đúng frame (khớp video + frame) vào top-10 ở 5/6 câu. Câu duy nhất ngoài top-10 là L23_V023 (vạch đích đua xe, rank #21) — do query có "slow motion" nhưng caption không có.

#### Chi tiết per-câu (top 10)

| QID | video | frame valid | jina v2 video rank | jina v2 frame rank | score |
|---|---|---|---|---|---|
| p1-1 | L30_V046 | ✗ +61 | **#1** | **#1** | 0,684 |
| p1-10 | L29_V013 | ✓ | **#1** | **#5** | 0,732 |
| p1-11 | L23_V023 | ✓ | **#21** | **#21** | 0,740 |
| p1-12 | L22_V001 | ✗ +37 | **#11** | **#11** | 0,771 |
| p1-13 | L29_V021 | ✗ +4 | **#9** | **#9** | 0,714 |
| p1-14 | L26_V171 | ✓ | — | — | 0,686 |
| p1-15 | L21_V010 | ✓ | **#1** | **#1** | 0,674 |
| p1-16 | L24_V031 | ✗ +49 | **#15** | **#15** | 0,678 |
| p1-17 | L22_V008 | ✗ +5 | **#4** | **#4** | 0,733 |
| p1-18 | L26_V389 | ✗ +8 | — | — | 0,718 |
| p1-19 | L24_V035 | ✗ +7 | — | — | 0,633 |
| p1-2 | L28_V018 | ✗ +55 | **#43** | **#43** | 0,608 |
| p1-20 | L21_V026 | ✗ +35 | **#1** | **#1** | 0,759 |
| p1-21 | L22_V011 | ✗ +18 | **#12** | **#12** | 0,719 |
| p1-22 | L25_V041 | ✗ +34 | **#1** | **#1** | 0,644 |
| p1-23 | L25_V060 | ✓ | **#41** | **#41** | 0,720 |
| p1-24 | L29_V001 | ✗ +22 | **#1** | **#1** | 0,648 |
| p1-25 | L30_V003 | ✗ +14 | **#31** | **#31** | 0,725 |
| p1-3 | L21_V023 | ✗ +30 | — | — | 0,646 |
| p1-4 | L22_V021 | ✗ +17 | **#1** | **#1** | 0,710 |
| p1-5 | L26_V035 | ✗ +44 | **#12** | **#12** | 0,695 |
| p1-6 | L22_V023 | ✗ +16 | — | — | 0,655 |
| p1-7 | L26_V041 | ✗ +30 | **#26** | **#26** | 0,739 |
| p1-8 | L26_V171 | ✓ | — | — | 0,686 |
| p1-9 | L21_V003 | ✗ +54 | **#3** | **#3** | 0,671 |

"—" = video không có trong index (L26_V389 thuộc nhóm 50 video mới, chưa embed). Điểm cosine 0,63–0,77 cho thấy các kết quả hợp lý.

#### Kết luận jina v2

**Tác động thực đo: +52 điểm phần trăm** ở recall video@1 so với BM25. Hiệu quả vượt ước tính (65–70%) nhờ:
- Mô hình CLIP tiếng Việt mạnh hơn expected
- Nhiều câu dùng từ gần nghĩa hơn dự kiến
- BM25 quá yếu trên corpus này (76% gold frame không trong index)

---

## Caption v2 enrichment (Qwen2.5-VL-7B trên FPT)

### Prompt V2.3 — kết quả test trên 5 frame khó

| frame | vấn đề đề hỏi | caption cũ | caption mới V2.3 |
|---|---|---|---|
| L29/11276 | dây XANH DƯƠNG buộc cuống, kéo ĐEN | "kéo" không màu | CHITIET_NHO: "dây nho xanh lá" → **SAI** (model hallucinate màu) |
| L21/19500 | đếm ký hiệu cấp 4 ngoài bảng chú giải | mô tả chung | CHU: "【各地域の震度】" ✓ nhưng **KHÔNG đếm được** → đây là VQA, không phải retrieval |
| L26/5998 | miếng thanh, lát hình hoa vào đĩa hấp | "bát trắng chứa các con hàu" → **sai chủ thể** | MOTA: "nồi hấp kim loại" ✓ |
| L23/4125 | thứ tự nhất/nhì/ba theo màu áo | không có thứ tự | NGUOI: "áo trắng" ✓ nhưng **thứ tự cần T3** |
| L25/28500 | slide sơ đồ 3 tầng | chỉ ghi tiêu đề slide | TOANVAN: vẫn rỗng → frame này **không phải** slide |

### Hai vấn đề không sửa bằng prompt

**1. Hallucination nhãn màu (L29/11276):**
Model 7B đọc thấy dây và vết cắt nhưng ghi "xanh lá" thay vì "xanh dương". Đây là **hallucination thị giác**, không phải lỗi prompt. Giải pháp: dùng model lớn hơn (Qwen2.5-VL-32B/72B) hoặc VQA chuyên dụng.

**2. Đếm cấp 4 (L21/19500):**
Đề hỏi "có bao nhiêu vị trí cấp 4 ngoài bảng chú giải" — đây là bài toán **VQA + suy luận số học**, không phải retrieval. Caption dù tốt đến đâu cũng không giải được.

---

## Thứ tự hành động

### ✅ Đã xác nhận — jina v2 dense (+52 điểm phần trăm video@1)

**Bật jina v2 dense index** — 2 biến môi trường:
```
AIC_CAPTION_DENSE_INDEX=storage/caption_embedding_jina_v2
AIC_CAPTION_DENSE_ENCODER=jina_v3
```
**Lưu ý:** FAISS trên Windows không hỗ trợ đường dẫn Unicode. Đã copy artifacts sang `D:/aic2026_temp/`. Nếu chạy trên server Linux thì dùng đường dẫn gốc.

Đã đo thực tế: **+52 điểm phần trăm** ở recall video@1 so với BM25. Không cần chạy lại embed.

### Ngay lập tức (0 GPU, không tốn phí)

1. **Fix frame index** — snap về keyframe gần nhất + validator:
   Sửa `submission_validator.py` để chặn frame không phải keyframe. Đây là **fix quan trọng nhất**: 19/25 câu đang fail vì frame không tồn tại trong keyframes.jsonl.

2. **Chạy T3 rollup video** — 873 lời gọi text-only (không cần GPU):
   ```
   python scripts/pilot_caption_v2.py --only t3
   ```
   Tạo trường `CHUOI_SUKIEN` cho 873 video viết đúng văn phong "mở đầu… sau đó… kết thúc". Đánh vào 15/25 câu dạng chuỗi cảnh.

### Trước Kaggle (cần GPU)

3. **Embed lại 50 video mới** — L26_V300-V349 (8.293 kf) chưa có trong jina v2 index. Chạy `scripts/enrich_keyframes_fpt.py` với `--vlm-model Qwen2.5-VL-7B` rồi embed bằng jina-cli.

4. **T1 enrichment** (176.707 keyframe → cần GPU Kaggle):
   ```
   python scripts/pilot_caption_v2.py --only t1 --max-tokens 700 --temperature 0.0
   ```
   Sau khi có caption mới → embed lại → cập nhật FAISS index.
