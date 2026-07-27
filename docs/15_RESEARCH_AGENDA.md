# 15. Nghiên cứu cần thực hiện để đảm bảo chất lượng service

Nguyên tắc cứng (đã áp dụng xuyên suốt repo, xem `scripts/eval_kis.py` docstring):
**không kết luận phương án nào tốt hơn khi chưa đo Recall@1/5/20/50/100, MRR,
video-Recall@K trên dev set** qua `python -m scripts.eval_kis`. Danh sách dưới đây là
câu hỏi mở, mỗi câu có hypothesis + cách đo cụ thể + điều kiện chấp nhận — không có
câu nào được chốt "tốt hơn" chỉ bằng suy luận.

Điều kiện tiên quyết cho phần lớn câu hỏi ở đây: ground-truth hiện chỉ có 14 query
(`docs/13_PRODUCTION_READINESS_INFO.md` mục 4) — số đo trên tập này chỉ mang tính
smoke-test, không đủ để chốt production default. Mở rộng GT là việc research nền
đứng trước mọi câu hỏi khác.

## Retrieval — đo được ngay bằng eval_kis, không cần GPU thật

1. **OCR-fuzzy: song song hay thay thế `bm25_ocr`?**
   Hypothesis: `ocr_fuzzy` bắt được lỗi OCR/dấu mà BM25 exact miss, nhưng chạy song
   song có thể nhân đôi trọng số OCR trong RRF (đúng cảnh báo trong docstring
   `online/adapters/ocr_fuzzy.py`).
   Đo: `eval_kis --mode fusion` với `AIC_ENABLE_OCR_FUZZY` bật/tắt, so cả 2 cấu hình
   (a) `bm25_ocr` + `ocr_fuzzy` cùng chạy, (b) chỉ `ocr_fuzzy` thay `bm25_ocr`.
   Chấp nhận: giữ cấu hình có Recall@1/5 cao hơn trên GT mở rộng (≥50 query).

2. **Ngưỡng `min_score`/`fuzzy_token_ratio` của OcrFuzzyRetriever**
   Mặc định `min_score=0.35`, `fuzzy_token_ratio=0.8` — chưa tune trên dữ liệu thật.
   Đo: quét lưới nhỏ (`min_score` ∈ {0.25, 0.35, 0.45}, `fuzzy_token_ratio` ∈
   {0.7, 0.8, 0.9}) qua `OcrFuzzyRetriever.build(..., min_score=..., fuzzy_token_ratio=...)`
   trong một script ablation riêng, chấm bằng Recall@5.

3. **Bật `--use-expansion` (VI→EN lexicon) mặc định có lợi không?**
   Rủi ro nêu sẵn trong `online/services/query_expansion.py`: từ đơn âm tiết dễ đồng
   âm sau khi bỏ dấu → false positive lan rộng. Lexicon hiện ~70 cụm, seed thủ công.
   Đo: `eval_kis --mode metadata_only --use-expansion` so với không dùng; xem thêm
   `per_query` (cờ `--json`) để soi query nào bị nhiễu bởi expansion sai.
   Chấp nhận: chỉ bật mặc định nếu Recall@K tăng VÀ không query nào tụt rank vì match
   sai (soi thủ công top-3 nghi vấn).

4. **`PreparedQueryPlanner` (tách target/ocr/context) có tăng recall nhánh dense
   không?** Đặc biệt với query dài, nhiều mệnh đề (kiểu thi KIS thật).
   Đo: `eval_kis --mode fusion --use-query-prep` so baseline; chú ý tham số
   `include_ocr_in_dense` (mặc định True) — thử tắt xem có bớt nhiễu nhánh dense.

5. **Rule bonus/penalty (`online/services/rules.py`) — giá trị mặc định đúng thang
   chưa?** Bonus 0.003–0.02 được chọn "cùng thang" RRF (~0.016/rank-1) nhưng chưa
   verify thực nghiệm.
   Đo: `eval_kis --mode fusion --use-rules`, nếu Recall@1 không đổi hoặc giảm thì thử
   scale `RuleConfig` (nhân đôi/giảm nửa từng bonus) — không đoán, đo lại mỗi lần đổi.

6. **RRF `rrf_k=60` có tối ưu cho corpus này không?**
   Đo: quét `rrf_k` ∈ {20, 40, 60, 100} qua `--metadata`/build_service tương đương
   trong script ablation nhỏ (hoặc thêm `--rrf-k` vào `eval_kis` nếu cần lặp thường
   xuyên); k nhỏ ưu tiên rank cao của từng nhánh hơn, k lớn làm mượt fusion.

## Model/embedding — cần hạ tầng GPU thật, chưa đo được trên máy local

7. **SigLIP2 vs CLIP ViT-L/14 vs ensemble** — chất lượng dense retrieval thật (khác
   với `HashingTextEncoder` ở backend `local`, chỉ để smoke test theo cảnh báo trong
   `eval_kis.py`). Đo: `--backend qdrant` với từng encoder, cùng GT set, trên corpus
   enrich thật bằng từng checkpoint.

8. **Qwen3-VL-32B caption multi-frame vs Qwen2.5-VL-7B caption per-frame hiện tại** —
   chất lượng caption ảnh hưởng trực tiếp BM25 caption + rerank tầng 2. Đo: human-audit
   ≥90% đúng (tiêu chí Phase 1 doc 11) trên mẫu 200 scene, cộng so sánh Recall@K khi
   dùng caption mới làm index.

9. **Ngưỡng quality-gate keyframe** (duplicate>0.98, blur) — cần calibrate trên mẫu
   human-audit 200 frame/đợt (đã nêu ở §4.C11 doc 11), chưa có số cụ thể.

10. **Rerank rubric top-300→50 (BGE) / top-20 (Qwen3-VL)** — trần số lượng ảnh hưởng
    trực tiếp p95 latency lúc thi. Đo: quét trần (vd 20/50/100 → 50) đo cả Recall@K
    lẫn p95 qua `scripts/load_test.py` (nếu có) trước khi chốt.

11. **Elasticsearch analyzer vi (ICU folding) vs BM25 in-memory hiện tại** — chỉ có ý
    nghĩa khi corpus đủ lớn (nêu ở §6 doc 11: "corpus thi đấu lớn mới phát huy ES").
    Đo: chạy song song cả hai trên corpus enrich thật, so eval_kis trước khi cắt hẳn
    BM25 in-memory.

## Clip embedding density — Search Mixing Console W1 (clip pooling)

12. **Clip embedding density**: baseline V1 (`offline/clip_pooling.py`) pool embedding
    của keyframe **có sẵn** trong scene (không extract frame mới); với
    `AIC_KEYFRAMES_PER_SCENE=1` mặc định, phần lớn clip suy biến còn 1 frame — pooling
    không tạo thêm tín hiệu temporal nào so với chính scene đó. Cần ablation trước khi
    coi baseline này là đủ cho `dense_visual_clip`/action search:
    - So sánh 1/3/5 keyframe/scene (`AIC_KEYFRAMES_PER_SCENE`) × baseline pooling hiện
      tại, đo Recall@K cho action-query riêng (câu hỏi có động từ/hành động) so với
      câu hỏi tĩnh, cộng Recall@K chung của `dense_visual_clip`.
    - So sánh với dedicated clip frame sampling (extract thêm frame trong từng clip
      window thay vì tái dùng keyframe scene) — cùng metric ở trên.
    - Đo thêm chi phí: thời gian offline processing, số lần gọi encoder, kích thước
      index — vì dedicated sampling sẽ tăng cả ba so với baseline tái dùng keyframe.
    Chỉ chuyển sang dedicated clip sampling khi ablation cho thấy Recall@K tăng đủ để
    bù chi phí đo được ở trên — không đoán, đo bằng `eval_kis.py`.

## Cách ghi lại kết quả

Mọi lần chạy ablation nên dùng `--json <path>` của `eval_kis.py` để lưu vết (mode,
flags, per-query rank) — tránh kết luận từ trí nhớ. Khi một câu hỏi ở trên có kết
quả, cập nhật ngay dòng tương ứng trong doc này (thắng/thua + số liệu + ngày đo) thay
vì để lại "chưa biết" mãi.
