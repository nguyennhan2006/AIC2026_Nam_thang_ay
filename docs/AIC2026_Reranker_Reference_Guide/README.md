# AIC 2026 — Reranker Experiment Reference Guide

## Mục đích

Bộ tài liệu này hỗ trợ:

1. Chuẩn bị môi trường riêng cho text reranker và multimodal reranker.
2. Tải model từ Hugging Face Hub theo cách có cache, resume và revision.
3. Chạy smoke test trước khi đưa model vào pipeline AIC.
4. Chuẩn hóa dữ liệu chưng cất nhãn pointwise/pairwise.
5. Chạy baseline, fine-tune và cascade reranking theo ma trận thí nghiệm.
6. Lưu manifest để có thể tái lập và so sánh kết quả.

Đây là **tài liệu tham khảo kỹ thuật**, tổng hợp theo tài liệu chính thức/model card/repository được cộng đồng sử dụng rộng rãi. Bộ tài liệu không khẳng định mọi lệnh sẽ chạy nguyên trạng trên mọi GPU, CUDA hoặc phiên bản thư viện.

## Nguyên tắc an toàn khi dùng

- Không cài hoặc nâng cấp thư viện giữa một phiên Python đang chạy.
- Tách môi trường text và vision-language.
- Smoke test model trước khi chạy toàn bộ candidate.
- Khóa revision model và phiên bản thư viện sau khi tìm được cấu hình chạy ổn.
- Không ghi đè gold/dev/test.
- Không dùng cùng video trong train và test.
- Không kết luận reranker tốt chỉ dựa vào loss hoặc accuracy nhị phân.
- Luôn đo metric downstream: KIS MRR/R@K, QA evidence hit, TRAKE event alignment.

## Lộ trình đọc nhanh

1. `00_SCOPE_AND_SOURCE_POLICY.md`
2. `01_ENVIRONMENT_SETUP.md`
3. `02_MODEL_DOWNLOAD_AND_CACHE.md`
4. `03_SMOKE_TESTS.md`
5. `04_LABEL_DISTILLATION_PIPELINE.md`
6. `05_TEXT_RERANKER_EXPERIMENTS.md`
7. `06_MULTIMODAL_RERANKER_EXPERIMENTS.md`
8. `07_EXPERIMENT_MATRIX_AND_EVALUATION.md`
9. `08_TROUBLESHOOTING.md`
10. `09_RUN_CHECKLIST.md`

## Model tham khảo chính

| Vai trò | Model ID | Ghi chú |
|---|---|---|
| Text reranker nhỏ | `Qwen/Qwen3-Reranker-0.6B` | Multilingual, instruction-aware, dùng làm zero-shot baseline hoặc student nghiên cứu |
| Text reranker dễ fine-tune | `BAAI/bge-reranker-v2-m3` | Có hệ sinh thái FlagEmbedding và hard-negative mining |
| Multimodal reranker | `Qwen/Qwen3-VL-Reranker-2B` | Nhận text, image, screenshot, video hoặc mixed input |
| Teacher lớn tùy chọn | Model VLM/reranker lớn hơn | Chỉ dùng tạo pseudo-label nếu phù hợp tài nguyên và luật cuộc thi |

## Cấu trúc output được khuyến nghị

```text
runs/
└── <experiment_id>/
    ├── manifest.json
    ├── config.yaml
    ├── environment.txt
    ├── raw_scores.jsonl
    ├── reranked_results.jsonl
    ├── metrics.json
    ├── metrics_by_group.csv
    ├── failures.jsonl
    └── checkpoints/
```
