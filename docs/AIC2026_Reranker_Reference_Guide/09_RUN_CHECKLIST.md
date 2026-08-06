# 09 — Checklist chạy thí nghiệm

## Trước khi tải

- [ ] Chọn environment text hoặc VL.
- [ ] Xác định model ID.
- [ ] Xác định cache path bền vững.
- [ ] Kiểm tra dung lượng ổ đĩa.
- [ ] Không để token trong code.

## Sau khi tải

- [ ] Có config/tokenizer/processor/weights.
- [ ] Ghi model revision.
- [ ] Ghi local path.
- [ ] Chạy offline load.
- [ ] Chạy positive-vs-negative smoke test.
- [ ] Ghi latency và peak VRAM.

## Trước khi tạo nhãn

- [ ] Candidate pool là union nhiều nhánh.
- [ ] Không đưa raw rank vào teacher prompt.
- [ ] Gold có quyền ưu tiên cao hơn teacher.
- [ ] Evidence ID tồn tại.
- [ ] Có rejected output.
- [ ] Split theo video.

## Trước khi fine-tune

- [ ] Train/dev/test bất giao theo video.
- [ ] Data schema valid.
- [ ] Có label statistics.
- [ ] Có hard-negative subtype.
- [ ] Có baseline zero-shot.
- [ ] Có run manifest.
- [ ] Có seed.

## Trong khi train

- [ ] Theo dõi train loss và dev ranking metric.
- [ ] Lưu best checkpoint theo downstream/dev metric hợp lý.
- [ ] Không chọn checkpoint chỉ bằng train loss.
- [ ] Ghi package versions.
- [ ] Ghi GPU và runtime.

## Sau khi train

- [ ] So E0/E1/E2/E3 trên cùng candidate pool.
- [ ] Kiểm tra per-query ranks.
- [ ] Kiểm tra nhóm OCR/ASR/temporal.
- [ ] Kiểm tra false-positive mới.
- [ ] Đo latency p50/p95.
- [ ] Chạy ít nhất một lần với offline mode.
- [ ] Đóng băng revision và requirements.

## Trước khi đưa vào hệ thống

- [ ] Model không làm giảm candidate recall do truncate/filter.
- [ ] Output map ngược được về video/frame.
- [ ] Có fallback khi reranker lỗi.
- [ ] Có timeout và batch limit.
- [ ] Có model version trong result provenance.
- [ ] Có thể tắt reranker bằng config.
