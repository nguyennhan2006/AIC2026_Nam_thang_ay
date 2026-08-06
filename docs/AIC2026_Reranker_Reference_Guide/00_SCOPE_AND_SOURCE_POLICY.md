# 00 — Phạm vi và chính sách nguồn

## 1. Ba mức độ tin cậy

### A. Được nguồn chính thức hỗ trợ trực tiếp

Ví dụ:

- Hugging Face Hub hỗ trợ `hf download`, `snapshot_download`, cache và `revision`.
- Model card Qwen3-Reranker cung cấp cách load bằng Sentence Transformers/Transformers.
- FlagEmbedding cung cấp inference, fine-tune reranker và hard-negative mining.
- Qwen3-VL-Embedding repository cung cấp môi trường, cách tải và inference multimodal reranker.
- Sentence Transformers cung cấp `CrossEncoder`, trainer, loss và evaluator.
- PEFT cung cấp LoRA để giảm số tham số cần cập nhật.
- Transformers cung cấp bitsandbytes cho 8-bit/4-bit.

### B. Cấu hình khởi đầu do dự án đề xuất

Các giá trị như batch size, max length, learning rate, số frame và tỷ lệ hard negative trong tài liệu này là **starting point**. Chúng chưa phải cấu hình tối ưu cho AIC.

### C. Chưa được khẳng định

- Không xem benchmark công khai là bằng chứng model sẽ tăng điểm AIC.
- Không xem model load thành công là bằng chứng fine-tune sẽ ổn.
- Không xem loss giảm là bằng chứng ranking tốt hơn.
- Không xem pseudo-label từ một teacher là gold.
- Không xem cấu hình quantized là tự động hợp lệ với luật giới hạn tham số.

## 2. Thứ tự ưu tiên nguồn

1. Model card/repository chính thức của model.
2. Tài liệu chính thức của Hugging Face, Sentence Transformers, PEFT, Accelerate.
3. Repository framework chính thức như FlagEmbedding.
4. Issue/discussion cộng đồng chỉ dùng để tham khảo lỗi thực tế, không dùng làm contract.
5. Blog bên thứ ba chỉ dùng khi tài liệu chính thức chưa đủ và phải ghi rõ.

## 3. Quy tắc đóng băng cấu hình

Sau khi smoke test thành công, ghi lại:

```yaml
model_id: Qwen/Qwen3-Reranker-0.6B
model_revision: "<commit_sha>"
python_version: "<x.y.z>"
torch_version: "<version>"
transformers_version: "<version>"
sentence_transformers_version: "<version>"
cuda_runtime: "<version>"
gpu_name: "<name>"
```

Không tiếp tục dùng `main` hoặc `latest` cho thí nghiệm so sánh chính thức.
