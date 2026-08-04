# Tài liệu tham khảo và bài học thiết kế

## 1. MEMORIA tại LSC 2025

Bài học áp dụng:

- Visual embedding + vector DB là nhánh cốt lõi.
- Textual annotations vẫn quan trọng để filter/explain.
- Query parser nên cho người dùng bật/tắt entity.
- Main-topic extraction không phải lúc nào cũng tốt hơn raw query.
- Event retrieval và previous/next event giúp điều hướng nhanh.
- Image-to-image trực tiếp tốt hơn chỉ caption ảnh upload.
- Nút chọn/gửi nhiều kết quả là chức năng thi đấu thực tế.

## 2. Sparse lexical representation với M-LLM

Bài học:

- Chuyển ảnh thành tags/captions rồi BM25 là nhánh bổ sung mạnh cho keyword/exact terms.
- Iterative keyword refinement nên được hỗ trợ trong UI.
- Fixed crop caption có thể tăng lexical coverage, nhưng hiệu quả bão hòa và phải đo trên dữ liệu AIC.
- Sparse không thay dense trong mọi query; dùng hybrid.

## 3. CoLLM / composed image retrieval

Bài học:

- Image + modification text cần hiểu compositional, không chỉ cộng score đơn giản.
- Hữu ích cho “giống ảnh này nhưng đổi hành động/bối cảnh”.
- Synthetic triplets và data quality có thể quan trọng hơn chỉ tăng model size.
- Benchmark ambiguity/hard negatives cần được xử lý.
- Đây là nhánh P2, không chặn baseline text-first.

## 4. Quy tắc chuyển từ paper sang production

1. Không sao chép solution nguyên xi khi task/data khác.
2. Xác định module-level hypothesis.
3. Tạo ablation trên dev set AIC.
4. Đo metric cuối và latency.
5. Chỉ bật khi lợi ích rõ, có fallback và UI support.
