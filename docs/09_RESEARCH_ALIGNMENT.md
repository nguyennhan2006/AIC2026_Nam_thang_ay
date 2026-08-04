# 09. Liên hệ hai tài liệu nghiên cứu

## Sparse lexical image retrieval

V1 triển khai đúng ý tưởng thực dụng: caption/tags/OCR/object biến visual thành
text có thể lập chỉ mục BM25; các lexical field tách riêng để cân trọng số và
giải thích. Crop caption và query expansion là extension P1 vì tăng chi phí
Offline đáng kể.

## CoLLM / composed image retrieval

Composed retrieval cần reference image cộng modification text. Contract hiện
đã có keyframe/image identity và vector seam để bổ sung, nhưng V1 chưa expose
endpoint upload/reference-image. Đây là V1.1 hợp lý sau khi text→video baseline,
index consistency và benchmark ổn định.

## Nguyên tắc áp dụng

Không gắn một paper thành toàn bộ kiến trúc. Sparse BM25 là một retriever trong
hybrid system; dense model xử lý semantic/visual; RRF hợp nhất; temporal linker
xử lý chuỗi. Đánh giá ablation từng nhánh trước khi tăng độ phức tạp.
