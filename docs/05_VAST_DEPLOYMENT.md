# 05. Triển khai Vast.ai

## Topology khuyến nghị

Một instance GPU chạy worker và có thể chạy Online/Qdrant trong compose. Public
duy nhất cổng 8000 của Online API; worker 8010 và Qdrant 6333 chỉ internal.
Gắn volume bền vững cho `storage`, Qdrant và model cache.

## Trình tự

1. Tạo instance có Docker, NVIDIA runtime và đủ disk.
2. Copy repo, `cp .env.example .env`.
3. Tạo hai token khác nhau: `AIC_GPU_API_KEY`, `AIC_ONLINE_API_KEY`.
4. Đặt `AIC_CORS_ORIGINS=http://IP_LOCAL:5173` hoặc origin thực.
5. Build/start compose.
6. Chạy Offline/export/index trước khi chuyển Online sang `qdrant`.
7. Kiểm tra `/v1/health`, query smoke, thumbnail và log.

## Mode an toàn khi chưa có index

Giữ `AIC_ONLINE_BACKEND=local` để kiểm tra Data/API/UI. Sau khi
`python -m offline index --qdrant` hoàn tất, đổi sang `qdrant` và restart backend.
Không để backend qdrant khởi động trước collection/vector cùng dimension.

## Network

Dùng TLS reverse proxy hoặc tunnel; không truyền token qua HTTP công cộng. Giới
hạn IP inbound nếu có thể. Không public dashboard Qdrant. Rotate token khi log
hoặc shell history có nguy cơ lộ.
