# 04. Online retrieval

## Search layers

1. Query planner tách sequence và tăng trọng số OCR khi có cụm từ chính xác.
2. Dense visual/text search Qdrant hoặc local vector baseline.
3. Sparse BM25 riêng cho caption, OCR, ASR, keyword.
4. Weighted Reciprocal Rank Fusion để tránh trộn raw score khác thang đo.
5. Hydrate canonical scene, refine keyframe theo lexical evidence.
6. AVS giới hạn kết quả/video; sequence beam-link cùng video, scene tăng dần.

## API

- `GET /v1/health`
- `POST /v1/search/kis`
- `POST /v1/search/avs`
- `POST /v1/search/sequence`
- `POST /v1/vqa`
- `GET /v1/scenes/{scene_id}`
- `GET /v1/media/{relative_path}`

Request search: `{query, top_k, filters, debug}`. Response luôn có scene,
best keyframe, timestamp, component evidence và latency. VQA mặc định trả evidence
extractive; chỉ bật generative answer khi có adapter và bắt buộc trả citation
scene/keyframe.

## UI local → Vast backend

Chạy `./scripts/run_local_ui.sh`, nhập URL public của backend và API token.
Trình duyệt gọi trực tiếp API; backend phải đưa origin UI vào
`AIC_CORS_ORIGINS`. Không dùng wildcard khi bật token. Media URL cũng qua backend
và bị giới hạn trong `AIC_DATA_ROOT`.
