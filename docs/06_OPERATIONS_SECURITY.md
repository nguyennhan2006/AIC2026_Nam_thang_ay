# 06. Vận hành, ổn định và bảo mật

## SLO gợi ý

- Health success ≥ 99.5% trong phiên thi.
- KIS/AVS p95 ≤ 2 giây với top_k 100 (không tính cold model load).
- Không mất canonical build; checksum verify 100% trước deploy.
- Query lỗi dependency trả 503 rõ ràng, không trả kết quả rỗng như thành công.

## Checklist trước phiên

- Verify export và manifest checksum.
- Kiểm tra encoder model/revision/dimension trùng index.
- Chạy 20 golden queries và sequence regression.
- Warm worker/model và Qdrant cache.
- Kiểm tra disk, RAM/VRAM, file descriptor, clock, TLS/token/CORS.
- Snapshot Qdrant và giữ portable local index để fallback.
- Chạy `python -m scripts.preflight`; sau deploy chạy
  `python -m scripts.load_test --url https://HOST --requests 50 --concurrency 5`.

## Failure handling

| Lỗi | Hành động |
|---|---|
| GPU timeout Offline | retry idempotent; resume ledger |
| Qdrant down | restart; nếu cần chuyển local index |
| export checksum sai | không deploy; rebuild từ checkpoint |
| model OOM | giảm concurrency/batch, không nuốt lỗi |
| UI CORS | sửa exact origin, không mở `*` |
| media 404 | kiểm tra mount và AIC_DATA_ROOT |

## Logging/metrics

Log query_id, task, latency, candidate count, backend và error code; không log
API token hoặc base64 ảnh. Production nên thêm Prometheus/OpenTelemetry tại API
boundary và alert theo health, p95, error rate, GPU OOM, disk > 85%.
