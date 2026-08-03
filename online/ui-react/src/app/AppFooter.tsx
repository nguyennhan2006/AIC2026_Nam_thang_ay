import type { HealthResponse } from "../types";

export interface AppFooterProps {
  health: HealthResponse | null;
}

/** Footer vận hành — mọi số đọc từ GET /v1/health, "—" khi chưa có thay vì
 * số giả. `is-optional` bị ẩn ở màn hẹp thay vì cho footer cuộn ngang. */
export function AppFooter({ health }: AppFooterProps) {
  return (
    <footer className="footer-bar">
      <span className="footer-item">
        Backend <strong>{health?.backend ?? "—"}</strong>
      </span>
      <span className="footer-item is-optional truncate">
        Dataset <strong className="truncate">{health?.dataset ?? "—"}</strong>
      </span>
      <span className="footer-item">
        Scenes <strong className="tabular">{health?.scene_count?.toLocaleString("vi-VN") ?? "—"}</strong>
      </span>
      <span className="footer-item is-optional">
        Keyframes <strong className="tabular">{health?.keyframe_count?.toLocaleString("vi-VN") ?? "—"}</strong>
      </span>
      <span className="footer-item is-optional">
        Build <strong>{health?.dataset_version ?? "—"}</strong>
      </span>
      <span className="footer-item footer-spacer">AIC 2026</span>
    </footer>
  );
}
