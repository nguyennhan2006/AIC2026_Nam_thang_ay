import type { HealthResponse } from "../types";

export interface AppFooterProps {
  health: HealthResponse | null;
}

function formatLastUpdated(iso: string | null | undefined): string {
  if (!iso) return "—";
  // dataset_version là build_id dạng "20260803T081508Z" (offline assemble) —
  // hiển thị nguyên văn thay vì suy diễn định dạng ngày khác.
  return iso;
}

/** Footer cố định — thông tin vận hành đọc từ GET /v1/health, không phải số
 * trang trí. "Team/status" chỉ là nhãn tĩnh (không claim dữ liệu backend). */
export function AppFooter({ health }: AppFooterProps) {
  return (
    <footer className="app-footer-bar">
      <span>Backend: {health?.backend ?? "—"}</span>
      <span>Dataset: {health?.dataset ?? "—"}</span>
      <span>Scenes: {health?.scene_count ?? "—"}</span>
      <span>Keyframes: {health?.keyframe_count ?? "—"}</span>
      <span>Last updated: {formatLastUpdated(health?.dataset_version)}</span>
      <span className="footer-team">AIC 2026 · Team</span>
      <span className={`status-dot ${health ? "status-ok" : "status-checking"}`} aria-hidden="true" />
    </footer>
  );
}
