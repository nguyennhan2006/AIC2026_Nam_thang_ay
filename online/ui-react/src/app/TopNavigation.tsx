import { CircleHelp, ScanSearch, Settings } from "lucide-react";
import { Badge, IconButton } from "../ui";

export type AppPage = "search" | "analytics" | "dataset" | "submission" | "system";

const NAV_TABS: { id: AppPage; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "analytics", label: "Analytics" },
  { id: "dataset", label: "Dataset" },
  { id: "system", label: "System" },
];

export interface TopNavigationProps {
  page: AppPage;
  onPageChange: (page: AppPage) => void;
  backendStatus: "checking" | "ok" | "error";
  backendLabel: string;
  /** `?demo=1` — hiện badge để không ai nhầm phiên demo với phiên thi đấu. */
  demo?: boolean;
}

/** Nav toàn cục: chỉ điều hướng cấp trang + trạng thái backend. Không chứa
 * control của trang nào (task/top-k/mode nằm ở studio-controls) — tách cấp
 * để không có ba hàng pill trông giống nhau chồng lên nhau. */
export function TopNavigation({ page, onPageChange, backendStatus, backendLabel, demo = false }: TopNavigationProps) {
  const statusClass =
    backendStatus === "ok" ? "status-dot status-ok" : backendStatus === "error" ? "status-dot status-error" : "status-dot status-checking";
  const statusText = backendStatus === "ok" ? backendLabel : backendStatus === "error" ? "mất kết nối" : "đang kiểm tra…";

  return (
    <header className="nav-bar">
      <div className="nav-brand">
        <span className="nav-brand-mark" aria-hidden="true">
          <ScanSearch size={15} />
        </span>
        <span className="nav-brand-name">AIC 2026 Video Search</span>
        <span className="nav-brand-version">v1.0</span>
        {demo && <Badge tone="warning">demo</Badge>}
      </div>

      {/* scroll-x: ở mobile rail trái bị ẩn nên đây là điều hướng DUY NHẤT —
          cho cuộn ngang thay vì cắt mất tab, và không được tràn ra body. */}
      <nav className="nav-tabs scroll-x" aria-label="Điều hướng chính">
        {NAV_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={tab.id === page ? "nav-tab is-active" : "nav-tab"}
            aria-current={tab.id === page ? "page" : undefined}
            onClick={() => onPageChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="nav-right">
        <span className="nav-status" title={`Backend: ${backendLabel}`}>
          <span className={statusClass} aria-hidden="true" />
          <span className="truncate">{statusText}</span>
        </span>
        <IconButton icon={<CircleHelp size={15} />} label="Trợ giúp" onClick={() => onPageChange("system")} />
        <IconButton icon={<Settings size={15} />} label="Cài đặt" onClick={() => onPageChange("system")} />
        <span className="nav-avatar" aria-hidden="true">
          AD
        </span>
      </div>
    </header>
  );
}
