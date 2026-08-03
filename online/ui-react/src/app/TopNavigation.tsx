export type AppPage = "search" | "history" | "dataset" | "submission" | "system";

const TOP_TABS: { id: AppPage; label: string }[] = [
  { id: "search", label: "Search" },
  { id: "history", label: "Analytics" },
  { id: "dataset", label: "Dataset" },
  { id: "system", label: "System" },
];

export interface TopNavigationProps {
  page: AppPage;
  onPageChange: (page: AppPage) => void;
  backendStatus: "checking" | "ok" | "error";
  backendLabel: string;
}

/** Top nav cố định — logo/tên, version, 4 tab điều hướng chính (alias của
 * cùng state với LeftRail, không phải hệ thống điều hướng thứ hai), trạng
 * thái backend, help/settings/avatar (thuần trang trí, không claim chức năng
 * nào chưa có). */
export function TopNavigation({ page, onPageChange, backendStatus, backendLabel }: TopNavigationProps) {
  const statusClass =
    backendStatus === "ok" ? "status-dot status-ok" : backendStatus === "error" ? "status-dot status-error" : "status-dot status-checking";
  return (
    <header className="top-nav-bar">
      <div className="brand">
        <span className="brand-logo" aria-hidden="true">
          🔎
        </span>
        <strong className="brand-name">AIC Video Search</strong>
        <span className="brand-version">v1.0.0</span>
      </div>
      <nav className="top-tabs" aria-label="Điều hướng chính">
        {TOP_TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={tab.id === page ? "top-tab active" : "top-tab"}
            onClick={() => onPageChange(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="top-nav-right">
        <span className={statusClass} title={backendLabel} />
        <span className="muted small">Backend: {backendLabel}</span>
        <button type="button" className="icon-only-btn" title="Trợ giúp" aria-label="Trợ giúp">
          ?
        </button>
        <button type="button" className="icon-only-btn" title="Cài đặt" aria-label="Cài đặt">
          ⚙
        </button>
        <span className="avatar" aria-hidden="true">
          AD
        </span>
      </div>
    </header>
  );
}
