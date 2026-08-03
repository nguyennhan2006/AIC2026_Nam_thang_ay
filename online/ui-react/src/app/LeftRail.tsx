import type { AppPage } from "./TopNavigation";

const RAIL_ITEMS: { id: AppPage; icon: string; label: string }[] = [
  { id: "search", icon: "🔍", label: "Search" },
  { id: "history", icon: "🕓", label: "Session / History" },
  { id: "dataset", icon: "🗂", label: "Dataset" },
  { id: "submission", icon: "📤", label: "Submission" },
  { id: "system", icon: "🩺", label: "Settings / Monitoring" },
];

export interface LeftRailProps {
  page: AppPage;
  onPageChange: (page: AppPage) => void;
}

/** Left icon rail cố định — cùng state điều hướng với TopNavigation (một hệ
 * thống điều hướng duy nhất, hiển thị ở hai chỗ). Icon active có nền accent
 * xanh (docs UI competition studio). */
export function LeftRail({ page, onPageChange }: LeftRailProps) {
  return (
    <nav className="left-rail-nav" aria-label="Điều hướng module">
      {RAIL_ITEMS.map((item) => (
        <button
          key={item.id}
          type="button"
          className={item.id === page ? "rail-icon active" : "rail-icon"}
          title={item.label}
          aria-label={item.label}
          aria-current={item.id === page ? "page" : undefined}
          onClick={() => onPageChange(item.id)}
        >
          <span aria-hidden="true">{item.icon}</span>
        </button>
      ))}
    </nav>
  );
}
