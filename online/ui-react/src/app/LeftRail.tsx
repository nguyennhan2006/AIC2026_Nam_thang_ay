import { Activity, Database, History, Search, Upload } from "lucide-react";
import type { AppPage } from "./TopNavigation";

const RAIL_ITEMS: { id: AppPage; icon: typeof Search; label: string }[] = [
  { id: "search", icon: Search, label: "Search" },
  { id: "analytics", icon: History, label: "Session / Analytics" },
  { id: "dataset", icon: Database, label: "Dataset" },
  { id: "submission", icon: Upload, label: "Submission" },
  { id: "system", icon: Activity, label: "System / Monitoring" },
];

export interface LeftRailProps {
  page: AppPage;
  onPageChange: (page: AppPage) => void;
}

/** Rail icon — cùng một state điều hướng với TopNavigation (một hệ thống,
 * hiển thị hai chỗ), không phải cấp điều hướng thứ hai. */
export function LeftRail({ page, onPageChange }: LeftRailProps) {
  return (
    <nav className="rail-nav" aria-label="Điều hướng module">
      {RAIL_ITEMS.map((item) => {
        const Icon = item.icon;
        return (
          <button
            key={item.id}
            type="button"
            className={item.id === page ? "rail-btn is-active" : "rail-btn"}
            title={item.label}
            aria-label={item.label}
            aria-current={item.id === page ? "page" : undefined}
            onClick={() => onPageChange(item.id)}
          >
            <Icon size={16} />
          </button>
        );
      })}
    </nav>
  );
}
