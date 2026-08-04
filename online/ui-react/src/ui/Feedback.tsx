import type { ReactNode } from "react";

export interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description: string;
  /** Gợi ý thao tác cụ thể — empty state phải nói "làm gì tiếp", không chỉ
   * "chưa có dữ liệu". */
  hints?: string[];
  action?: ReactNode;
  size?: "sm" | "md";
}

export function EmptyState({ icon, title, description, hints, action, size = "md" }: EmptyStateProps) {
  return (
    <div className={`empty-state empty-state-${size}`}>
      <div className="empty-state-icon" aria-hidden="true">
        {icon}
      </div>
      <p className="empty-state-title">{title}</p>
      <p className="empty-state-desc">{description}</p>
      {hints && hints.length > 0 && (
        <ul className="empty-state-hints">
          {hints.map((hint) => (
            <li key={hint}>{hint}</li>
          ))}
        </ul>
      )}
      {action && <div className="empty-state-action">{action}</div>}
    </div>
  );
}

/** Khối xám nhấp nháy giữ đúng chỗ của nội dung sắp tới — tránh layout nhảy
 * khi dữ liệu về (và tránh "vùng rỗng thô" trong lúc chờ). */
export function Skeleton({ width = "100%", height = 12, radius = "var(--radius-badge)" }: { width?: string | number; height?: string | number; radius?: string }) {
  return <span className="skeleton" style={{ width, height, borderRadius: radius }} aria-hidden="true" />;
}

export function SkeletonCard() {
  return (
    <div className="skeleton-card" aria-hidden="true">
      <Skeleton height={92} radius="var(--radius-nested)" />
      <Skeleton width="70%" height={11} />
      <Skeleton width="45%" height={10} />
    </div>
  );
}

export function SkeletonRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="skeleton-rows" aria-hidden="true" aria-busy="true">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="skeleton-row">
          <Skeleton width="88px" height={11} />
          <Skeleton height={5} radius="var(--radius-pill)" />
          <Skeleton width="40px" height={11} />
        </div>
      ))}
    </div>
  );
}

export function InlineError({ message }: { message: string }) {
  return (
    <p className="inline-error" role="alert">
      {message}
    </p>
  );
}
