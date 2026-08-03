import type { ReactNode } from "react";

export interface AppShellProps {
  topNav: ReactNode;
  leftRail: ReactNode;
  footer: ReactNode;
  children: ReactNode;
}

/** Grid gốc của toàn app: top nav (52px) + left rail (56px, sticky) + main
 * (scroll riêng) + footer (32px). Xem docs UI competition studio §4.1. */
export function AppShell({ topNav, leftRail, footer, children }: AppShellProps) {
  return (
    <div className="app-shell">
      <div className="top-nav">{topNav}</div>
      <div className="left-rail">{leftRail}</div>
      <main className="app-main">{children}</main>
      <div className="app-footer">{footer}</div>
    </div>
  );
}
