import type { ReactNode } from "react";

export interface AppShellProps {
  nav: ReactNode;
  rail: ReactNode;
  footer: ReactNode;
  children: ReactNode;
}

/** Khung gốc: nav (48px) + rail (52px) + main + footer (30px), khoá trong
 * 100dvh. `main` KHÔNG cuộn — mỗi panel bên trong tự cuộn, nên header/footer
 * luôn đứng yên và trang không bao giờ sinh scrollbar ở body. */
export function AppShell({ nav, rail, footer, children }: AppShellProps) {
  return (
    <div className="app-shell">
      <div className="shell-nav">{nav}</div>
      <div className="shell-rail">{rail}</div>
      <main className="shell-main">{children}</main>
      <div className="shell-footer">{footer}</div>
    </div>
  );
}
