import type { ReactNode } from "react";

export interface SurfaceProps {
  children: ReactNode;
  /** `panel` = bề mặt nội dung chính; `raised` = thẻ nằm bên trong panel. */
  tone?: "panel" | "raised";
  /** Panel chiếm hết chiều cao cột và tự cuộn phần thân (workbench 3 cột). */
  fill?: boolean;
  padded?: boolean;
  className?: string;
}

/** Bề mặt cơ sở. Mọi khối nội dung phải nằm trong một Surface — nền/viền/
 * bo góc/đổ bóng chỉ được định nghĩa ở đây để elevation hierarchy nhất quán,
 * component con không tự vẽ nền riêng. */
export function Surface({ children, tone = "panel", fill = false, padded = true, className = "" }: SurfaceProps) {
  const classes = ["surface", `surface-${tone}`, fill ? "surface-fill" : "", padded ? "surface-padded" : "", className]
    .filter(Boolean)
    .join(" ");
  return <div className={classes}>{children}</div>;
}

export interface PanelHeaderProps {
  title: string;
  /** Nhãn phụ bên phải tiêu đề (vd số lượng kết quả). */
  meta?: ReactNode;
  /** Nút/điều khiển căn phải. */
  actions?: ReactNode;
  icon?: ReactNode;
  /** Header dính khi thân panel cuộn. */
  sticky?: boolean;
}

/** Header chuẩn của một panel: eyebrow-title + meta + actions trên một hàng
 * cao cố định, để mọi panel trong workbench thẳng hàng nhau. */
export function PanelHeader({ title, meta, actions, icon, sticky = false }: PanelHeaderProps) {
  return (
    <div className={sticky ? "panel-header panel-header-sticky" : "panel-header"}>
      {icon && <span className="panel-header-icon">{icon}</span>}
      <h3 className="panel-header-title truncate">{title}</h3>
      {meta != null && <span className="panel-header-meta tabular">{meta}</span>}
      {actions && <div className="panel-header-actions">{actions}</div>}
    </div>
  );
}

/** Thân panel cuộn được — dùng chung với `Surface fill` để chỉ phần này cuộn,
 * header/footer của panel đứng yên. */
export function PanelBody({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`panel-body scroll-y ${className}`}>{children}</div>;
}

/** Nhóm control trong panel, có nhãn eyebrow — thay cho việc lồng nhiều
 * Surface vào nhau (mỗi cấp lồng thêm một viền cứng nữa). */
export function Section({ title, children, actions }: { title: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="panel-section">
      <div className="panel-section-head">
        <span className="eyebrow">{title}</span>
        {actions}
      </div>
      {children}
    </section>
  );
}
