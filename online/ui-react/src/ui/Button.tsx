import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Loader2 } from "lucide-react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

export interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  children?: ReactNode;
  variant?: Variant;
  size?: Size;
  /** Hiện spinner và tự khoá nút — không cần tự set `disabled` kèm. */
  loading?: boolean;
  icon?: ReactNode;
  block?: boolean;
}

/** Nút chuẩn. Có đủ 5 trạng thái (hover/active/focus-visible/disabled/loading)
 * ở một chỗ, để không component nào phải tự dựng lại nút riêng. */
export function Button({
  children,
  variant = "secondary",
  size = "md",
  loading = false,
  icon,
  block = false,
  className = "",
  disabled,
  type = "button",
  ...rest
}: ButtonProps) {
  const classes = ["btn", `btn-${variant}`, `btn-${size}`, block ? "btn-block" : "", loading ? "is-loading" : "", className]
    .filter(Boolean)
    .join(" ");
  return (
    <button type={type} className={classes} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {loading ? <Loader2 className="btn-spinner" size={14} aria-hidden="true" /> : icon}
      {children != null && <span className="btn-label">{children}</span>}
    </button>
  );
}

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  icon: ReactNode;
  /** Bắt buộc: nút chỉ có icon phải có nhãn cho screen reader. */
  label: string;
  active?: boolean;
  size?: Size;
  variant?: "ghost" | "control";
}

export function IconButton({
  icon,
  label,
  active = false,
  size = "md",
  variant = "ghost",
  className = "",
  type = "button",
  ...rest
}: IconButtonProps) {
  const classes = ["icon-btn", `icon-btn-${variant}`, `icon-btn-${size}`, active ? "is-active" : "", className]
    .filter(Boolean)
    .join(" ");
  return (
    <button type={type} className={classes} title={label} aria-label={label} aria-pressed={active || undefined} {...rest}>
      {icon}
    </button>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "success" | "warning" | "danger";
}) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
