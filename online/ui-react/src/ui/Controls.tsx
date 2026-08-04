import type { ReactNode } from "react";
import { useId } from "react";

/* ------------------------------------------------------------------ */
/* SegmentedControl — chọn 1 trong N, dùng cho Simple/Advanced, Global/Per-step.
   KHÁC Tabs: segmented đổi *chế độ* của cùng một nội dung, tabs đổi *nội dung*.
   Phân biệt rõ để không có 3 hàng pill trông giống hệt nhau cùng cấp.        */
/* ------------------------------------------------------------------ */

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
  title?: string;
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = "md",
  ariaLabel,
}: {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  size?: "sm" | "md";
  ariaLabel: string;
}) {
  return (
    <div className={`segmented segmented-${size}`} role="radiogroup" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          disabled={option.disabled}
          title={option.title}
          className={value === option.value ? "segment is-active" : "segment"}
          onClick={() => onChange(option.value)}
        >
          {option.icon}
          <span>{option.label}</span>
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Tabs — đổi nội dung của vùng bên dưới. Underline chứ không phải pill, để
   phân cấp thấp hơn SegmentedControl một bậc.                              */
/* ------------------------------------------------------------------ */

export interface TabItem<T extends string> {
  value: T;
  label: string;
  count?: number;
  icon?: ReactNode;
}

export function Tabs<T extends string>({
  items,
  value,
  onChange,
  ariaLabel,
}: {
  items: TabItem<T>[];
  value: T;
  onChange: (value: T) => void;
  ariaLabel: string;
}) {
  return (
    <div className="tabs scroll-x" role="tablist" aria-label={ariaLabel}>
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          role="tab"
          aria-selected={value === item.value}
          className={value === item.value ? "tab is-active" : "tab"}
          onClick={() => onChange(item.value)}
        >
          {item.icon}
          <span>{item.label}</span>
          {item.count != null && item.count > 0 && <span className="tab-count tabular">{item.count}</span>}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Slider — native range nhưng CÓ filled progress track. Phần đã chọn được tô
   bằng gradient tính từ `--fill` (%), nên không cần div chồng lên input và
   vẫn giữ nguyên accessibility/keyboard của input[type=range].             */
/* ------------------------------------------------------------------ */

export function Slider({
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
  disabled = false,
  ariaLabel,
  tone,
}: {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  ariaLabel: string;
  /** Màu track đã tô — mặc định accent. */
  tone?: string;
}) {
  const clamped = Math.min(Math.max(value, min), max);
  const fill = max === min ? 0 : ((clamped - min) / (max - min)) * 100;
  return (
    <input
      type="range"
      className="slider"
      min={min}
      max={max}
      step={step}
      value={clamped}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{ ["--fill" as string]: `${fill}%`, ["--slider-tone" as string]: tone }}
      onChange={(event) => onChange(Number(event.target.value))}
    />
  );
}

/* ------------------------------------------------------------------ */
/* NumericInput — width cố định, tabular-nums, không spinner mặc định của
   trình duyệt (spinner làm lệch grid và chen vào chữ số).                  */
/* ------------------------------------------------------------------ */

export function NumericInput({
  value,
  onChange,
  min,
  max,
  step,
  disabled = false,
  placeholder,
  ariaLabel,
  width = "48px",
}: {
  value: number | "" | null | undefined;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  placeholder?: string;
  ariaLabel: string;
  width?: string;
}) {
  return (
    <input
      type="number"
      className="numeric-input tabular"
      style={{ width }}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      placeholder={placeholder}
      aria-label={ariaLabel}
      value={value ?? ""}
      onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
    />
  );
}

export function TextField({
  value,
  onChange,
  placeholder,
  label,
  type = "text",
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  label: string;
  type?: string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="text-input"
        type={type}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}

export function SelectField({
  value,
  onChange,
  label,
  children,
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  children: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="field field-inline">
      <label className="field-label" htmlFor={id}>
        {label}
      </label>
      <select id={id} className="select-input" value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
        {children}
      </select>
    </div>
  );
}

export function Checkbox({
  checked,
  onChange,
  label,
  disabled = false,
  title,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <label className={disabled ? "checkbox is-disabled" : "checkbox"} title={title}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} />
      <span className="checkbox-label">{label}</span>
    </label>
  );
}
