import { Lock, LockOpen } from "lucide-react";
import { Slider, NumericInput } from "./Controls";

export interface WeightRowProps {
  label: string;
  value: number;
  onValueChange: (value: number) => void;
  enabled: boolean;
  onEnabledChange?: (enabled: boolean) => void;
  locked?: boolean;
  onLockToggle?: () => void;
  disabled?: boolean;
  /** Nhãn phụ hiện khi hover cả hàng (vd lý do branch bị degraded). */
  title?: string;
  badge?: string;
  min?: number;
  max?: number;
  step?: number;
  tone?: string;
}

/** Một hàng trọng số. Grid CỐ ĐỊNH 18/92/1fr/48/24 — nhờ vậy checkbox, nhãn,
 * slider, ô số và nút khoá của mọi hàng thẳng cột tuyệt đối, nhãn không bao
 * giờ wrap (đã truncate + title), và ô số không bị slider đè lên. */
export function WeightRow({
  label,
  value,
  onValueChange,
  enabled,
  onEnabledChange,
  locked = false,
  onLockToggle,
  disabled = false,
  title,
  badge,
  min = 0,
  // Thanh trượt tới 5, ô số tới 10 (trần thật của API). Tách có chủ đích:
  // mọi bản chấm dựng sẵn nằm trong 0–5, để thanh trượt 0–10 sẽ dồn vùng hay
  // dùng nhất (0–1) vào một phần mười chiều dài và gần như không kéo chính xác
  // được. Ai cần >5 thì gõ thẳng vào ô số.
  max = 5,
  step = 0.01,
  tone,
}: WeightRowProps) {
  const inert = disabled || !enabled;
  return (
    <div className={inert ? "weight-row is-inert" : "weight-row"} title={title}>
      {onEnabledChange ? (
        <input
          type="checkbox"
          className="weight-row-check"
          checked={enabled}
          disabled={disabled}
          aria-label={`Bật ${label}`}
          onChange={(event) => onEnabledChange(event.target.checked)}
        />
      ) : (
        <span className="weight-row-check-spacer" />
      )}

      <span className="weight-row-label truncate" title={label}>
        {label}
        {badge && <span className="weight-row-badge">{badge}</span>}
      </span>

      <Slider
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={inert}
        onChange={onValueChange}
        ariaLabel={`Trọng số ${label}`}
        tone={tone}
      />

      <NumericInput
        value={Number(value.toFixed(2))}
        min={min}
        max={10}
        step={step}
        disabled={inert}
        ariaLabel={`Giá trị trọng số ${label}`}
        width="100%"
        onChange={(next) => onValueChange(next ?? 0)}
      />

      {onLockToggle ? (
        <button
          type="button"
          className={locked ? "weight-row-lock is-locked" : "weight-row-lock"}
          title={locked ? `${label}: đang khoá, bấm để mở` : `${label}: khoá giá trị khi Normalize`}
          aria-label={locked ? `Mở khoá ${label}` : `Khoá ${label}`}
          aria-pressed={locked}
          onClick={onLockToggle}
        >
          {locked ? <Lock size={12} /> : <LockOpen size={12} />}
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}
