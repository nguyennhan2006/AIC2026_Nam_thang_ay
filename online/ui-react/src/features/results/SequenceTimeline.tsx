import type { TrakeResultItem } from "../../types";

const STEP_COLORS = ["#2387ff", "#56d87a", "#f0ad3d", "#ea4c89", "#a75df4", "#36c7d9"];

export interface SequenceTimelineProps {
  sequence: TrakeResultItem;
  activeStepIndex: number | null;
  onSelectStep: (index: number) => void;
}

function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return [h, m, s].map((part) => String(part).padStart(2, "0")).join(":");
}

/** Timeline ngang dưới Best Sequence — marker màu theo step, click để nhảy
 * tới đúng frame (docs §11.4). Không có endpoint nào expose tổng thời lượng
 * video cho frontend, nên phạm vi timeline là [step đầu, step cuối] của
 * CHÍNH chuỗi này (co giãn theo dữ liệu thật) — không phải toàn bộ video. */
export function SequenceTimeline({ sequence, activeStepIndex, onSelectStep }: SequenceTimelineProps) {
  const timestamps = sequence.steps
    .map((step) => step.timestamp_sec)
    .filter((value): value is number => value !== null);
  if (timestamps.length < 2) return null;

  const start = Math.min(...timestamps);
  const end = Math.max(...timestamps);
  const span = end - start || 1;
  // Đệm 5% mỗi bên để marker đầu/cuối không dính sát mép track.
  const pad = span * 0.05;

  return (
    <div className="sequence-timeline">
      <div className="sequence-timeline-track">
        {sequence.steps
          .filter((step) => step.timestamp_sec !== null)
          .map((step) => {
            const value = step.timestamp_sec as number;
            const percent = ((value - start + pad) / (span + pad * 2)) * 100;
            const stepIndex = step.step - 1;
            return (
              <button
                key={`${step.step}-${step.frame_idx}`}
                type="button"
                className={activeStepIndex === stepIndex ? "timeline-marker active" : "timeline-marker"}
                style={{ left: `${percent}%`, background: STEP_COLORS[stepIndex % STEP_COLORS.length] }}
                title={`Step ${step.step} · frame ${step.frame_idx} · ${value.toFixed(2)}s`}
                onClick={() => onSelectStep(stepIndex)}
              />
            );
          })}
      </div>
      <div className="sequence-timeline-labels">
        <span>{formatDuration(start)}</span>
        <span>{formatDuration(end)}</span>
      </div>
    </div>
  );
}
