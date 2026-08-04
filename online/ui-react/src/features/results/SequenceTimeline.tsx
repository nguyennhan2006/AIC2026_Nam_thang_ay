import type { TrakeResultItem } from "../../types";

export interface SequenceTimelineProps {
  sequence: TrakeResultItem;
  activeStepIndex: number | null;
  onSelectStep: (index: number) => void;
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Timeline ngang dưới Best Sequence — click marker để nhảy tới bước đó.
 * Không endpoint nào expose tổng thời lượng video, nên phạm vi timeline là
 * [bước đầu, bước cuối] của CHÍNH chuỗi này, không phải toàn bộ video. */
export function SequenceTimeline({ sequence, activeStepIndex, onSelectStep }: SequenceTimelineProps) {
  const timestamps = sequence.steps.map((step) => step.timestamp_sec).filter((value): value is number => value !== null);
  if (timestamps.length < 2) return null;

  const start = Math.min(...timestamps);
  const end = Math.max(...timestamps);
  const span = end - start || 1;
  const pad = span * 0.06;

  return (
    <div className="timeline">
      <div className="timeline-track" />
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
              className={activeStepIndex === stepIndex ? "timeline-mark is-active" : "timeline-mark"}
              style={{ left: `calc(${percent}% )` }}
              title={`Bước ${step.step} · frame ${step.frame_idx} · ${value.toFixed(2)}s`}
              aria-label={`Bước ${step.step} tại ${value.toFixed(1)} giây`}
              onClick={() => onSelectStep(stepIndex)}
            />
          );
        })}
      <span className="timeline-label" style={{ left: 0, transform: "none" }}>
        {formatTime(start)}
      </span>
      <span className="timeline-label" style={{ right: 0, left: "auto", transform: "none" }}>
        {formatTime(end)}
      </span>
    </div>
  );
}
