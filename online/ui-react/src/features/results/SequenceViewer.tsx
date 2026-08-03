import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import type { TrakeResultItem, TrakeStep } from "../../types";

const STEP_COLORS = ["#2387ff", "#56d87a", "#f0ad3d", "#ea4c89", "#a75df4", "#36c7d9"];

function stepColor(index: number): string {
  return STEP_COLORS[index % STEP_COLORS.length];
}

export interface SequenceViewerProps {
  sequence: TrakeResultItem;
  stepQueries: string[];
  apiConfig: ApiClientConfig;
  activeStepIndex: number | null;
  onSelectStep: (index: number) => void;
}

function StepCard({
  step,
  index,
  stepText,
  apiConfig,
  active,
  onSelect,
}: {
  step: TrakeStep | null;
  index: number;
  stepText: string;
  apiConfig: ApiClientConfig;
  active: boolean;
  onSelect: () => void;
}) {
  if (step === null) {
    return (
      <div className="sequence-step-card sequence-step-missing">
        <span className="sequence-step-badge" style={{ background: stepColor(index) }}>
          {index + 1}
        </span>
        <p className="sequence-step-text">{stepText || `Step ${index + 1}`}</p>
        <p className="muted small">Missing</p>
      </div>
    );
  }
  const timestampSec = step.timestamp_sec;
  return (
    <button
      type="button"
      className={active ? "sequence-step-card active" : "sequence-step-card"}
      onClick={onSelect}
    >
      <span className="sequence-step-badge" style={{ background: stepColor(index) }}>
        {index + 1}
      </span>
      <p className="sequence-step-text">{stepText || `Step ${index + 1}`}</p>
      <div className="sequence-step-thumb-wrap">
        {step.image_path ? (
          // eslint-disable-next-line jsx-a11y/alt-text
          <img className="sequence-step-thumb" loading="lazy" src={mediaUrl(apiConfig, step.image_path)} alt="" />
        ) : (
          <div className="sequence-step-thumb sequence-step-thumb-empty">Không có ảnh</div>
        )}
        {timestampSec !== null && <span className="sequence-step-time">{timestampSec.toFixed(2)}s</span>}
      </div>
      <p className="muted small">
        Frame: {step.frame_idx} · Scene: {step.scene_id ?? "—"}
      </p>
      <p className="muted small">
        Score: {step.confidence.toFixed(2)} ·{" "}
        <span className={step.refinement === "dense_window" ? "refinement-dense" : "refinement-sparse"}>{step.refinement}</span>
      </p>
    </button>
  );
}

/** Best Sequence — mỗi event một card ngang, nối bằng mũi tên; step thiếu
 * bằng chứng (missing_steps) hiện card dashed thay vì làm cả chuỗi biến mất
 * (docs §11.3). */
export function SequenceViewer({ sequence, stepQueries, apiConfig, activeStepIndex, onSelectStep }: SequenceViewerProps) {
  const missing = new Set(sequence.missing_steps);
  let stepPointer = 0;
  const cards = stepQueries.map((text, index) => {
    if (missing.has(index + 1)) {
      return { step: null, text, index };
    }
    const step = sequence.steps[stepPointer];
    stepPointer += 1;
    return { step: step ?? null, text, index };
  });

  return (
    <div className="sequence-viewer">
      <div className="sequence-viewer-row">
        {cards.map((card, position) => (
          <div key={card.index} className="sequence-step-slot">
            <StepCard
              step={card.step}
              index={card.index}
              stepText={card.text}
              apiConfig={apiConfig}
              active={activeStepIndex === card.index}
              onSelect={() => onSelectStep(card.index)}
            />
            {position < cards.length - 1 && (
              <span className="sequence-arrow" aria-hidden="true">
                →
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
