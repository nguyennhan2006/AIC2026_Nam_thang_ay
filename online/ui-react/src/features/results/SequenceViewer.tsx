import { ImageOff } from "lucide-react";
import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import type { TrakeResultItem, TrakeStep } from "../../types";

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
  const label = stepText || `Bước ${index + 1}`;

  if (step === null) {
    return (
      <div className="sequence-step is-missing" title={`${label} — không tìm được candidate`}>
        <span className="sequence-step-index">{index + 1}</span>
        <span className="sequence-step-thumb">
          <ImageOff size={14} />
        </span>
        <span className="sequence-step-label truncate">missing</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      className={active ? "sequence-step is-active" : "sequence-step"}
      onClick={onSelect}
      aria-pressed={active}
      title={`${label} · frame ${step.frame_idx} · ${step.refinement}`}
    >
      <span className="sequence-step-index">{index + 1}</span>
      <span className="sequence-step-thumb">
        {step.image_path ? (
          <img loading="lazy" src={mediaUrl(apiConfig, step.image_path)} alt={label} />
        ) : (
          <ImageOff size={14} />
        )}
      </span>
      <span className="sequence-step-label truncate tabular">
        f{step.frame_idx}
        {step.timestamp_sec != null && ` · ${step.timestamp_sec.toFixed(1)}s`}
      </span>
    </button>
  );
}

/** Best Sequence — mỗi bước một card ngang. Bước thiếu bằng chứng
 * (missing_steps) vẫn hiện card mờ thay vì biến mất, để không ai tưởng chuỗi
 * chỉ có ngần ấy bước. */
export function SequenceViewer({ sequence, stepQueries, apiConfig, activeStepIndex, onSelectStep }: SequenceViewerProps) {
  const missing = new Set(sequence.missing_steps);
  // `stepQueries` đến từ `query_plan.events`, mà query_plan CHỈ có khi bật
  // debug — nếu chỉ map theo nó thì tắt debug là toàn bộ Best Sequence biến
  // mất dù backend vẫn trả đủ steps. Số bước lấy từ chính chuỗi, nhãn mới là
  // thứ tuỳ chọn.
  const stepCount = Math.max(stepQueries.length, sequence.steps.length + sequence.missing_steps.length);
  let stepPointer = 0;
  const cards = Array.from({ length: stepCount }, (_, index) => {
    const text = stepQueries[index] ?? "";
    if (missing.has(index + 1)) return { step: null, text, index };
    const step = sequence.steps[stepPointer];
    stepPointer += 1;
    return { step: step ?? null, text, index };
  });

  return (
    <div className="sequence-steps scroll-x">
      {cards.map((card) => (
        <StepCard
          key={card.index}
          step={card.step}
          index={card.index}
          stepText={card.text}
          apiConfig={apiConfig}
          active={activeStepIndex === card.index}
          onSelect={() => onSelectStep(card.index)}
        />
      ))}
    </div>
  );
}
