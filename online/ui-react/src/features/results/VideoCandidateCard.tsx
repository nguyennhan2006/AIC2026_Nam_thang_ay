import { ImageOff } from "lucide-react";
import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import type { TrakeResultItem } from "../../types";

export interface VideoCandidateCardProps {
  sequence: TrakeResultItem;
  apiConfig: ApiClientConfig;
  active: boolean;
  onSelect: () => void;
}

/** Video candidate (Stage A). Backend KHÔNG có field "temporal consistency"
 * riêng — không bịa thêm số; `ordering_score` đã chính là độ nhất quán thứ
 * tự nên chỉ hiện đúng nó một lần. */
export function VideoCandidateCard({ sequence, apiConfig, active, onSelect }: VideoCandidateCardProps) {
  const thumbnail = sequence.steps.find((step) => step.image_path)?.image_path ?? null;
  const totalSteps = sequence.steps.length + sequence.missing_steps.length;

  return (
    <button type="button" className={active ? "candidate-card is-active" : "candidate-card"} onClick={onSelect} aria-pressed={active}>
      <span className="candidate-thumb">
        {thumbnail ? (
          <img loading="lazy" src={mediaUrl(apiConfig, thumbnail)} alt={`Đại diện chuỗi ${sequence.video_id}`} />
        ) : (
          <ImageOff size={16} />
        )}
      </span>
      <span className="candidate-meta">
        <span className="candidate-title truncate">
          #{sequence.rank} {sequence.video_id}
        </span>
        <span className="candidate-sub tabular">
          {sequence.steps.length}/{totalSteps} bước · order {sequence.ordering_score.toFixed(2)}
        </span>
      </span>
    </button>
  );
}
