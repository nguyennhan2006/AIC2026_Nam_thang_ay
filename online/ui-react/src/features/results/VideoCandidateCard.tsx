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

/** Video candidate (Stage A).
 *
 * Hiện ảnh của MỌI bước, không phải mỗi bước đầu. Đo trên dữ liệu thật: một
 * truy vấn 3 bước trả 20 chuỗi khác nhau hoàn toàn, nhưng **16/20 dùng chung
 * frame bước 1**, nên card cũ vẽ ra 16 ô giống hệt nhau và người dùng tưởng
 * bấm gì cũng ra một kết quả. Chuỗi chỉ phân biệt được ở các bước SAU.
 *
 * Và hiện `sequence_score` chứ không phải `ordering_score`: cùng bộ 20 chuỗi
 * đó, `ordering_score` bằng 0.50 ở CẢ HAI MƯƠI, còn `sequence_score` trải từ
 * 3.516 xuống 3.000. Một con số như nhau ở mọi dòng thì không giúp chọn.
 */
export function VideoCandidateCard({ sequence, apiConfig, active, onSelect }: VideoCandidateCardProps) {
  const totalSteps = sequence.steps.length + sequence.missing_steps.length;

  return (
    <button
      type="button"
      className={active ? "candidate-card is-active" : "candidate-card"}
      onClick={onSelect}
      aria-pressed={active}
      title={`Chuỗi #${sequence.rank} · frame ${sequence.frame_ids.join(", ")}`}
    >
      <span className="candidate-strip">
        {sequence.steps.map((step) => (
          <span className="candidate-thumb" key={`${step.step}-${step.frame_idx}`}>
            {step.image_path ? (
              <img
                loading="lazy"
                src={mediaUrl(apiConfig, step.image_path)}
                alt={`Bước ${step.step} của chuỗi ${sequence.video_id}`}
              />
            ) : (
              <ImageOff size={14} />
            )}
            <span className="candidate-thumb-index tabular">{step.step}</span>
          </span>
        ))}
      </span>
      <span className="candidate-meta">
        <span className="candidate-title truncate">
          #{sequence.rank} {sequence.video_id}
        </span>
        <span className="candidate-sub tabular">
          {sequence.steps.length}/{totalSteps} bước · điểm {sequence.sequence_score.toFixed(2)}
        </span>
      </span>
    </button>
  );
}
