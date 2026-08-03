import { useState } from "react";
import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import type { TrakeResultItem } from "../../types";

export interface VideoCandidateCardProps {
  sequence: TrakeResultItem;
  apiConfig: ApiClientConfig;
  active: boolean;
  onSelect: () => void;
}

/** Video candidate card — rank/thumbnail/video_id/score/metrics thật (step
 * coverage, order score = ordering_score, avg step score = trung bình
 * confidence các step đã tìm được). KHÔNG có field "temporal consistency"
 * riêng ở backend nên không hiển thị số giả — dùng lại ordering_score vì đó
 * đúng là độ nhất quán thời gian giữa các step (docs §11.2). */
export function VideoCandidateCard({ sequence, apiConfig, active, onSelect }: VideoCandidateCardProps) {
  const [expanded, setExpanded] = useState(false);
  const thumbnail = sequence.steps.find((step) => step.image_path)?.image_path ?? null;
  const avgStepScore =
    sequence.steps.length > 0 ? sequence.steps.reduce((sum, step) => sum + step.confidence, 0) / sequence.steps.length : 0;
  // Score hiển thị chuẩn hoá về thang trực quan [0,1] — sequence_score gốc là
  // tổng cộng dồn (beam score + video score), có thể > 1.
  const displayScore = Math.max(0, Math.min(1, sequence.sequence_score / 5));

  return (
    <div className={active ? "video-candidate-card active" : "video-candidate-card"}>
      <button type="button" className="video-candidate-main" onClick={onSelect}>
        <span className="rank">#{sequence.rank}</span>
        <div className="video-candidate-thumb-wrap">
          {thumbnail ? (
            // eslint-disable-next-line jsx-a11y/alt-text
            <img className="video-candidate-thumb" loading="lazy" src={mediaUrl(apiConfig, thumbnail)} alt="" />
          ) : (
            <div className="video-candidate-thumb video-candidate-thumb-empty">—</div>
          )}
        </div>
        <div className="video-candidate-body">
          <div className="video-candidate-head">
            <strong>{sequence.video_id}</strong>
            {sequence.degraded && <span className="degraded-badge">degraded</span>}
            <output>Score: {displayScore.toFixed(3)}</output>
          </div>
          <div className="video-candidate-score-bar">
            <div className="video-candidate-score-fill" style={{ width: `${displayScore * 100}%` }} />
          </div>
          <div className="video-candidate-metrics">
            <span>
              Step Coverage <strong>{Math.round(sequence.step_coverage * (sequence.steps.length + sequence.missing_steps.length))}/
                {sequence.steps.length + sequence.missing_steps.length}</strong>
            </span>
            <span>
              Order Score <strong>{sequence.ordering_score.toFixed(2)}</strong>
            </span>
            <span>
              Avg Step Score <strong>{avgStepScore.toFixed(2)}</strong>
            </span>
            <span>
              Temporal Consistency <strong>{sequence.ordering_score.toFixed(2)}</strong>
            </span>
          </div>
        </div>
      </button>
      <button type="button" className="video-candidate-expand" onClick={() => setExpanded((prev) => !prev)}>
        {expanded ? "▲" : "▼"}
      </button>
      {expanded && (
        <ul className="video-candidate-detail">
          {sequence.steps.map((step) => (
            <li key={step.step}>
              Step {step.step}: frame {step.frame_idx} · scene {step.scene_id ?? "—"} · confidence {step.confidence.toFixed(2)}
            </li>
          ))}
          {sequence.missing_steps.map((step) => (
            <li key={`missing-${step}`} className="warning-text">
              Step {step}: missing
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
