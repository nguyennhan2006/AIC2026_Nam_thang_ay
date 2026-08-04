import { ImageOff, TriangleAlert } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { mediaUrl } from "../api";
import type { SearchHit } from "../types";

export interface ResultCardProps {
  hit: SearchHit;
  onInspect: (candidateId: string) => void;
  apiConfig: ApiClientConfig;
  selected?: boolean;
}

/** Card kết quả. Bắt buộc hiện: video_id, frame_idx, timestamp, branch
 * contribution, safe-frame score, evidence preview — cả card là một nút
 * (chọn để xem ở Preview), không phải div có nút con. */
export function ResultCard({ hit, onInspect, apiConfig, selected = false }: ResultCardProps) {
  const contributions = Object.entries(hit.branch_contributions).sort((a, b) => b[1] - a[1]);
  const maxContribution = contributions.length > 0 ? contributions[0][1] : 0;

  return (
    <button
      type="button"
      className={selected ? "result-card is-selected" : "result-card"}
      data-candidate-id={hit.candidate_id}
      aria-pressed={selected}
      onClick={() => onInspect(hit.candidate_id)}
    >
      <div className="result-thumb">
        {hit.best_keyframe_path ? (
          <img loading="lazy" src={mediaUrl(apiConfig, hit.best_keyframe_path)} alt={`Khung hình ${hit.best_frame_idx} của ${hit.video_id}`} />
        ) : (
          <span className="result-thumb-empty">
            <ImageOff size={18} />
          </span>
        )}
        <span className="result-rank">#{hit.rank}</span>
        <span className="result-score tabular">{hit.score.toFixed(3)}</span>
        {hit.warnings.length > 0 && (
          <span className="result-warn" title={hit.warnings[0]}>
            <TriangleAlert size={11} />
          </span>
        )}
      </div>

      <div className="result-body">
        <span className="result-title truncate">{hit.video_id}</span>
        <span className="result-sub tabular">
          frame {hit.best_frame_idx ?? "—"}
          {hit.best_timestamp_sec != null && ` · ${hit.best_timestamp_sec.toFixed(1)}s`}
          {hit.safe_frame_score != null && ` · sf ${hit.safe_frame_score.toFixed(2)}`}
        </span>

        {contributions.length > 0 && (
          <div className="contrib-list">
            {contributions.slice(0, 3).map(([name, value]) => (
              <span key={name} className="contrib" title={`${name}: ${value.toFixed(5)}`}>
                <span className="contrib-name truncate">{name.replace(/^bm25_/, "")}</span>
                <span className="contrib-bar" aria-hidden="true">
                  <span style={{ width: `${maxContribution > 0 ? (value / maxContribution) * 100 : 0}%` }} />
                </span>
              </span>
            ))}
          </div>
        )}

        {hit.evidence.length > 0 && (
          <span className="result-evidence truncate" title={hit.evidence[0].text}>
            <span className="result-evidence-tag">{hit.evidence[0].modality}</span>
            {hit.evidence[0].text}
          </span>
        )}
      </div>
    </button>
  );
}
