import type { ApiClientConfig } from "../api";
import { mediaUrl } from "../api";
import type { SearchHit } from "../types";

export interface ResultCardProps {
  hit: SearchHit;
  onInspect: (candidateId: string) => void;
  apiConfig: ApiClientConfig;
}

/** Result card bắt buộc hiện: video_id, frame_idx, timestamp, branch
 * contribution, safe-frame score, evidence preview, branch/provenance —
 * đúng danh sách docs 01082026 §16.3 "Result card". */
export function ResultCard({ hit, onInspect, apiConfig }: ResultCardProps) {
  const contributions = Object.entries(hit.branch_contributions).sort((a, b) => b[1] - a[1]);
  const thumbnail = hit.best_keyframe_path ? (
    // eslint-disable-next-line jsx-a11y/alt-text
    <img className="thumb" loading="lazy" src={mediaUrl(apiConfig, hit.best_keyframe_path)} alt={hit.best_keyframe_id ?? ""} />
  ) : (
    <div className="thumb thumb-empty">Không có ảnh</div>
  );

  return (
    <article className={`card${hit.warnings.length ? " card-warning" : ""}`} data-candidate-id={hit.candidate_id}>
      <header className="card-head">
        <span className="rank">#{hit.rank}</span>
        <output>{hit.score.toFixed(4)}</output>
      </header>
      <div className="thumb-wrap">{thumbnail}</div>
      <div className="card-meta">
        <strong>
          {hit.video_id} · frame {hit.best_frame_idx}
        </strong>
        <span>
          {hit.best_timestamp_sec != null ? `${hit.best_timestamp_sec.toFixed(2)}s` : `${hit.start_sec.toFixed(1)}s`}
          {hit.safe_frame_score != null && ` · safe-frame ${hit.safe_frame_score.toFixed(3)}`}
        </span>
        <div className="chips">
          {hit.matched_branches.map((branch) => (
            <span key={branch} title={branch}>
              {branch}
            </span>
          ))}
        </div>
      </div>
      {contributions.length > 0 && (
        <ul className="reason compact">
          {contributions.slice(0, 3).map(([name, value]) => (
            <li key={name}>
              <span>{name}</span>
              <output>{value.toFixed(5)}</output>
            </li>
          ))}
        </ul>
      )}
      {hit.evidence.length > 0 && (
        <p className="evidence-preview">
          <strong>{hit.evidence[0].modality}</strong>: {hit.evidence[0].text.slice(0, 90)}
        </p>
      )}
      {hit.warnings.length > 0 && (
        <p className="muted warning-text">⚠ {hit.warnings[0]}</p>
      )}
      <button type="button" className="inspect-btn" onClick={() => onInspect(hit.candidate_id)}>
        Xem evidence
      </button>
    </article>
  );
}
