import { useEffect, useState } from "react";
import type { ApiClientConfig } from "../api";
import { ApiError, getEvidence, mediaUrl } from "../api";
import type { EvidencePack } from "../types";

export interface EvidenceInspectorProps {
  apiConfig: ApiClientConfig;
  candidateId: string | null;
  onClose?: () => void;
}

/** Panel/modal gọi GET /v1/evidence/{candidate_id} — dựng lazy phía backend
 * (online/services/evidence_builder.py), nên mỗi lần mở một candidate mới
 * đều là một request thật, không phải dữ liệu đã có sẵn trong SearchHit. */
export function EvidenceInspector({ apiConfig, candidateId, onClose }: EvidenceInspectorProps) {
  const [pack, setPack] = useState<EvidencePack | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!candidateId) {
      setPack(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    getEvidence(apiConfig, candidateId)
      .then((result) => {
        if (!cancelled) setPack(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiConfig, candidateId]);

  if (!candidateId) return null;

  return (
    <div className="evidence-panel" role="dialog" aria-label="Evidence Inspector">
      <header className="evidence-head">
        <h3>Evidence · {candidateId}</h3>
        {onClose && (
          <button type="button" onClick={onClose} aria-label="Đóng">
            ✕
          </button>
        )}
      </header>
      {loading && <p className="muted">Đang tải evidence…</p>}
      {error && <p className="muted">Lỗi: {error}</p>}
      {pack && !loading && !error && (
        <div className="evidence-body">
          <p>
            <strong>{pack.video_id}</strong> · frame [{pack.start_frame}, {pack.end_frame_exclusive})
            {pack.best_frame_idx != null && <> · best_frame_idx={pack.best_frame_idx}</>}
          </p>
          {pack.caption_text && (
            <section>
              <h4>Caption</h4>
              <p>{pack.caption_text}</p>
            </section>
          )}
          {pack.ocr_text && (
            <section>
              <h4>OCR</h4>
              <p>{pack.ocr_text}</p>
            </section>
          )}
          {pack.asr_window && (
            <section>
              <h4>ASR</h4>
              <p>{pack.asr_window}</p>
            </section>
          )}
          {pack.keyframes.length > 0 && (
            <section>
              <h4>Keyframes ({pack.keyframes.length})</h4>
              <div className="evidence-thumbs">
                {pack.keyframes.map((frame) => (
                  <figure key={frame.keyframe_id}>
                    {/* eslint-disable-next-line jsx-a11y/alt-text */}
                    <img loading="lazy" src={mediaUrl(apiConfig, frame.image_path)} alt={frame.keyframe_id} />
                    <figcaption>frame_idx={frame.frame_idx}</figcaption>
                  </figure>
                ))}
              </div>
            </section>
          )}
          {(pack.previous_context || pack.next_context) && (
            <section>
              <h4>Neighbor context</h4>
              <div className="neighbor-grid">
                <div>
                  <p className="muted">Trước</p>
                  <p>{pack.previous_context ? pack.previous_context.caption ?? "(không caption)" : "—"}</p>
                </div>
                <div>
                  <p className="muted">Sau</p>
                  <p>{pack.next_context ? pack.next_context.caption ?? "(không caption)" : "—"}</p>
                </div>
              </div>
            </section>
          )}
          {Object.keys(pack.branch_contributions).length > 0 && (
            <section>
              <h4>Branch contributions</h4>
              <ul className="reason">
                {Object.entries(pack.branch_contributions)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, value]) => (
                    <li key={name}>
                      <span>{name}</span>
                      <output>{value.toFixed(5)}</output>
                    </li>
                  ))}
              </ul>
            </section>
          )}
          {pack.rule_adjustments.length > 0 && (
            <section>
              <h4>Rule adjustments</h4>
              <ul className="reason">
                {pack.rule_adjustments.map((item, i) => (
                  <li key={i}>
                    <span>{item.rule}</span>
                    <output>{item.delta >= 0 ? "+" : ""}{item.delta.toFixed(4)}</output>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {pack.dataset_version && <p className="muted">dataset_version={pack.dataset_version}</p>}
        </div>
      )}
    </div>
  );
}
