import { useEffect, useState } from "react";
import { X } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { ApiError, getEvidence, mediaUrl } from "../api";
import type { EvidencePack } from "../types";
import { IconButton, InlineError, Skeleton } from "../ui";

export interface EvidenceInspectorProps {
  apiConfig: ApiClientConfig;
  candidateId: string | null;
  onClose?: () => void;
}

function EvidenceSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="evidence-section">
      <span className="eyebrow">{title}</span>
      {children}
    </section>
  );
}

/** GET /v1/evidence/{candidate_id} — backend dựng lazy, nên mỗi candidate mở
 * ra là một request thật, không phải dữ liệu đã nằm sẵn trong SearchHit. */
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

  if (loading) {
    return (
      <div className="evidence">
        <Skeleton height={11} width="55%" />
        <Skeleton height={52} radius="var(--radius-nested)" />
        <Skeleton height={52} radius="var(--radius-nested)" />
      </div>
    );
  }

  if (error) return <InlineError message={error} />;
  if (!pack) return null;

  const contributions = Object.entries(pack.branch_contributions).sort((a, b) => b[1] - a[1]);

  return (
    <div className="evidence">
      <header className="evidence-head">
        <span className="truncate" title={candidateId}>
          <strong>{pack.video_id}</strong>
          <span className="text-tertiary">
            {" "}
            · frames [{pack.start_frame}, {pack.end_frame_exclusive})
            {pack.best_frame_idx != null && ` · best ${pack.best_frame_idx}`}
          </span>
        </span>
        {onClose && <IconButton icon={<X size={13} />} label="Đóng" size="sm" onClick={onClose} />}
      </header>

      {pack.caption_text && (
        <EvidenceSection title="Caption">
          <p className="evidence-text">{pack.caption_text}</p>
        </EvidenceSection>
      )}
      {pack.ocr_text && (
        <EvidenceSection title="OCR">
          <p className="evidence-text">{pack.ocr_text}</p>
        </EvidenceSection>
      )}
      {pack.asr_window && (
        <EvidenceSection title="ASR">
          <p className="evidence-text">{pack.asr_window}</p>
        </EvidenceSection>
      )}

      {pack.keyframes.length > 0 && (
        <EvidenceSection title={`Keyframes (${pack.keyframes.length})`}>
          <div className="evidence-thumbs">
            {pack.keyframes.map((frame) => (
              <figure key={frame.keyframe_id}>
                <img loading="lazy" src={mediaUrl(apiConfig, frame.image_path)} alt={frame.keyframe_id} />
                <figcaption className="tabular">{frame.frame_idx}</figcaption>
              </figure>
            ))}
          </div>
        </EvidenceSection>
      )}

      {contributions.length > 0 && (
        <EvidenceSection title="Branch contributions">
          <div className="detail-list">
            {contributions.map(([name, value]) => (
              <div key={name} className="detail-row">
                <span className="detail-label truncate">{name}</span>
                <span className="detail-value tabular">{value.toFixed(5)}</span>
              </div>
            ))}
          </div>
        </EvidenceSection>
      )}

      {pack.rule_adjustments.length > 0 && (
        <EvidenceSection title="Rule adjustments">
          <div className="detail-list">
            {pack.rule_adjustments.map((item, index) => (
              <div key={index} className="detail-row">
                <span className="detail-label truncate">{item.rule}</span>
                <span className="detail-value tabular">
                  {item.delta >= 0 ? "+" : ""}
                  {item.delta.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </EvidenceSection>
      )}
    </div>
  );
}
