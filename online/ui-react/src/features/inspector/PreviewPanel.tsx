import { useEffect, useMemo, useRef, useState } from "react";
import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import { BranchStatusPanel } from "../../components/BranchStatusPanel";
import { EvidenceInspector } from "../../components/EvidenceInspector";
import type { SearchResponse, TrakeResultItem, TrakeStep } from "../../types";

export interface PreviewPanelProps {
  apiConfig: ApiClientConfig;
  result: SearchResponse | null;
  selectedSequence: TrakeResultItem | null;
  activeStepIndex: number | null;
}

type InspectorTab = "preview" | "evidence" | "trace";

function videoPathFor(result: SearchResponse | null, videoId: string | undefined): string | null {
  if (!result || !videoId) return null;
  const sequence = result.sequences.find((item) => item.video_id === videoId);
  return sequence?.scenes[0]?.video_path ?? null;
}

function activeStep(sequence: TrakeResultItem | null, stepIndex: number | null): TrakeStep | null {
  if (!sequence || stepIndex === null) return null;
  return sequence.steps.find((step) => step.step - 1 === stepIndex) ?? null;
}

/** Preview & Details — panel sticky bên phải: media player + step details +
 * evidence + trace (docs §12). Chia ba tab để "mức 3" (trace) chỉ tải khi
 * người dùng thực sự mở, không render JSON dài mặc định (docs §3.3/§11.1). */
export function PreviewPanel({ apiConfig, result, selectedSequence, activeStepIndex }: PreviewPanelProps) {
  const [tab, setTab] = useState<InspectorTab>("preview");
  const videoRef = useRef<HTMLVideoElement>(null);
  const step = activeStep(selectedSequence, activeStepIndex);
  const videoPath = videoPathFor(result, selectedSequence?.video_id);
  const [videoError, setVideoError] = useState(false);

  useEffect(() => {
    setVideoError(false);
  }, [videoPath]);

  useEffect(() => {
    if (videoRef.current && step?.timestamp_sec != null) {
      videoRef.current.currentTime = step.timestamp_sec;
    }
  }, [step?.timestamp_sec]);

  const evidenceCandidateId = useMemo(() => step?.scene_id ?? null, [step]);

  if (!result) {
    return (
      <div className="preview-panel">
        <p className="muted">Chưa có kết quả để xem trước.</p>
      </div>
    );
  }

  return (
    <div className="preview-panel">
      <div className="preview-tabs">
        <button type="button" className={tab === "preview" ? "weight-tab active" : "weight-tab"} onClick={() => setTab("preview")}>
          Preview
        </button>
        <button type="button" className={tab === "evidence" ? "weight-tab active" : "weight-tab"} onClick={() => setTab("evidence")}>
          Evidence
        </button>
        <button type="button" className={tab === "trace" ? "weight-tab active" : "weight-tab"} onClick={() => setTab("trace")}>
          Trace
        </button>
      </div>

      {tab === "preview" && (
        <>
          <div className="preview-media-wrap">
            {videoPath && !videoError ? (
              <video
                ref={videoRef}
                className="preview-media"
                src={mediaUrl(apiConfig, videoPath)}
                controls
                onError={() => setVideoError(true)}
              />
            ) : step?.image_path ? (
              // eslint-disable-next-line jsx-a11y/alt-text
              <img className="preview-media" src={mediaUrl(apiConfig, step.image_path)} alt="" />
            ) : (
              <div className="preview-media preview-media-empty">Chưa có media để xem trước</div>
            )}
          </div>

          {step ? (
            <div className="preview-step-details">
              <h4>
                Thông tin step hiện tại{" "}
                {selectedSequence && (
                  <span className="muted small">
                    Step {step.step} / {selectedSequence.steps.length + selectedSequence.missing_steps.length}
                  </span>
                )}
              </h4>
              <dl className="preview-detail-list">
                <dt>Frame Index</dt>
                <dd>{step.frame_idx}</dd>
                <dt>Timestamp</dt>
                <dd>{step.timestamp_sec != null ? `${step.timestamp_sec.toFixed(2)}s` : "—"}</dd>
                <dt>Scene</dt>
                <dd>{step.scene_id ?? "—"}</dd>
                <dt>Confidence</dt>
                <dd>{step.confidence.toFixed(3)}</dd>
                <dt>Refinement</dt>
                <dd>{step.refinement}</dd>
              </dl>
            </div>
          ) : (
            <p className="muted">Chọn một step trong Best Sequence để xem chi tiết.</p>
          )}
        </>
      )}

      {tab === "evidence" && <EvidenceInspector apiConfig={apiConfig} candidateId={evidenceCandidateId} />}

      {tab === "trace" && (
        <div className="preview-trace">
          <BranchStatusPanel statuses={result.branch_status} />
          {result.warnings.length > 0 && (
            <ul className="reason">
              {result.warnings.map((warning, index) => (
                <li key={index}>
                  <span className="warning-text">{warning}</span>
                </li>
              ))}
            </ul>
          )}
          {result.query_plan && (
            <details>
              <summary>Modality weights (query_plan)</summary>
              <ul className="reason">
                {Object.entries(result.query_plan.modality_weights).map(([name, value]) => (
                  <li key={name}>
                    <span>{name}</span>
                    <output>{value.toFixed(3)}</output>
                  </li>
                ))}
              </ul>
            </details>
          )}
          <button
            type="button"
            onClick={() => {
              const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `${result.query_id}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}
          >
            Download JSON
          </button>
        </div>
      )}
    </div>
  );
}
