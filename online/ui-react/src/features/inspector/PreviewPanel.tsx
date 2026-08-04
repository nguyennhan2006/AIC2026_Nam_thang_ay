import { useEffect, useRef, useState } from "react";
import { Download, Eye, FileText, ImageOff, PanelRight } from "lucide-react";
import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import { BranchStatusPanel } from "../../components/BranchStatusPanel";
import { EvidenceInspector } from "../../components/EvidenceInspector";
import type { SearchResponse, SearchHit, TrakeResultItem, TrakeStep } from "../../types";
import { Button, EmptyState, PanelBody, PanelHeader, Surface, Tabs } from "../../ui";

export interface PreviewPanelProps {
  apiConfig: ApiClientConfig;
  result: SearchResponse | null;
  selectedSequence: TrakeResultItem | null;
  activeStepIndex: number | null;
  /** Candidate đang chọn ở Results (task không phải TRAKE). */
  selectedHit: SearchHit | null;
}

type InspectorTab = "preview" | "evidence" | "trace";

function videoPathFor(result: SearchResponse | null, videoId: string | undefined): string | null {
  if (!result || !videoId) return null;
  return result.sequences.find((item) => item.video_id === videoId)?.scenes[0]?.video_path ?? null;
}

function activeStep(sequence: TrakeResultItem | null, stepIndex: number | null): TrakeStep | null {
  if (!sequence || stepIndex === null) return null;
  return sequence.steps.find((step) => step.step - 1 === stepIndex) ?? null;
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span className="detail-label">{label}</span>
      <span className="detail-value truncate tabular" title={value}>
        {value}
      </span>
    </div>
  );
}

/** Preview & Details — rail phải. Ba tab để "mức 3" (trace) chỉ render khi
 * người dùng mở, không đổ JSON dài xuống cuối trang như bản cũ. */
export function PreviewPanel({ apiConfig, result, selectedSequence, activeStepIndex, selectedHit }: PreviewPanelProps) {
  const [tab, setTab] = useState<InspectorTab>("preview");
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoError, setVideoError] = useState(false);

  const step = activeStep(selectedSequence, activeStepIndex);
  const videoPath = videoPathFor(result, selectedSequence?.video_id ?? selectedHit?.video_id);
  const imagePath = step?.image_path ?? selectedHit?.best_keyframe_path ?? null;
  const seekTo = step?.timestamp_sec ?? selectedHit?.best_timestamp_sec ?? null;
  const evidenceCandidateId = step?.scene_id ?? selectedHit?.candidate_id ?? null;

  useEffect(() => setVideoError(false), [videoPath]);
  useEffect(() => {
    if (videoRef.current && seekTo != null) videoRef.current.currentTime = seekTo;
  }, [seekTo]);

  const header = <PanelHeader title="Preview & Details" icon={<PanelRight size={14} />} />;

  if (!result) {
    return (
      <Surface fill className="preview-panel">
        {header}
        <EmptyState
          icon={<Eye size={20} />}
          title="Chưa có gì để xem trước"
          description="Sau khi tìm kiếm, chọn một kết quả ở giữa để xem khung hình, bằng chứng và trace của nó tại đây."
        />
      </Surface>
    );
  }

  return (
    <Surface fill className="preview-panel">
      {header}
      <Tabs
        ariaLabel="Chi tiết kết quả"
        value={tab}
        onChange={setTab}
        items={[
          { value: "preview", label: "Preview" },
          { value: "evidence", label: "Evidence" },
          { value: "trace", label: "Trace" },
        ]}
      />

      {tab === "preview" && !step && !selectedHit && (
        /* Chưa chọn gì: MỘT empty state duy nhất — không dựng khung media đen
           rỗng rồi kèm thêm một empty state nữa bên dưới. */
        <EmptyState
          icon={<Eye size={20} />}
          title="Chưa chọn kết quả"
          description="Bấm vào một card ở cột giữa để xem khung hình, thời điểm và bằng chứng của nó."
          hints={["Card đang chọn sẽ được viền xanh", "Tab Evidence gọi GET /v1/evidence cho candidate đó"]}
        />
      )}

      {tab === "preview" && (step || selectedHit) && (
        <PanelBody className="preview-body">
          <div className="preview-media">
            {videoPath && !videoError ? (
              <video ref={videoRef} src={mediaUrl(apiConfig, videoPath)} controls onError={() => setVideoError(true)} />
            ) : imagePath ? (
              <img src={mediaUrl(apiConfig, imagePath)} alt="Khung hình đang chọn" />
            ) : (
              <span className="preview-media-empty">
                <ImageOff size={18} />
                <span>Không có media</span>
              </span>
            )}
          </div>

          {step ? (
            <div className="detail-list">
              <DetailRow label="Video" value={selectedSequence?.video_id ?? "—"} />
              <DetailRow
                label="Bước"
                value={`${step.step} / ${(selectedSequence?.steps.length ?? 0) + (selectedSequence?.missing_steps.length ?? 0)}`}
              />
              <DetailRow label="Frame" value={String(step.frame_idx)} />
              <DetailRow label="Timestamp" value={step.timestamp_sec != null ? `${step.timestamp_sec.toFixed(2)}s` : "—"} />
              <DetailRow label="Scene" value={step.scene_id ?? "—"} />
              <DetailRow label="Confidence" value={step.confidence.toFixed(3)} />
              <DetailRow label="Refinement" value={step.refinement} />
            </div>
          ) : selectedHit != null ? (
            <div className="detail-list">
              <DetailRow label="Video" value={selectedHit.video_id} />
              <DetailRow label="Frame" value={String(selectedHit.best_frame_idx ?? "—")} />
              <DetailRow
                label="Timestamp"
                value={selectedHit.best_timestamp_sec != null ? `${selectedHit.best_timestamp_sec.toFixed(2)}s` : "—"}
              />
              <DetailRow label="Scene" value={selectedHit.scene_id ?? "—"} />
              <DetailRow label="Score" value={selectedHit.score.toFixed(4)} />
              <DetailRow
                label="Safe-frame"
                value={selectedHit.safe_frame_score != null ? selectedHit.safe_frame_score.toFixed(3) : "—"}
              />
              <DetailRow label="Branches" value={selectedHit.matched_branches.join(", ") || "—"} />
            </div>
          ) : null}
        </PanelBody>
      )}

      {tab === "evidence" && (
        <PanelBody>
          {evidenceCandidateId ? (
            <EvidenceInspector apiConfig={apiConfig} candidateId={evidenceCandidateId} />
          ) : (
            <EmptyState
              size="sm"
              icon={<FileText size={16} />}
              title="Chưa chọn candidate"
              description="Evidence được dựng lazy ở backend — chọn một kết quả để gọi GET /v1/evidence."
            />
          )}
        </PanelBody>
      )}

      {tab === "trace" && (
        <PanelBody className="trace-body">
          <BranchStatusPanel statuses={result.branch_status} />

          {result.warnings.length > 0 && (
            <ul className="warning-list">
              {result.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          )}

          {result.query_plan && (
            <div className="detail-list">
              {Object.entries(result.query_plan.modality_weights).map(([name, value]) => (
                <DetailRow key={name} label={name} value={value.toFixed(3)} />
              ))}
            </div>
          )}

          <Button
            size="sm"
            variant="secondary"
            icon={<Download size={13} />}
            block
            onClick={() => {
              const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const anchor = document.createElement("a");
              anchor.href = url;
              anchor.download = `${result.query_id}.json`;
              anchor.click();
              URL.revokeObjectURL(url);
            }}
          >
            Tải JSON đầy đủ
          </Button>
        </PanelBody>
      )}
    </Surface>
  );
}
