import { useEffect, useRef, useState } from "react";
import { Download, Eye, FileText, ImageOff, PanelRight } from "lucide-react";
import type { ApiClientConfig } from "../../api";
import { mediaUrl } from "../../api";
import { BranchStatusPanel } from "../../components/BranchStatusPanel";
import { EvidenceInspector } from "../../components/EvidenceInspector";
import type { PlaybackWindow, SearchResponse, SearchHit, TrakeResultItem, TrakeStep } from "../../types";
import { Button, EmptyState, PanelBody, PanelHeader, Surface, Tabs } from "../../ui";

export interface PreviewPanelProps {
  apiConfig: ApiClientConfig;
  result: SearchResponse | null;
  selectedSequence: TrakeResultItem | null;
  activeStepIndex: number | null;
  /** Candidate đang chọn ở Results (task không phải TRAKE). */
  selectedHit: SearchHit | null;
  /** Cho phép đổi bước ngay tại panel xem; thiếu thì dải bước chỉ để đọc. */
  onSelectStep?: (index: number) => void;
}

type InspectorTab = "preview" | "evidence" | "trace";

/** Cửa sổ phát của thứ đang chọn.
 *
 * Bản cũ tra `result.sequences[].scenes[0].video_path` — chỉ TRAKE mới có
 * `sequences`, nên KIS/QA/AVS luôn rơi về ảnh tĩnh dù `selectedHit.video_path`
 * nằm ngay đó. Và kể cả khi tìm ra đường dẫn, nó phát TOÀN BỘ video chứ không
 * phải đoạn của kết quả.
 *
 * Nay backend trả `playback` đã nới bối cảnh sẵn (scene p50 chỉ 4.1 giây, xem
 * đúng 4 giây thì không hiểu chuyện gì). UI chỉ đọc, không tự tính lại.
 */
function playbackFor(
  sequence: TrakeResultItem | null,
  hit: SearchHit | null
): PlaybackWindow | null {
  return sequence?.playback ?? hit?.playback ?? null;
}

/** `#t=start,end` để trình duyệt chỉ phát đúng đoạn — endpoint `/v1/media`
 * đã hỗ trợ HTTP Range nên tua được.
 *
 * `base` PHẢI là kết quả của `mediaUrl()`. `media_path` là đường dẫn tương
 * đối trần; tự nối chuỗi sẽ ra `/v1/media/%2Fv1%2Fmedia%2F...` và nhận 400. */
function fragmentUrl(base: string, window: PlaybackWindow): string {
  return `${base}#t=${window.start_sec.toFixed(3)},${window.end_sec.toFixed(3)}`;
}

/** Dải bước của chuỗi TRAKE, ngay trong panel xem.
 *
 * Không có nó thì mỗi lần muốn xem bước khác phải quay về cột giữa — mà một
 * dòng TRAKE là `video_id, f1, ..., fn` và điểm phụ thuộc CẢ n khoảnh khắc,
 * nên duyệt qua lại giữa các bước là thao tác chính chứ không phải phụ.
 */
function StepStrip({
  steps, active, onPick,
}: {
  steps: TrakeStep[];
  active: number | null;
  onPick: ((index: number) => void) | undefined;
}) {
  return (
    <div className="step-strip" role="group" aria-label="Các bước trong chuỗi">
      {steps.map((step) => (
        <button
          key={`${step.step}-${step.frame_idx}`}
          type="button"
          className={step.step - 1 === active ? "step-chip is-active" : "step-chip"}
          onClick={() => onPick?.(step.step - 1)}
          disabled={!onPick}
          title={`Bước ${step.step} · frame ${step.frame_idx}`}
        >
          <span className="step-chip-index tabular">{step.step}</span>
          <span className="step-chip-time tabular">
            {step.timestamp_sec != null ? `${step.timestamp_sec.toFixed(1)}s` : `#${step.frame_idx}`}
          </span>
        </button>
      ))}
    </div>
  );
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
export function PreviewPanel({ apiConfig, result, selectedSequence, activeStepIndex, selectedHit, onSelectStep }: PreviewPanelProps) {
  const [tab, setTab] = useState<InspectorTab>("preview");
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoError, setVideoError] = useState(false);

  const step = activeStep(selectedSequence, activeStepIndex);
  const playback = playbackFor(selectedSequence, selectedHit);
  const imagePath = step?.image_path ?? selectedHit?.best_keyframe_path ?? null;
  // Ưu tiên timestamp của step đang chọn (TRAKE bấm vào từng bước), nếu không
  // thì nhảy tới khung được nộp chứ không tới đầu đoạn đã nới.
  const seekTo = step?.timestamp_sec ?? playback?.focus_sec ?? null;
  const evidenceCandidateId = step?.scene_id ?? selectedHit?.candidate_id ?? null;

  useEffect(() => setVideoError(false), [playback?.media_path]);

  // Đặt `currentTime` TRƯỚC khi metadata tải xong thì trình duyệt bỏ qua —
  // video nằm im ở đầu đoạn dù đã tính đúng mốc giây, và người dùng thấy
  // "video không khớp với bước đang chọn". Phải thử ngay (khi video đã sẵn
  // sàng từ lần chọn trước) VÀ nghe `loadedmetadata` cho lần tải mới.
  // `SubmissionBoard` đã xử lý bẫy này từ trước; ở đây thì chưa.
  useEffect(() => {
    const element = videoRef.current;
    if (!element || seekTo == null) return;
    const apply = () => {
      element.currentTime = Math.max(seekTo, 0);
    };
    if (element.readyState >= 1) apply();
    element.addEventListener("loadedmetadata", apply);
    return () => element.removeEventListener("loadedmetadata", apply);
  }, [seekTo, playback?.media_path]);

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

      {tab === "preview" && !step && !selectedHit && !selectedSequence && (
        /* Chưa chọn gì: MỘT empty state duy nhất — không dựng khung media đen
           rỗng rồi kèm thêm một empty state nữa bên dưới. */
        <EmptyState
          icon={<Eye size={20} />}
          title="Chưa chọn kết quả"
          description="Bấm vào một card ở cột giữa để xem khung hình, thời điểm và bằng chứng của nó."
          hints={["Card đang chọn sẽ được viền xanh", "Tab Evidence gọi GET /v1/evidence cho candidate đó"]}
        />
      )}

      {tab === "preview" && (step || selectedHit || selectedSequence) && (
        <PanelBody className="preview-body">
          <div className="preview-media">
            {playback && !videoError ? (
              <video
                ref={videoRef}
                src={fragmentUrl(mediaUrl(apiConfig, playback.media_path), playback)}
                controls
                onError={() => setVideoError(true)}
              />
            ) : imagePath ? (
              <img src={mediaUrl(apiConfig, imagePath)} alt="Khung hình đang chọn" />
            ) : (
              <span className="preview-media-empty">
                <ImageOff size={18} />
                <span>Không có media</span>
              </span>
            )}
          </div>

          {selectedSequence && selectedSequence.steps.length > 1 && (
            <StepStrip
              steps={selectedSequence.steps}
              active={activeStepIndex}
              onPick={onSelectStep}
            />
          )}

          {playback ? (
            <p className="preview-media-note">
              Đang phát {playback.start_sec.toFixed(1)}s – {playback.end_sec.toFixed(1)}s
              {" "}({(playback.end_sec - playback.start_sec).toFixed(1)}s, đã nới ±
              {playback.pad_sec.toFixed(0)}s quanh khung nộp ở {playback.focus_sec.toFixed(1)}s)
            </p>
          ) : imagePath ? (
            /* Phân biệt "chưa có video nguồn" với "player hỏng" — V002/V003
               hiện chỉ được cấp ảnh keyframe, không có mp4. */
            <p className="preview-media-note">
              Chưa có file video cho {selectedHit?.video_id ?? selectedSequence?.video_id ?? "video này"};
              đang hiển thị khung hình tĩnh.
            </p>
          ) : null}

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
