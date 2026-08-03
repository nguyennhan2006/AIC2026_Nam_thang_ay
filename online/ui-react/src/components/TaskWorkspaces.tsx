import type { ApiClientConfig } from "../api";
import { SequenceTimeline } from "../features/results/SequenceTimeline";
import { SequenceViewer } from "../features/results/SequenceViewer";
import { VideoCandidateCard } from "../features/results/VideoCandidateCard";
import type { AvsResultItem, KisResultItem, QaResultItem, TrakeResultItem } from "../types";

/** KIS Safe Frame Workspace — mỗi dòng là một (video_id, frame_idx) sẵn sàng
 * nộp; safe_frame_score và must_match_coverage là hai tín hiệu quyết định
 * frame này có "an toàn" để nộp không (online/services/safe_frame.py). */
export function KisWorkspace({ items }: { items: KisResultItem[] }) {
  if (items.length === 0) return <p className="muted">Chưa có kết quả KIS.</p>;
  return (
    <table className="workspace-table">
      <thead>
        <tr>
          <th>#</th>
          <th>video_id</th>
          <th>frame_idx</th>
          <th>score</th>
          <th>safe-frame</th>
          <th>must-match</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={`${item.video_id}-${item.frame_idx}`}>
            <td>{item.rank}</td>
            <td>{item.video_id}</td>
            <td>{item.frame_idx}</td>
            <td>{item.score.toFixed(4)}</td>
            <td>{item.safe_frame_score != null ? item.safe_frame_score.toFixed(3) : "—"}</td>
            <td>{item.must_match_coverage != null ? `${Math.round(item.must_match_coverage * 100)}%` : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

const VERIFIER_LABEL: Record<QaResultItem["verifier_status"], string> = {
  SUPPORTED: "✅ supported",
  PARTIAL: "🟡 partial",
  CONTRADICTED: "❌ contradicted",
  INSUFFICIENT: "⚪ insufficient",
};

/** QA Evidence Studio — bộ ba (video, frame, answer) + verifier độc lập
 * (online/services/qa.py verify_answer chạy tách khỏi tool sinh ra answer). */
export function QaWorkspace({ items }: { items: QaResultItem[] }) {
  if (items.length === 0) return <p className="muted">Chưa có kết quả QA.</p>;
  return (
    <table className="workspace-table">
      <thead>
        <tr>
          <th>#</th>
          <th>video_id</th>
          <th>frame_idx</th>
          <th>answer</th>
          <th>type</th>
          <th>verifier</th>
          <th>joint_score</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={`${item.video_id}-${item.frame_idx}-${item.canonical_answer}`}>
            <td>{item.rank}</td>
            <td>{item.video_id}</td>
            <td>{item.frame_idx}</td>
            <td>
              <strong>{item.answer}</strong>
            </td>
            <td>{item.answer_type}</td>
            <td>{VERIFIER_LABEL[item.verifier_status]}</td>
            <td>{item.joint_score.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export interface TrakeWorkspaceProps {
  items: TrakeResultItem[];
  stepQueries: string[];
  apiConfig: ApiClientConfig;
  selectedIndex: number;
  onSelectSequence: (index: number) => void;
  activeStepIndex: number | null;
  onSelectStep: (index: number) => void;
}

/** TRAKE Alignment Studio — video candidates (Stage A) + Best Sequence ngang
 * (Stage B) + timeline (Stage C tinh chỉnh frame). docs UI competition studio
 * §11.2/§11.3/§11.4. */
export function TrakeWorkspace({
  items,
  stepQueries,
  apiConfig,
  selectedIndex,
  onSelectSequence,
  activeStepIndex,
  onSelectStep,
}: TrakeWorkspaceProps) {
  if (items.length === 0) return <p className="muted">Chưa có kết quả TRAKE.</p>;
  const selected = items[Math.min(selectedIndex, items.length - 1)];

  return (
    <div className="trake-workspace">
      <h4 className="results-subheading">Top Video Candidates</h4>
      <div className="video-candidate-list">
        {items.map((item, index) => (
          <VideoCandidateCard
            key={`${item.video_id}-${item.rank}`}
            sequence={item}
            apiConfig={apiConfig}
            active={index === selectedIndex}
            onSelect={() => onSelectSequence(index)}
          />
        ))}
      </div>

      {selected && (
        <>
          <h4 className="results-subheading">
            Best Sequence ({selected.video_id}) <span className="muted small">R-Score {selected.ordering_score.toFixed(2)}</span>
          </h4>
          <SequenceViewer
            sequence={selected}
            stepQueries={stepQueries}
            apiConfig={apiConfig}
            activeStepIndex={activeStepIndex}
            onSelectStep={onSelectStep}
          />
          <SequenceTimeline sequence={selected} activeStepIndex={activeStepIndex} onSelectStep={onSelectStep} />
        </>
      )}
    </div>
  );
}

/** AVS Relevance/Diversity Workspace — relevance_grade 0–3 + cluster_id (MMR
 * đã gom sự kiện gần trùng, online/services/avs.py). */
export function AvsWorkspace({ items }: { items: AvsResultItem[] }) {
  if (items.length === 0) return <p className="muted">Chưa có kết quả AVS.</p>;
  return (
    <table className="workspace-table">
      <thead>
        <tr>
          <th>#</th>
          <th>video_id</th>
          <th>segment</th>
          <th>frame range</th>
          <th>grade</th>
          <th>cluster</th>
          <th>score</th>
        </tr>
      </thead>
      <tbody>
        {items.map((item) => (
          <tr key={item.segment_id}>
            <td>{item.rank}</td>
            <td>{item.video_id}</td>
            <td>{item.segment_id}</td>
            <td>
              [{item.start_frame}, {item.end_frame}]
            </td>
            <td>{"★".repeat(item.relevance_grade)}{"☆".repeat(3 - item.relevance_grade)}</td>
            <td>{item.cluster_id ?? "—"}</td>
            <td>{item.score.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
