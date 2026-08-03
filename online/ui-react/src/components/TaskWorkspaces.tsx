import { CircleCheck, CircleDashed, CircleSlash, CircleX, Inbox } from "lucide-react";
import type { ReactNode } from "react";
import type { ApiClientConfig } from "../api";
import { SequenceTimeline } from "../features/results/SequenceTimeline";
import { SequenceViewer } from "../features/results/SequenceViewer";
import { VideoCandidateCard } from "../features/results/VideoCandidateCard";
import type { AvsResultItem, KisResultItem, QaResultItem, TrakeResultItem } from "../types";
import { EmptyState } from "../ui";

function TaskEmpty({ task, hint }: { task: string; hint: string }) {
  return (
    <EmptyState
      size="sm"
      icon={<Inbox size={18} />}
      title={`Chưa có kết quả ${task}`}
      description={hint}
    />
  );
}

function WorkspaceTable({ head, children }: { head: ReactNode; children: ReactNode }) {
  return (
    <div className="results-scroll scroll-y">
      <table className="data-table">
        <thead>{head}</thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

/** KIS Safe Frame Workspace — mỗi dòng là một (video_id, frame_idx) sẵn sàng
 * nộp; safe_frame_score và must_match_coverage quyết định frame có "an toàn"
 * để nộp không (online/services/safe_frame.py). */
export function KisWorkspace({ items }: { items: KisResultItem[] }) {
  if (items.length === 0) return <TaskEmpty task="KIS" hint="Chạy tìm kiếm với task KIS để có danh sách frame sẵn sàng nộp." />;
  return (
    <WorkspaceTable
      head={
        <tr>
          <th scope="col">#</th>
          <th scope="col">Video</th>
          <th scope="col" className="num">Frame</th>
          <th scope="col" className="num">Score</th>
          <th scope="col" className="num">Safe-frame</th>
          <th scope="col" className="num">Must-match</th>
        </tr>
      }
    >
      {items.map((item) => (
        <tr key={`${item.video_id}-${item.frame_idx}`}>
          <td className="num muted-cell">{item.rank}</td>
          <td className="truncate">{item.video_id}</td>
          <td className="num tabular">{item.frame_idx}</td>
          <td className="num tabular">{item.score.toFixed(4)}</td>
          <td className="num tabular">{item.safe_frame_score != null ? item.safe_frame_score.toFixed(3) : "—"}</td>
          <td className="num tabular">{item.must_match_coverage != null ? `${Math.round(item.must_match_coverage * 100)}%` : "—"}</td>
        </tr>
      ))}
    </WorkspaceTable>
  );
}

const VERIFIER_META: Record<QaResultItem["verifier_status"], { icon: ReactNode; label: string; tone: string }> = {
  SUPPORTED: { icon: <CircleCheck size={12} />, label: "supported", tone: "verdict-ok" },
  PARTIAL: { icon: <CircleDashed size={12} />, label: "partial", tone: "verdict-warn" },
  CONTRADICTED: { icon: <CircleX size={12} />, label: "contradicted", tone: "verdict-bad" },
  INSUFFICIENT: { icon: <CircleSlash size={12} />, label: "insufficient", tone: "verdict-muted" },
};

/** QA Evidence Studio — bộ ba (video, frame, answer) + verifier độc lập
 * (verify_answer chạy tách khỏi tool sinh ra answer). */
export function QaWorkspace({ items }: { items: QaResultItem[] }) {
  if (items.length === 0) return <TaskEmpty task="QA" hint="Đặt câu hỏi ở ô truy vấn với task QA để nhận câu trả lời kèm bằng chứng." />;
  return (
    <WorkspaceTable
      head={
        <tr>
          <th scope="col">#</th>
          <th scope="col">Video</th>
          <th scope="col" className="num">Frame</th>
          <th scope="col">Answer</th>
          <th scope="col">Type</th>
          <th scope="col">Verifier</th>
          <th scope="col" className="num">Joint</th>
        </tr>
      }
    >
      {items.map((item) => {
        const verdict = VERIFIER_META[item.verifier_status];
        return (
          <tr key={`${item.video_id}-${item.frame_idx}-${item.canonical_answer}`}>
            <td className="num muted-cell">{item.rank}</td>
            <td className="truncate">{item.video_id}</td>
            <td className="num tabular">{item.frame_idx}</td>
            <td className="cell-strong">{item.answer}</td>
            <td className="muted-cell">{item.answer_type}</td>
            <td>
              <span className={`verdict ${verdict.tone}`}>
                {verdict.icon}
                {verdict.label}
              </span>
            </td>
            <td className="num tabular">{item.joint_score.toFixed(4)}</td>
          </tr>
        );
      })}
    </WorkspaceTable>
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
 * (Stage B) + timeline (Stage C tinh chỉnh frame). */
export function TrakeWorkspace({
  items,
  stepQueries,
  apiConfig,
  selectedIndex,
  onSelectSequence,
  activeStepIndex,
  onSelectStep,
}: TrakeWorkspaceProps) {
  if (items.length === 0)
    return <TaskEmpty task="TRAKE" hint="Mô tả chuỗi sự kiện theo thứ tự (phân tách bằng dấu chấm phẩy) rồi tìm kiếm với task TRAKE." />;
  const selected = items[Math.min(selectedIndex, items.length - 1)];

  return (
    <div className="results-scroll scroll-y">
      <section className="result-group">
        <header className="result-group-head">
          <span className="eyebrow">Video candidates</span>
          <span className="result-group-meta tabular">{items.length}</span>
        </header>
        <div className="candidate-row scroll-x">
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
      </section>

      {selected && (
        <section className="result-group">
          <header className="result-group-head">
            <span className="eyebrow">Best sequence</span>
            <span className="result-group-meta truncate">
              {selected.video_id} · R-Score {selected.ordering_score.toFixed(2)}
            </span>
          </header>
          <SequenceViewer
            sequence={selected}
            stepQueries={stepQueries}
            apiConfig={apiConfig}
            activeStepIndex={activeStepIndex}
            onSelectStep={onSelectStep}
          />
          <SequenceTimeline sequence={selected} activeStepIndex={activeStepIndex} onSelectStep={onSelectStep} />
        </section>
      )}
    </div>
  );
}

/** AVS Relevance/Diversity Workspace — relevance_grade 0–3 + cluster_id
 * (MMR đã gom sự kiện gần trùng, online/services/avs.py). */
export function AvsWorkspace({ items }: { items: AvsResultItem[] }) {
  if (items.length === 0) return <TaskEmpty task="AVS" hint="Mô tả chủ đề chung rồi tìm kiếm với task AVS để nhận nhiều đoạn đa dạng." />;
  return (
    <WorkspaceTable
      head={
        <tr>
          <th scope="col">#</th>
          <th scope="col">Video</th>
          <th scope="col">Segment</th>
          <th scope="col" className="num">Frames</th>
          <th scope="col">Grade</th>
          <th scope="col">Cluster</th>
          <th scope="col" className="num">Score</th>
        </tr>
      }
    >
      {items.map((item) => (
        <tr key={item.segment_id}>
          <td className="num muted-cell">{item.rank}</td>
          <td className="truncate">{item.video_id}</td>
          <td className="truncate muted-cell">{item.segment_id}</td>
          <td className="num tabular">
            {item.start_frame}–{item.end_frame}
          </td>
          <td>
            <span className="grade" title={`relevance_grade ${item.relevance_grade}/3`}>
              {[0, 1, 2].map((index) => (
                <span key={index} className={index < item.relevance_grade ? "grade-dot is-on" : "grade-dot"} />
              ))}
            </span>
          </td>
          <td className="muted-cell">{item.cluster_id ?? "—"}</td>
          <td className="num tabular">{item.score.toFixed(4)}</td>
        </tr>
      ))}
    </WorkspaceTable>
  );
}
