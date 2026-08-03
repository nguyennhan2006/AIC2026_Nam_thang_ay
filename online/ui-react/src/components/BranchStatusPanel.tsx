import { TriangleAlert } from "lucide-react";
import type { BranchStatus } from "../types";

const STATE_LABEL: Record<BranchStatus["state"], string> = {
  success: "ok",
  disabled: "tắt",
  empty: "rỗng",
  timeout: "timeout",
  unavailable: "n/a",
  failed: "failed",
};

const DEGRADED_STATES = new Set<BranchStatus["state"]>(["timeout", "unavailable", "failed"]);

/** UI phải thấy được branch nào lỗi, không phải đoán từ việc số kết quả tụt
 * xuống (docs 01082026 §7 W2 DoD). */
export function BranchStatusPanel({ statuses }: { statuses: BranchStatus[] }) {
  if (statuses.length === 0) return null;
  const degraded = statuses.filter((item) => DEGRADED_STATES.has(item.state));

  return (
    <div className="branch-status">
      {degraded.length > 0 && (
        <p className="branch-status-warn">
          <TriangleAlert size={12} />
          <span>
            {degraded.length} nhánh gặp sự cố: {degraded.map((item) => item.execution_id).join(", ")}
          </span>
        </p>
      )}
      <div className="branch-list">
        {statuses.map((item) => (
          <div
            key={item.execution_id}
            className={DEGRADED_STATES.has(item.state) ? "branch-item is-degraded" : "branch-item"}
            title={item.warning ?? item.execution_id}
          >
            <span className="truncate">{item.execution_id}</span>
            <span className="branch-state">{STATE_LABEL[item.state]}</span>
            <span className="num tabular">{item.latency_ms}ms</span>
            <span className="num tabular">{item.candidate_count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
