import type { BranchStatus } from "../types";

const STATE_LABEL: Record<BranchStatus["state"], string> = {
  success: "OK",
  disabled: "tắt",
  empty: "rỗng",
  timeout: "timeout",
  unavailable: "unavailable",
  failed: "failed",
};

const DEGRADED_STATES = new Set(["timeout", "unavailable", "failed"]);

/** UI phải thấy được branch nào lỗi (docs 01082026 §7 W2 DoD) — không phải
 * đoán từ số lượng kết quả tụt xuống. */
export function BranchStatusPanel({ statuses }: { statuses: BranchStatus[] }) {
  if (statuses.length === 0) return null;
  const degraded = statuses.filter((item) => DEGRADED_STATES.has(item.state));
  return (
    <div className="branch-status-panel">
      {degraded.length > 0 && (
        <p className="warning-text">
          ⚠ {degraded.length} branch gặp sự cố: {degraded.map((item) => item.execution_id).join(", ")}
        </p>
      )}
      <table className="mixing-table compact">
        <thead>
          <tr>
            <th>Execution</th>
            <th>Trạng thái</th>
            <th>Latency</th>
            <th>Candidates</th>
          </tr>
        </thead>
        <tbody>
          {statuses.map((item) => (
            <tr key={item.execution_id} className={DEGRADED_STATES.has(item.state) ? "row-degraded" : undefined}>
              <td>{item.execution_id}</td>
              <td title={item.warning ?? ""}>{STATE_LABEL[item.state]}</td>
              <td>{item.latency_ms}ms</td>
              <td>{item.candidate_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
