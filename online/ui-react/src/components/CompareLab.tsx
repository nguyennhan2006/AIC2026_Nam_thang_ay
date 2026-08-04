import { useState } from "react";
import { TriangleAlert } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { ApiError, getSearchSession, replaySearchSession } from "../api";
import type { SearchExecutionTrace } from "../types";

function TraceCard({
  trace,
  onReplay,
}: {
  trace: SearchExecutionTrace | null;
  onReplay: () => void;
}) {
  if (!trace) return <p className="muted">Nhập session_id rồi bấm Tải.</p>;
  const degraded = trace.branch_status.filter((item) =>
    ["timeout", "unavailable", "failed"].includes(item.state)
  );
  return (
    <div className="trace-card">
      <p>
        <strong>{trace.session_id}</strong> · {trace.task} · {trace.status}
      </p>
      <p className="muted small">query: {trace.raw_request.query}</p>
      <p className="muted small">
        took {trace.took_ms.toFixed(1)}ms · dataset {trace.dataset_version ?? "?"} · replay_count {trace.replay_count}
      </p>
      {degraded.length > 0 && (
        <p className="branch-status-warn">
          <TriangleAlert size={12} />
          <span>
            {degraded.length} nhánh degraded lúc chạy: {degraded.map((d) => d.execution_id).join(", ")}
          </span>
        </p>
      )}
      {trace.warnings.length > 0 && (
        <ul className="issue-list">
          {trace.warnings.map((w, i) => (
            <li key={i} className="issue-warning">
              {w}
            </li>
          ))}
        </ul>
      )}
      <button type="button" onClick={onReplay}>
        Replay session này
      </button>
    </div>
  );
}

/** Compare Lab — so sánh trace của hai session (thường: session gốc vs.
 * session replay của nó, đã có replayed_from trỏ ngược). Không diff sâu kết
 * quả (đó là việc của Results Explorer) — mục đích ở đây là so cấu hình,
 * trạng thái branch và thời gian chạy giữa hai lần search. */
export function CompareLab({ apiConfig }: { apiConfig: ApiClientConfig }) {
  const [leftId, setLeftId] = useState("");
  const [rightId, setRightId] = useState("");
  const [left, setLeft] = useState<SearchExecutionTrace | null>(null);
  const [right, setRight] = useState<SearchExecutionTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load(id: string, setter: (trace: SearchExecutionTrace) => void) {
    setError(null);
    try {
      setter(await getSearchSession(apiConfig, id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  }

  async function replay(id: string, setter: (trace: SearchExecutionTrace) => void, setId: (id: string) => void) {
    setError(null);
    try {
      const response = await replaySearchSession(apiConfig, id);
      setId(response.query_id);
      setter(await getSearchSession(apiConfig, response.query_id));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  }

  return (
    <div className="compare-lab">
      {error && <p className="muted">Lỗi: {error}</p>}
      <div className="compare-grid">
        <div>
          <label>
            Session A (session_id)
            <input value={leftId} onChange={(e) => setLeftId(e.target.value)} placeholder="query_id của lần search trước" />
          </label>
          <button type="button" onClick={() => load(leftId, setLeft)} disabled={!leftId.trim()}>
            Tải
          </button>
          <TraceCard trace={left} onReplay={() => replay(leftId, setLeft, setLeftId)} />
        </div>
        <div>
          <label>
            Session B (session_id)
            <input value={rightId} onChange={(e) => setRightId(e.target.value)} placeholder="query_id của lần search khác" />
          </label>
          <button type="button" onClick={() => load(rightId, setRight)} disabled={!rightId.trim()}>
            Tải
          </button>
          <TraceCard trace={right} onReplay={() => replay(rightId, setRight, setRightId)} />
        </div>
      </div>
    </div>
  );
}
