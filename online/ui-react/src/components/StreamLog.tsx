import type { StreamEvent } from "../types";

function summarize(event: StreamEvent): string {
  switch (event.type) {
    case "search_started":
      return `search_started · task=${event.task}`;
    case "query_prepared":
      return `query_prepared · "${event.normalized_query}"`;
    case "branch_started":
      return `branch_started · ${event.execution_id}`;
    case "branch_completed":
      return `branch_completed · ${event.execution_id} · ${event.candidate_count} candidate · ${event.latency_ms}ms`;
    case "branch_failed":
      return `branch_failed · ${event.execution_id} · ${event.state}${event.warning ? ` — ${event.warning}` : ""}`;
    case "fusion_completed":
      return `fusion_completed · ${event.candidate_count} candidate`;
    case "rerank_completed":
      return `rerank_completed · ${event.stages.map((s) => `${s.stage}=${s.applied ? "ok" : "skip"}`).join(", ")}`;
    case "evidence_ready":
      return `evidence_ready · ${event.count} kết quả`;
    case "alignment_completed":
      return `alignment_completed · ${event.sequence_count} chuỗi${event.note ? ` — ${event.note}` : ""}`;
    case "search_completed":
      return `search_completed · query_id=${event.query_id}`;
    case "error":
      return `error · ${event.message}`;
  }
}

const CLASS_BY_TYPE: Partial<Record<StreamEvent["type"], string>> = {
  branch_failed: "issue-warning",
  error: "issue-error",
};

/** Log sự kiện SSE thô — chứng minh stream là THẬT (mỗi dòng tới đúng lúc
 * giai đoạn đó xong ở backend), không phải progress bar giả lập. */
export function StreamLog({ events }: { events: StreamEvent[] }) {
  if (events.length === 0) return null;
  return (
    <ol className="stream-log">
      {events.map((event, i) => (
        <li key={i} className={CLASS_BY_TYPE[event.type]}>
          {summarize(event)}
        </li>
      ))}
    </ol>
  );
}
