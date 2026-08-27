import type { StreamEvent } from "../types";

/** Query mà MỖI ENGINE nhận — bảng, không phải một dòng.
 *
 * Bốn engine nhận bốn chuỗi khác nhau (Query Routing V2), và khi kết quả sai
 * thì câu hỏi đầu tiên luôn là "engine nào đã đem chuỗi nào đi tìm". Nhồi tất
 * cả vào một dòng log thì đọc không ra. */
function QueryBundle({ event }: { event: Extract<StreamEvent, { type: "query_bundle" }> }) {
  const rows: [string, string][] = [
    ["dense_visual", event.visual_query],
    ["  ↳ bản EN", event.visual_query_en],
    ["bm25_caption", event.caption_query],
    ["bm25_ocr", event.ocr_query],
    ["bm25_asr", event.asr_query],
  ];
  return (
    <div className="stream-bundle">
      <div className="stream-bundle-head">
        <span className={`stream-tag stream-tag-${event.source}`}>{event.source}</span>
        <span>
          intent={event.intent} · answer={event.answer_type} · complexity={event.complexity}
        </span>
        {event.llm && (
          <span className="stream-dim">
            {event.llm.prompt} · ghi đè: {event.llm.fields.join(", ")}
          </span>
        )}
      </div>
      <table className="stream-bundle-table">
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th>{label}</th>
              <td className={value ? undefined : "stream-dim"}>{value || "(rỗng)"}</td>
            </tr>
          ))}
          {event.events.length > 0 &&
            event.events.map((text, i) => (
              <tr key={`ev-${i}`}>
                <th>event[{i}]</th>
                <td>{text}</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

function summarize(event: StreamEvent): string {
  switch (event.type) {
    case "search_started":
      return `search_started · task=${event.task}`;
    case "query_prepared":
      return `query_prepared · "${event.normalized_query}"`;
    case "trake_step":
      return `trake_step · [${event.step}/${event.total_steps}] "${event.text}"`;
    case "branch_started":
      // Hiện luôn chuỗi nhánh này đem đi tìm, kèm trường nguồn. `normalized_query`
      // ở đây nghĩa là nhánh CHƯA được nối vào Query Routing V2 — nó nhận query
      // thô, kể cả phần đánh số "(1) (2)" của đề bài.
      return event.query_sent
        ? `branch_started · ${event.execution_id} ← ${event.query_source} · "${event.query_sent}"`
        : `branch_started · ${event.execution_id}`;
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
    case "query_bundle":
      return ""; // render riêng bằng <QueryBundle>
  }
}

const CLASS_BY_TYPE: Partial<Record<StreamEvent["type"], string>> = {
  branch_failed: "issue-warning",
  error: "issue-error",
  trake_step: "stream-step",
};

/** Log sự kiện SSE thô — chứng minh stream là THẬT (mỗi dòng tới đúng lúc
 * giai đoạn đó xong ở backend), không phải progress bar giả lập. */
export function StreamLog({ events }: { events: StreamEvent[] }) {
  if (events.length === 0) return null;
  return (
    <ol className="stream-log">
      {events.map((event, i) => (
        <li key={i} className={CLASS_BY_TYPE[event.type]}>
          {event.type === "query_bundle" ? <QueryBundle event={event} /> : summarize(event)}
        </li>
      ))}
    </ol>
  );
}
