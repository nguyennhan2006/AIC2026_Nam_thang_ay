import type { TaskType } from "../types";

export interface QueryStudioProps {
  apiBase: string;
  onApiBaseChange: (value: string) => void;
  apiToken: string;
  onApiTokenChange: (value: string) => void;
  task: TaskType;
  onTaskChange: (task: TaskType) => void;
  query: string;
  onQueryChange: (value: string) => void;
  topK: number;
  onTopKChange: (value: number) => void;
  debug: boolean;
  onDebugChange: (value: boolean) => void;
  streaming: boolean;
  onStreamingChange: (value: boolean) => void;
  onSubmit: () => void;
  onHealthCheck: () => void;
  submitting: boolean;
}

const TASK_LABELS: Record<TaskType, string> = {
  TEXTUAL_KIS: "Textual KIS",
  QA: "Q&A",
  TRAKE: "TRAKE",
  AVS: "AVS",
};

export function QueryStudio(props: QueryStudioProps) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        props.onSubmit();
      }}
    >
      <label>
        API base
        <input value={props.apiBase} onChange={(e) => props.onApiBaseChange(e.target.value)} />
      </label>
      <label>
        API token (tùy chọn)
        <input value={props.apiToken} onChange={(e) => props.onApiTokenChange(e.target.value)} type="password" />
      </label>
      <label>
        Task
        <select value={props.task} onChange={(e) => props.onTaskChange(e.target.value as TaskType)}>
          {(Object.keys(TASK_LABELS) as TaskType[]).map((task) => (
            <option key={task} value={task}>
              {TASK_LABELS[task]}
            </option>
          ))}
        </select>
      </label>
      <label className="query-label wide">
        Câu truy vấn
        <textarea
          value={props.query}
          onChange={(e) => props.onQueryChange(e.target.value)}
          placeholder={
            props.task === "QA"
              ? "Đặt câu hỏi, vd: Có bao nhiêu xe máy va chạm?"
              : "Mô tả cảnh cần tìm…"
          }
        />
      </label>
      <label>
        Top-k
        <input type="number" min={1} max={200} value={props.topK} onChange={(e) => props.onTopKChange(Number(e.target.value))} />
      </label>
      <label className="check">
        <input type="checkbox" checked={props.debug} onChange={(e) => props.onDebugChange(e.target.checked)} />
        Debug (kèm query_plan)
      </label>
      <label className="check">
        <input type="checkbox" checked={props.streaming} onChange={(e) => props.onStreamingChange(e.target.checked)} />
        Stream (SSE)
      </label>
      <button type="submit" disabled={props.submitting || !props.query.trim()}>
        {props.submitting ? "Đang chạy…" : "Tìm kiếm"}
      </button>
      <button type="button" onClick={props.onHealthCheck}>
        Kiểm tra server
      </button>
    </form>
  );
}
