import { useEffect, useState } from "react";
import type { QueryPlan, TaskType } from "../types";

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
  parsedEvents: QueryPlan["events"];
}

const TASK_OPTIONS: { value: TaskType; label: string; icon: string }[] = [
  { value: "TEXTUAL_KIS", label: "KIS", icon: "🔎" },
  { value: "QA", label: "QA", icon: "❓" },
  { value: "TRAKE", label: "TRAKE", icon: "🔗" },
  { value: "AVS", label: "AVS", icon: "🌐" },
];

const QUERY_MAX_LENGTH = 1500;
const MODE_KEY = "aic_query_mode";

function loadMode(): "simple" | "advanced" {
  return localStorage.getItem(MODE_KEY) === "advanced" ? "advanced" : "simple";
}

/** Query header — gọn tối đa cho Simple mode: chỉ task/query/top-k/search.
 * Advanced mở thêm kết nối (API base/token) + streaming, hai thứ hiếm khi
 * đổi và không nên chiếm chỗ mặc định (docs §7.2, yêu cầu "gọn gàng, tối
 * giản"). */
export function QueryStudio(props: QueryStudioProps) {
  const [mode, setMode] = useState<"simple" | "advanced">(loadMode);

  useEffect(() => {
    localStorage.setItem(MODE_KEY, mode);
  }, [mode]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (!props.submitting && props.query.trim()) props.onSubmit();
    }
  }

  return (
    <div className="query-card">
      <div className="query-toolbar">
        <div className="task-selector" role="tablist" aria-label="Chọn task">
          {TASK_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              role="tab"
              aria-selected={props.task === option.value}
              className={props.task === option.value ? "task-chip active" : "task-chip"}
              onClick={() => props.onTaskChange(option.value)}
            >
              <span aria-hidden="true">{option.icon}</span> {option.label}
            </button>
          ))}
        </div>

        <label className="topk-field">
          Top-K
          <input
            type="number"
            min={1}
            max={200}
            value={props.topK}
            onChange={(e) => props.onTopKChange(Number(e.target.value))}
          />
        </label>

        <div className="mode-toggle">
          <button type="button" className={mode === "simple" ? "mode-btn active" : "mode-btn"} onClick={() => setMode("simple")}>
            Simple
          </button>
          <button type="button" className={mode === "advanced" ? "mode-btn active" : "mode-btn"} onClick={() => setMode("advanced")}>
            Advanced
          </button>
        </div>
      </div>

      <div className="query-composer">
        <textarea
          value={props.query}
          maxLength={QUERY_MAX_LENGTH}
          onChange={(e) => props.onQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            props.task === "QA"
              ? "Đặt câu hỏi, vd: Có bao nhiêu xe máy va chạm?"
              : props.task === "TRAKE"
                ? 'Mô tả chuỗi sự kiện theo thứ tự, vd: "... (1) ...; (2) ...; (3) ..."'
                : "Mô tả cảnh cần tìm…"
          }
        />
        <div className="query-composer-footer">
          <span className="muted small">{props.query.length}/{QUERY_MAX_LENGTH}</span>
          <span className="muted small">Ctrl+Enter để tìm kiếm</span>
        </div>
      </div>

      {props.task === "TRAKE" && props.parsedEvents.length >= 2 && (
        <div className="event-chips">
          <span className="event-chip-count">{props.parsedEvents.length} sự kiện</span>
          {props.parsedEvents.map((event) => (
            <span key={event.event_idx} className="event-chip" title={event.text}>
              {event.event_idx + 1}. {event.text}
            </span>
          ))}
        </div>
      )}

      <div className="query-actions">
        <button type="button" onClick={props.onSubmit} disabled={props.submitting || !props.query.trim()}>
          {props.submitting ? "Đang chạy…" : "🔍 Tìm kiếm"}
        </button>
        <button type="button" className="secondary-btn" onClick={() => props.onQueryChange("")} disabled={!props.query}>
          Xoá
        </button>
        <label className="check">
          <input type="checkbox" checked={props.debug} onChange={(e) => props.onDebugChange(e.target.checked)} />
          Debug
        </label>
      </div>

      {mode === "advanced" && (
        <div className="advanced-fields">
          <label>
            API base
            <input value={props.apiBase} onChange={(e) => props.onApiBaseChange(e.target.value)} />
          </label>
          <label>
            API token (tùy chọn)
            <input value={props.apiToken} onChange={(e) => props.onApiTokenChange(e.target.value)} type="password" />
          </label>
          <label className="check">
            <input type="checkbox" checked={props.streaming} onChange={(e) => props.onStreamingChange(e.target.checked)} />
            Stream (SSE)
          </label>
          <button type="button" className="secondary-btn" onClick={props.onHealthCheck}>
            Kiểm tra server
          </button>
        </div>
      )}
    </div>
  );
}
