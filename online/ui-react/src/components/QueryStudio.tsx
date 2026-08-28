import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { Braces, HelpCircle, Link2, Radio, Search, Sparkles, X } from "lucide-react";
import type { QueryPlan, TaskType } from "../types";
import type { KisConstraints } from "../types";
import { Badge, Button, Checkbox, NumericInput, SegmentedControl, TextField } from "../ui";
import type { SegmentedOption } from "../ui";

export interface QueryStudioProps {
  apiBase: string;
  onApiBaseChange: (value: string) => void;
  apiToken: string;
  onApiTokenChange: (value: string) => void;
  task: TaskType;
  onTaskChange: (task: TaskType) => void;
  query: string;
  onQueryChange: (value: string) => void;
  kisMode: "simple" | "structured";
  onKisModeChange: (value: "simple" | "structured") => void;
  kisConstraints: Record<keyof KisConstraints, string>;
  onKisConstraintsChange: (value: Record<keyof KisConstraints, string>) => void;
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
  /** Nội dung cạnh ô truy vấn (dataset stats) — nằm cùng hàng để không phải
   * dựng thêm một cột `aside` riêng chỉ để rồi phải gỡ ra ở màn hẹp. */
  aside?: ReactNode;
}

const TASK_OPTIONS: SegmentedOption<TaskType>[] = [
  { value: "TEXTUAL_KIS", label: "KIS", icon: <Search size={13} /> },
  { value: "QA", label: "QA", icon: <HelpCircle size={13} /> },
  { value: "TRAKE", label: "TRAKE", icon: <Link2 size={13} /> },
  { value: "AVS", label: "AVS", icon: <Sparkles size={13} /> },
];

const TASK_HINTS: Record<TaskType, string> = {
  TEXTUAL_KIS: "Mô tả khung hình cần tìm — càng cụ thể về vật thể, chữ trên màn hình, bối cảnh càng tốt.",
  QA: "Đặt câu hỏi về nội dung video, ví dụ: “Biển cảnh báo ghi gì?”",
  TRAKE: "Mô tả chuỗi sự kiện theo thứ tự, phân tách bằng dấu chấm phẩy hoặc “sau đó”.",
  AVS: "Mô tả chủ đề chung — hệ thống trả về nhiều đoạn liên quan thay vì một khung hình duy nhất.",
};

const QUERY_MAX_LENGTH = 4000;
const MODE_KEY = "aic_query_mode";
const KIS_FIELDS: { key: keyof KisConstraints; label: string; placeholder: string }[] = [
  { key: "visual", label: "Visual", placeholder: "mô tả hình ảnh chính cần tìm" },
  { key: "ocr", label: "OCR", placeholder: "chữ xuất hiện trên màn hình" },
  { key: "asr", label: "ASR", placeholder: "lời nói hoặc âm thanh cần khớp" },
  { key: "must", label: "Must-have", placeholder: "điều bắt buộc, mỗi dòng một ý" },
  { key: "negative", label: "Negative", placeholder: "điều không được có" },
];
const EMPTY_KIS_CONSTRAINTS: Record<keyof KisConstraints, string> = {
  visual: "",
  ocr: "",
  asr: "",
  must: "",
  negative: "",
};

function loadMode(): "simple" | "advanced" {
  if (typeof localStorage === "undefined") return "simple";
  return localStorage.getItem(MODE_KEY) === "advanced" ? "advanced" : "simple";
}

/** Query composer. Task/Top-K/Mode nằm ở hàng `studio-controls` phía trên
 * (cấp cao hơn), ô nhập + hành động nằm trong card này — tách hai cấp để
 * không có nhiều hàng pill cùng trọng số thị giác. */
export function QueryStudio(props: QueryStudioProps) {
  const [mode, setMode] = useState<"simple" | "advanced">(loadMode);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const kisConstraints = props.kisConstraints ?? EMPTY_KIS_CONSTRAINTS;

  useEffect(() => {
    localStorage.setItem(MODE_KEY, mode);
  }, [mode]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [props.query]);

  const structuredText = Object.entries(kisConstraints)
    .filter(([key]) => key !== "negative")
    .map(([, value]) => value)
    .join("\n")
    .trim();
  const canSubmit = !props.submitting && (
    props.task === "TEXTUAL_KIS" && props.kisMode === "structured"
      ? structuredText.length > 0
      : props.query.trim().length > 0
  );

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      if (canSubmit) props.onSubmit();
    }
  }

  return (
    <>
      <div className="studio-controls">
        <div className="control-group">
          <span className="control-group-label">Task</span>
          <SegmentedControl ariaLabel="Chọn loại nhiệm vụ" options={TASK_OPTIONS} value={props.task} onChange={props.onTaskChange} />
        </div>

        {props.task === "TEXTUAL_KIS" && (
          <div className="control-group">
            <span className="control-group-label">KIS</span>
            <SegmentedControl
              ariaLabel="Chọn mode KIS"
              value={props.kisMode}
              onChange={props.onKisModeChange}
              options={[
                { value: "simple", label: "Simple" },
                { value: "structured", label: "Structured" },
              ]}
            />
          </div>
        )}

        <div className="control-group">
          <span className="control-group-label">Top-K</span>
          <NumericInput
            value={props.topK}
            min={1}
            max={200}
            ariaLabel="Số kết quả tối đa"
            width="68px"
            onChange={(value) => props.onTopKChange(value ?? 20)}
          />
        </div>

        <div className="control-group studio-controls-spacer">
          <SegmentedControl
            ariaLabel="Chế độ hiển thị"
            value={mode}
            onChange={setMode}
            options={[
              { value: "simple", label: "Simple" },
              { value: "advanced", label: "Advanced" },
            ]}
          />
        </div>
      </div>

      <div className="query-row">
      <div className="query-card">
        <div className="query-main">
          <div className="query-head">
            <span className="eyebrow">Query</span>
            {props.task === "TRAKE" && props.parsedEvents.length >= 2 && (
              <Badge tone="accent">{props.parsedEvents.length} bước</Badge>
            )}
            {props.streaming && (
              <Badge tone="neutral">
                <Radio size={9} /> stream
              </Badge>
            )}
          </div>

          {props.task === "TEXTUAL_KIS" && props.kisMode === "structured" ? (
            <div className="structured-kis-grid">
              {KIS_FIELDS.map((field) => (
                <label key={field.key} className="structured-kis-field">
                  <span className="field-label">{field.label}</span>
                  <textarea
                    className="structured-kis-textarea"
                    rows={field.key === "negative" ? 2 : 3}
                    value={kisConstraints[field.key]}
                    placeholder={field.placeholder}
                    aria-label={`Structured KIS ${field.label}`}
                    onChange={(event) => props.onKisConstraintsChange({
                      ...kisConstraints,
                      [field.key]: event.target.value,
                    })}
                    onKeyDown={handleKeyDown}
                  />
                </label>
              ))}
            </div>
          ) : (
            <textarea
              ref={textareaRef}
              className="query-textarea"
              rows={1}
              value={props.query}
              maxLength={QUERY_MAX_LENGTH}
              placeholder={TASK_HINTS[props.task]}
              aria-label="Nội dung truy vấn"
              onChange={(event) => props.onQueryChange(event.target.value)}
              onKeyDown={handleKeyDown}
            />
          )}

          {props.task === "TRAKE" && props.parsedEvents.length >= 2 && (
            <div className="query-chips">
              {props.parsedEvents.map((event) => (
                <span key={event.event_idx} className="event-chip" title={event.text}>
                  <span className="event-chip-index">{event.event_idx + 1}</span>
                  <span className="truncate">{event.text}</span>
                </span>
              ))}
            </div>
          )}

          <div className="query-footer">
            <span className="query-hint">Ctrl+Enter để tìm</span>
            <Checkbox checked={props.debug} onChange={props.onDebugChange} label="Debug" />
            <span className="query-counter tabular">
              {props.task === "TEXTUAL_KIS" && props.kisMode === "structured"
                ? `${structuredText.length}/${QUERY_MAX_LENGTH}`
                : `${props.query.length}/${QUERY_MAX_LENGTH}`}
            </span>
          </div>
        </div>

        <div className="query-actions">
          <Button variant="primary" icon={<Search size={14} />} loading={props.submitting} disabled={!canSubmit} onClick={props.onSubmit} block>
            {props.submitting ? "Đang tìm" : "Tìm kiếm"}
          </Button>
          <Button
            variant="ghost"
            icon={<X size={14} />}
            disabled={
              props.task === "TEXTUAL_KIS" && props.kisMode === "structured"
                ? !Object.values(kisConstraints).some((value) => value.trim())
                : !props.query
            }
            onClick={() => {
              if (props.task === "TEXTUAL_KIS" && props.kisMode === "structured") {
                props.onKisConstraintsChange(EMPTY_KIS_CONSTRAINTS);
              } else {
                props.onQueryChange("");
              }
            }}
            block
          >
            Xoá
          </Button>
        </div>
      </div>
        {props.aside}
      </div>

      {mode === "advanced" && (
        <div className="advanced-drawer">
          <TextField label="API base" value={props.apiBase} onChange={props.onApiBaseChange} placeholder="http://localhost:8000" />
          <TextField label="API token" value={props.apiToken} onChange={props.onApiTokenChange} type="password" placeholder="(tuỳ chọn)" />
          <div className="field">
            <span className="field-label">Tuỳ chọn</span>
            <Checkbox checked={props.streaming} onChange={props.onStreamingChange} label="Stream kết quả (SSE)" />
          </div>
          <div className="field">
            <span className="field-label">Kết nối</span>
            <Button size="sm" variant="secondary" icon={<Braces size={13} />} onClick={props.onHealthCheck}>
              Kiểm tra server
            </Button>
          </div>
        </div>
      )}
    </>
  );
}
