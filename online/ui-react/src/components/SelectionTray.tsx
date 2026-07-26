import { buildSubmissionCsv, downloadCsv, ExportValidationError, MAX_SUBMISSION_ROWS, submissionFilename } from "../exportCsv";
import type { Task, TrayItem } from "../types";

export interface SelectionTrayProps {
  task: Task;
  selection: Map<string, TrayItem>;
  onRemove: (sceneId: string) => void;
  onClear: () => void;
  onRefine: () => void;
  trayAnswer: string;
  onTrayAnswerChange: (value: string) => void;
  onStatus: (message: string) => void;
}

export function SelectionTray({
  task,
  selection,
  onRemove,
  onClear,
  onRefine,
  trayAnswer,
  onTrayAnswerChange,
  onStatus,
}: SelectionTrayProps) {
  const items = [...selection.values()];
  const hasItems = items.length > 0;
  const overLimit = items.length > MAX_SUBMISSION_ROWS;

  function handleExport() {
    try {
      const csv = buildSubmissionCsv({ task, items, vqaAnswer: trayAnswer });
      downloadCsv(csv, submissionFilename(task));
      onStatus(`Đã xuất ${items.length} dòng (${task.toUpperCase()}).`);
    } catch (error) {
      if (error instanceof ExportValidationError) {
        onStatus(error.message);
      } else {
        throw error;
      }
    }
  }

  return (
    <aside id="tray">
      <h2>
        Đã chọn{" "}
        <span id="tray-count" className={overLimit ? "over-limit" : ""}>
          ({items.length}/{MAX_SUBMISSION_ROWS})
        </span>
      </h2>
      <p className="tray-hint">
        Tick vào kết quả để đưa vào đây, rồi lọc lại hoặc xuất CSV nộp bài. Mỗi file CSV chỉ nộp cho MỘT câu truy vấn
        (đúng quy chế BTC) — xuất xong hãy bấm "Xoá hết" trước khi làm câu tiếp theo.
      </p>
      {task === "vqa" && (
        <label id="tray-answer-wrap" className="tray-answer">
          Câu trả lời VQA (áp dụng cho mọi dòng khi xuất)
          <input
            id="tray-answer"
            type="text"
            maxLength={100}
            placeholder="Tối đa 100 ký tự, VI hoặc EN"
            value={trayAnswer}
            onChange={(e) => onTrayAnswerChange(e.target.value)}
          />
        </label>
      )}
      <ol id="tray-list">
        {items.map((item) => (
          <li key={item.scene_id}>
            <div>
              <strong>{item.video_id}</strong>
              <span>
                {item.scene_id} · {(item.best_timestamp_sec ?? item.start_sec).toFixed(2)}s
              </span>
            </div>
            <button type="button" className="tray-remove" aria-label="Bỏ khỏi danh sách" onClick={() => onRemove(item.scene_id)}>
              ×
            </button>
          </li>
        ))}
      </ol>
      <div className="tray-actions">
        <button id="tray-refine" type="button" disabled={!hasItems} onClick={onRefine}>
          Tìm lại chỉ trong các video đã chọn
        </button>
        <button id="tray-export" type="button" disabled={!hasItems || overLimit} onClick={handleExport}>
          Xuất CSV nộp bài
        </button>
        <button id="tray-clear" type="button" disabled={!hasItems} onClick={onClear}>
          Xoá hết
        </button>
      </div>
    </aside>
  );
}
