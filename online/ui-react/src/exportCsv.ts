// Format nộp bài chính thức BTC AIC 2026 (KHÔNG phải CSV RFC4180 thông thường):
//   KIS/AVS/sequence : <video_id>, <frame_idx>                — không header, tối đa 100 dòng
//   VQA              : <video_id>, <frame_idx>, "<answer>"    — answer <=100 ký tự, giữ nguyên mọi dòng
// Port 1:1 từ online/ui/app.js (đã verify bằng Playwright thật trên UI cũ) — đổi công thức
// ở đây phải đổi cả bên app.js hoặc bỏ hẳn app.js, không để 2 nguồn lệch nhau.
import type { Task, TrayItem } from "./types";

export const MAX_SUBMISSION_ROWS = 100;

export class ExportValidationError extends Error {}

function frameIdxFromKeyframeId(keyframeId: string | null): string {
  const match = /_F(\d+)$/.exec(keyframeId ?? "");
  return match ? String(Number(match[1])) : "";
}

export interface BuildCsvOptions {
  task: Task;
  items: TrayItem[];
  vqaAnswer?: string;
}

/** Trả về nội dung CSV (chưa kèm newline cuối). Ném ExportValidationError nếu vi phạm
 * quy chế (quá 100 dòng, VQA thiếu/qua dài answer) — KHÔNG tự sửa/silent-truncate. */
export function buildSubmissionCsv({ task, items, vqaAnswer }: BuildCsvOptions): string {
  if (items.length === 0) {
    throw new ExportValidationError("Chưa chọn kết quả nào để xuất.");
  }
  if (items.length > MAX_SUBMISSION_ROWS) {
    throw new ExportValidationError(
      `Đang chọn ${items.length} dòng, vượt giới hạn ${MAX_SUBMISSION_ROWS} — bỏ bớt trước khi xuất.`
    );
  }
  let answer = "";
  if (task === "vqa") {
    answer = (vqaAnswer ?? "").trim();
    if (!answer) {
      throw new ExportValidationError("Nhập câu trả lời VQA trước khi xuất CSV.");
    }
    if (answer.length > 100) {
      throw new ExportValidationError("Câu trả lời VQA vượt quá 100 ký tự.");
    }
  }
  const lines = items.map((item) => {
    const frameIdx = frameIdxFromKeyframeId(item.best_keyframe_id);
    return task === "vqa"
      ? `${item.video_id}, ${frameIdx}, "${answer.replace(/"/g, '""')}"`
      : `${item.video_id}, ${frameIdx}`;
  });
  return lines.join("\n");
}

export function submissionFilename(task: Task): string {
  return `aic2026_${task}_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
}

export function downloadCsv(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
