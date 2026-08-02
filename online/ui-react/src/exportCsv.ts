// Trước PR-10, CSV được build VÀ validate ở phía client (buildSubmissionCsv),
// tách rời khỏi backend — hai nguồn "đúng luật" có thể lệch nhau nếu chỉ sửa
// một bên. Từ PR-08, backend đã có online/competition/submission_builder.py +
// submission_validator.py với dữ liệu thật (video_frame_count, v.v.); client
// giờ chỉ gọi POST /v1/submissions/build rồi TẢI XUỐNG nội dung CSV nhận về
// (xem components/SubmissionBoard.tsx).
export function downloadCsv(content: string, filename: string): void {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function submissionFilename(task: string): string {
  return `aic2026_${task.toLowerCase()}_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
}
