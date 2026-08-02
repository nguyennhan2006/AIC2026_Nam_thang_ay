import { useState } from "react";
import type { ApiClientConfig } from "../api";
import { ApiError, buildSubmission } from "../api";
import { downloadCsv, submissionFilename } from "../exportCsv";
import type { AvsResultItem, KisResultItem, QaResultItem, SubmissionBuildResponse, TaskType, TrakeResultItem } from "../types";
import { zoneForRank } from "../types";

export interface SubmissionBoardProps {
  apiConfig: ApiClientConfig;
  task: TaskType;
  kis: KisResultItem[];
  qa: QaResultItem[];
  trake: TrakeResultItem[];
  avs: AvsResultItem[];
}

/** Submission Board — build CSV đúng format BTC qua backend thật
 * (online/competition/submission_builder.py + submission_validator.py), hiện
 * 5 vùng ranking (1 / 2–5 / 6–20 / 21–50 / 51–100) và issue kèm mức độ. */
export function SubmissionBoard({ apiConfig, task, kis, qa, trake, avs }: SubmissionBoardProps) {
  const [result, setResult] = useState<SubmissionBuildResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (task === "AVS") {
    return (
      <div className="submission-board">
        <p className="muted">
          AVS là task nội bộ mở rộng, không có format nộp bài chính thức của BTC (docs 01082026 §17).
        </p>
        {avs.length > 0 && (
          <p className="muted small">
            {avs.length} segment · phân bố grade:{" "}
            {[0, 1, 2, 3].map((grade) => `${grade}★=${avs.filter((item) => item.relevance_grade === grade).length}`).join(", ")}
          </p>
        )}
      </div>
    );
  }

  const zoneCounts = new Map<string, number>();
  const rankedItems = task === "TEXTUAL_KIS" ? kis : task === "QA" ? qa : trake;
  for (const item of rankedItems) {
    const zone = zoneForRank(item.rank);
    zoneCounts.set(zone, (zoneCounts.get(zone) ?? 0) + 1);
  }

  async function runBuild() {
    setLoading(true);
    setError(null);
    try {
      const body =
        task === "TEXTUAL_KIS" ? { task, kis } : task === "QA" ? { task, qa } : { task, trake };
      const response = await buildSubmission(apiConfig, body);
      setResult(response);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="submission-board">
      <div className="zone-summary">
        {["rank_1", "ranks_2_5", "ranks_6_20", "ranks_21_50", "ranks_51_100"].map((zone) => (
          <span key={zone} className="zone-chip">
            {zone}: {zoneCounts.get(zone) ?? 0}
          </span>
        ))}
      </div>
      <button type="button" onClick={runBuild} disabled={loading || rankedItems.length === 0}>
        {loading ? "Đang build…" : `Build submission (${rankedItems.length} dòng)`}
      </button>
      {error && <p className="muted">Lỗi: {error}</p>}
      {result && (
        <div className="submission-result">
          <p className={result.has_errors ? "warning-text" : ""}>
            {result.item_count} dòng · {result.has_errors ? "CÓ LỖI, không nên nộp" : "hợp lệ"}
          </p>
          {result.issues.length > 0 && (
            <ul className="issue-list">
              {result.issues.map((issue, i) => (
                <li key={i} className={issue.severity === "error" ? "issue-error" : "issue-warning"}>
                  [{issue.severity}] {issue.code}: {issue.message}
                  {issue.row_index != null && ` (dòng ${issue.row_index})`}
                </li>
              ))}
            </ul>
          )}
          <pre className="csv-preview">{result.csv}</pre>
          <button
            type="button"
            onClick={() => downloadCsv(result.csv, submissionFilename(task))}
            disabled={result.has_errors}
          >
            Tải CSV xuống
          </button>
        </div>
      )}
    </div>
  );
}
