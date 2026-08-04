import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDown, ArrowUp, FileDown, ImageOff, Play, RotateCcw, X } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { ApiError, buildSubmission, mediaUrl } from "../api";
import { downloadCsv, submissionFilename } from "../exportCsv";
import type {
  AvsResultItem,
  KisResultItem,
  QaResultItem,
  SearchHit,
  SubmissionBuildResponse,
  TaskType,
  TrakeResultItem,
} from "../types";
import { zoneForRank } from "../types";
import { Badge, Button, EmptyState, IconButton, InlineError } from "../ui";

export interface SubmissionBoardProps {
  apiConfig: ApiClientConfig;
  task: TaskType;
  kis: KisResultItem[];
  qa: QaResultItem[];
  trake: TrakeResultItem[];
  avs: AvsResultItem[];
  /** Dùng để tra `video_path` và quy đổi frame -> giây. Không gọi thêm API. */
  results?: SearchHit[];
}

/** Một dòng sẽ nộp, đã tách khỏi kiểu của từng task để bảng dùng chung. */
interface Row {
  key: string;
  videoId: string;
  frameIdx: number;
  sceneId: string | null;
  answer?: string;
  frameIds?: number[];
}

function toRows(task: TaskType, kis: KisResultItem[], qa: QaResultItem[], trake: TrakeResultItem[]): Row[] {
  if (task === "TEXTUAL_KIS") {
    return kis.map((item) => ({
      key: `${item.video_id}-${item.frame_idx}`,
      videoId: item.video_id, frameIdx: item.frame_idx, sceneId: item.scene_id,
    }));
  }
  if (task === "QA") {
    return qa.map((item) => ({
      key: `${item.video_id}-${item.frame_idx}-${item.canonical_answer}`,
      videoId: item.video_id, frameIdx: item.frame_idx, sceneId: item.scene_id,
      answer: item.answer,
    }));
  }
  return trake.map((item) => ({
    key: `${item.video_id}-${item.rank}`,
    videoId: item.video_id, frameIdx: item.frame_ids[0] ?? 0,
    sceneId: item.steps[0]?.scene_id ?? null, frameIds: item.frame_ids,
  }));
}

/** Quy đổi frame -> giây bằng chính scene chứa nó.
 *
 * `SearchHit` mang đủ `start_frame/end_frame_exclusive` và `start_sec/end_sec`,
 * nên fps suy ra được tại chỗ — không cần thêm endpoint, và không phải giả
 * định 30fps (video khác fps sẽ tua sai chỗ).
 */
function seekSecondsFor(hit: SearchHit, frameIdx: number): number {
  const frameSpan = hit.end_frame_exclusive - hit.start_frame;
  const secondSpan = hit.end_sec - hit.start_sec;
  if (frameSpan <= 0 || secondSpan <= 0) return hit.start_sec;
  const fps = frameSpan / secondSpan;
  return hit.start_sec + (frameIdx - hit.start_frame) / fps;
}

export function SubmissionBoard({ apiConfig, task, kis, qa, trake, avs, results = [] }: SubmissionBoardProps) {
  const [result, setResult] = useState<SubmissionBuildResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<Row[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [videoError, setVideoError] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const sourceRows = useMemo(() => toRows(task, kis, qa, trake), [task, kis, qa, trake]);

  // Kết quả mới về thì thứ tự thủ công cũ không còn ý nghĩa — nạp lại từ đầu.
  useEffect(() => {
    setRows(sourceRows);
    setSelected(null);
    setResult(null);
  }, [sourceRows]);

  const hitByScene = useMemo(() => {
    const map = new Map<string, SearchHit>();
    for (const hit of results) if (!map.has(hit.scene_id)) map.set(hit.scene_id, hit);
    return map;
  }, [results]);

  const selectedRow = rows.find((row) => row.key === selected) ?? null;
  const selectedHit = selectedRow?.sceneId ? hitByScene.get(selectedRow.sceneId) ?? null : null;
  const seekTo = selectedRow && selectedHit ? seekSecondsFor(selectedHit, selectedRow.frameIdx) : null;

  useEffect(() => setVideoError(false), [selectedHit?.video_path]);

  // Đặt `currentTime` TRƯỚC khi metadata tải xong thì trình duyệt bỏ qua —
  // video vẫn nằm ở 0:00 dù đã tính đúng mốc giây. Phải thử ngay (khi video
  // đã sẵn sàng từ lần chọn trước) VÀ nghe `loadedmetadata` cho lần tải mới.
  useEffect(() => {
    const element = videoRef.current;
    if (!element || seekTo == null) return;
    const target = Math.max(seekTo, 0);
    const apply = () => {
      element.currentTime = target;
    };
    if (element.readyState >= 1) apply();
    element.addEventListener("loadedmetadata", apply);
    return () => element.removeEventListener("loadedmetadata", apply);
  }, [seekTo, selectedHit?.video_path]);

  function move(index: number, delta: number) {
    setRows((current) => {
      const next = [...current];
      const target = index + delta;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
    setResult(null);
  }

  function remove(index: number) {
    setRows((current) => current.filter((_, position) => position !== index));
    setResult(null);
  }

  async function runBuild() {
    setLoading(true);
    setError(null);
    try {
      // Gửi lên theo ĐÚNG thứ tự đang hiển thị, đánh lại rank 1..N. BTC cho
      // phép tự quyết thứ tự, nên thứ tự thủ công phải thắng thứ tự của model.
      const order = new Map(rows.map((row, index) => [row.key, index]));
      const pick = <T extends { rank: number }>(items: T[], keyOf: (item: T) => string): T[] =>
        items
          .filter((item) => order.has(keyOf(item)))
          .sort((a, b) => order.get(keyOf(a))! - order.get(keyOf(b))!)
          .map((item, index) => ({ ...item, rank: index + 1 }));

      const body =
        task === "TEXTUAL_KIS"
          ? { task, kis: pick(kis, (i) => `${i.video_id}-${i.frame_idx}`) }
          : task === "QA"
            ? { task, qa: pick(qa, (i) => `${i.video_id}-${i.frame_idx}-${i.canonical_answer}`) }
            : { task, trake: pick(trake, (i) => `${i.video_id}-${i.rank}`) };
      setResult(await buildSubmission(apiConfig, body));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  if (task === "AVS") {
    return (
      <div className="submission-board">
        <p className="muted">
          AVS là task nội bộ mở rộng, không có format nộp bài chính thức của BTC (docs 01082026 §17).
        </p>
        {avs.length > 0 && (
          <p className="muted small">
            {avs.length} segment · {[0, 1, 2, 3].map((g) => `grade ${g}: ${avs.filter((i) => i.relevance_grade === g).length}`).join(" · ")}
          </p>
        )}
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <EmptyState
        icon={<FileDown size={20} />}
        title="Chưa có dòng nào để nộp"
        description="Chạy tìm kiếm trước; kết quả của task đang chọn sẽ hiện ở đây để sắp thứ tự rồi build CSV."
      />
    );
  }

  const zoneCounts = new Map<string, number>();
  rows.forEach((_row, index) => {
    const zone = zoneForRank(index + 1);
    zoneCounts.set(zone, (zoneCounts.get(zone) ?? 0) + 1);
  });
  const reordered = rows.length !== sourceRows.length || rows.some((row, i) => row.key !== sourceRows[i]?.key);

  return (
    <div className="submission-board">
      <div className="submission-toolbar">
        <span className="eyebrow">{rows.length} dòng</span>
        {reordered && <Badge tone="warning">đã sắp lại tay</Badge>}
        {reordered && (
          <IconButton
            icon={<RotateCcw size={14} />}
            label="Khôi phục thứ tự của hệ thống"
            size="sm"
            variant="control"
            onClick={() => { setRows(sourceRows); setResult(null); }}
          />
        )}
        <div className="submission-zones">
          {["rank_1", "ranks_2_5", "ranks_6_20", "ranks_21_50", "ranks_51_100"].map((zone) => (
            <span key={zone} className="zone-chip">{zone.replace("ranks_", "").replace("rank_", "")}: {zoneCounts.get(zone) ?? 0}</span>
          ))}
        </div>
        <Button variant="primary" size="sm" loading={loading} onClick={runBuild}>
          Build CSV
        </Button>
      </div>

      <div className="submission-split">
        <ol className="submission-list scroll-y">
          {rows.map((row, index) => (
            <li
              key={row.key}
              className={row.key === selected ? "submission-row is-selected" : "submission-row"}
            >
              <span className="submission-rank tabular">{index + 1}</span>
              <button
                type="button"
                className="submission-main"
                onClick={() => setSelected(row.key === selected ? null : row.key)}
                title="Bấm để xem đoạn video tại frame này"
              >
                <Play size={12} />
                <span className="truncate">
                  {row.videoId} · frame <span className="tabular">{row.frameIdx}</span>
                  {row.answer && ` · ${row.answer}`}
                  {row.frameIds && row.frameIds.length > 1 && ` · ${row.frameIds.length} bước`}
                </span>
              </button>
              <IconButton icon={<ArrowUp size={14} />} label="Lên một bậc" size="sm" variant="control"
                          disabled={index === 0} onClick={() => move(index, -1)} />
              <IconButton icon={<ArrowDown size={14} />} label="Xuống một bậc" size="sm" variant="control"
                          disabled={index === rows.length - 1} onClick={() => move(index, 1)} />
              <IconButton icon={<X size={14} />} label="Bỏ khỏi bài nộp" size="sm" variant="control"
                          onClick={() => remove(index)} />
            </li>
          ))}
        </ol>

        <div className="submission-preview">
          {selectedRow == null ? (
            <EmptyState
              size="sm"
              icon={<Play size={16} />}
              title="Chọn một dòng để xem"
              description="Video sẽ tua thẳng tới đúng frame sẽ nộp."
            />
          ) : selectedHit?.video_path && !videoError ? (
            <>
              <video
                ref={videoRef}
                src={mediaUrl(apiConfig, selectedHit.video_path)}
                controls
                onError={() => setVideoError(true)}
              />
              <p className="submission-preview-meta tabular">
                {selectedRow.videoId} · frame {selectedRow.frameIdx}
                {seekTo != null && ` · ${seekTo.toFixed(2)}s`}
              </p>
            </>
          ) : selectedHit?.best_keyframe_path ? (
            <>
              <img src={mediaUrl(apiConfig, selectedHit.best_keyframe_path)} alt={`frame ${selectedRow.frameIdx}`} />
              <p className="submission-preview-meta">Không phát được video — hiện keyframe thay thế.</p>
            </>
          ) : (
            <EmptyState
              size="sm"
              icon={<ImageOff size={16} />}
              title="Không có media"
              description="Dòng này không tra được scene trong kết quả hiện tại nên chưa có đường dẫn video."
            />
          )}
        </div>
      </div>

      {error && <InlineError message={error} />}
      {result && (
        <div className="submission-result">
          <p className={result.has_errors ? "warning-text" : "muted"}>
            {result.item_count} dòng · {result.has_errors ? "CÓ LỖI, không nên nộp" : "hợp lệ"}
          </p>
          {result.issues.length > 0 && (
            <ul className="issue-list">
              {result.issues.map((issue, index) => (
                <li key={index} className={issue.severity === "error" ? "issue-error" : "issue-warning"}>
                  [{issue.severity}] {issue.code}: {issue.message}
                  {issue.row_index != null && ` (dòng ${issue.row_index})`}
                </li>
              ))}
            </ul>
          )}
          <pre className="csv-preview scroll-y">{result.csv}</pre>
          <Button
            variant="secondary" size="sm" icon={<FileDown size={13} />}
            disabled={result.has_errors}
            onClick={() => downloadCsv(result.csv, submissionFilename(task))}
          >
            Tải CSV xuống
          </Button>
        </div>
      )}
    </div>
  );
}
