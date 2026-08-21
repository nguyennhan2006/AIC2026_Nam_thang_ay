import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle, ArrowDown, ArrowUp, CheckCircle2, FileDown, HelpCircle, ImageOff,
  ListOrdered, MessageSquareText, Pencil, Play, RotateCcw, SlidersHorizontal, Trash2, X,
} from "lucide-react";
import type { ApiClientConfig } from "../api";
import { ApiError, buildSubmission, mediaUrl } from "../api";
import { downloadCsv, submissionFilename } from "../exportCsv";
import type {
  AvsResultItem,
  KisResultItem,
  PlaybackWindow,
  QaResultItem,
  SearchHit,
  SubmissionBuildResponse,
  TaskType,
  TrakeResultItem,
  TrakeStep,
} from "../types";
import { zoneForRank } from "../types";
import type { TunerRow } from "./FrameTuner";
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
  /** Báo lên trên chuỗi nào đang chọn, để panel xem bên phải đi theo.
   *  Thiếu callback này thì bảng và panel xem chỉ vào hai chuỗi khác nhau —
   *  bấm dòng 2 mà bên phải vẫn hiện chuỗi 0. */
  onSelectSequence?: (index: number) => void;
  /** "Lưu & chỉnh frame": đẩy đúng các dòng ĐANG chọn (kể cả thứ tự đã sắp
   *  tay) sang tab chỉnh frame. Thiếu callback này thì tab chỉnh frame vẫn
   *  nạp từ kết quả tìm kiếm thô, nên mọi thao tác sắp xếp và loại bỏ ở đây
   *  bị bỏ qua — người dùng chỉnh một danh sách khác với danh sách sắp nộp. */
  onEditRows?: (rows: TunerRow[]) => void;
}

/** Biểu tượng của một dòng: nói ngay dòng này ĐÁNG TIN tới đâu.
 *
 * Có nó vì người dùng dễ nhầm ba thứ trông giống nhau trong cùng một bảng:
 * chuỗi đầy đủ, chuỗi có frame do hệ nội suy, và chuỗi thiếu hẳn bước. Ba loại
 * này cần ba hành động khác nhau trước khi nộp.
 */
function RowIcon({ task, row }: { task: TaskType; row: Row }) {
  if (task === "QA") return <MessageSquareText size={13} aria-label="Câu trả lời" />;
  if (task !== "TRAKE") return <Play size={13} aria-label="Một frame" />;
  const holes = row.missingSteps?.length ?? 0;
  const guessed = row.steps?.filter((step) => step.refinement === "interpolated").length ?? 0;
  if (holes > 0) {
    return (
      <span className="row-icon is-danger" title={`Thiếu ${holes} bước — dòng này chưa nộp được`}>
        <HelpCircle size={13} />
      </span>
    );
  }
  if (guessed > 0) {
    return (
      <span className="row-icon is-warning" title={`${guessed} bước là frame nội suy, không phải bằng chứng — nên xem lại`}>
        <AlertTriangle size={13} />
      </span>
    );
  }
  return (
    <span className="row-icon is-ok" title="Đủ bước, mọi frame đều có bằng chứng">
      <CheckCircle2 size={13} />
    </span>
  );
}

/** Một dòng sẽ nộp, đã tách khỏi kiểu của từng task để bảng dùng chung. */
interface Row {
  key: string;
  videoId: string;
  frameIdx: number;
  sceneId: string | null;
  answer?: string;
  frameIds?: number[];
  /** TRAKE: một dòng LÀ một chuỗi. Giữ đủ step để duyệt lại từng khoảnh khắc
   *  trước khi nộp — bản cũ chỉ giữ `frame_ids[0]` nên người dùng không xem
   *  được chuỗi, chỉ thấy đúng cảnh đầu. */
  steps?: TrakeStep[];
  /** Bước hệ không tìm được gì. Rỗng = chuỗi đủ. */
  missingSteps?: number[];
  playback?: PlaybackWindow | null;
  /** Vị trí trong mảng gốc của task — dùng để đồng bộ với panel xem. */
  sourceIndex: number;
}

/** Dòng đang chọn ở bảng nộp -> dòng cho tab chỉnh frame.
 *
 * Giữ NGUYÊN thứ tự và tập dòng của bảng nộp: đó là điểm khác biệt so với việc
 * tab chỉnh frame tự nạp từ kết quả tìm kiếm. Bước thiếu được dựng sẵn ở điểm
 * giữa hai mốc lân cận để có chỗ bám mà kéo.
 */
function toTunerRows(task: TaskType, rows: Row[]): TunerRow[] {
  if (task !== "TRAKE") {
    return rows.map((row, index) => ({
      id: `sub-${index}`, videoId: row.videoId,
      originalFrame: row.frameIdx, frame: row.frameIdx, answer: row.answer,
    }));
  }
  const out: TunerRow[] = [];
  rows.forEach((row, chain) => {
    const known = new Map((row.steps ?? []).map((step) => [step.step, step]));
    const total = Math.max(0, ...known.keys(), ...(row.missingSteps ?? []));
    for (let step = 1; step <= total; step += 1) {
      const entry = known.get(step);
      if (entry) {
        out.push({
          id: `sub-${chain}-${step}`, videoId: row.videoId,
          originalFrame: entry.frame_idx, frame: entry.frame_idx,
          chain: chain + 1, step,
          placeholder: entry.refinement === "interpolated",
        });
        continue;
      }
      let before: number | null = null;
      let after: number | null = null;
      for (let probe = step - 1; probe >= 1; probe -= 1) {
        const found = known.get(probe);
        if (found) { before = found.frame_idx; break; }
      }
      for (let probe = step + 1; probe <= total; probe += 1) {
        const found = known.get(probe);
        if (found) { after = found.frame_idx; break; }
      }
      const guess =
        before != null && after != null ? Math.round((before + after) / 2)
        : before != null ? before + 1
        : after != null ? Math.max(0, after - 1)
        : 0;
      out.push({
        id: `sub-${chain}-${step}`, videoId: row.videoId,
        originalFrame: guess, frame: guess, chain: chain + 1, step,
        placeholder: true,
      });
    }
  });
  return out;
}


function toRows(task: TaskType, kis: KisResultItem[], qa: QaResultItem[], trake: TrakeResultItem[]): Row[] {
  if (task === "TEXTUAL_KIS") {
    return kis.map((item, index) => ({
      key: `${item.video_id}-${item.frame_idx}`, sourceIndex: index,
      videoId: item.video_id, frameIdx: item.frame_idx, sceneId: item.scene_id,
    }));
  }
  if (task === "QA") {
    return qa.map((item, index) => ({
      key: `${item.video_id}-${item.frame_idx}-${item.canonical_answer}`, sourceIndex: index,
      videoId: item.video_id, frameIdx: item.frame_idx, sceneId: item.scene_id,
      answer: item.answer,
    }));
  }
  return trake.map((item, index) => ({
    key: `${item.video_id}-${item.rank}`, sourceIndex: index,
    videoId: item.video_id, frameIdx: item.frame_ids[0] ?? 0,
    sceneId: item.steps[0]?.scene_id ?? null, frameIds: item.frame_ids,
    steps: item.steps, missingSteps: item.missing_steps ?? [],
    playback: item.playback ?? null,
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

/** Dải bước của một chuỗi TRAKE — bấm để nhảy tới đúng khoảnh khắc đó.
 *
 * Không có nó thì không xem lại được chuỗi trước khi nộp: bảng chỉ hiện frame
 * đầu, mà một dòng TRAKE là `video_id, f1, ..., fn` và điểm phụ thuộc CẢ n
 * khoảnh khắc lẫn thứ tự của chúng.
 */
function StepStrip({
  steps, active, onPick,
}: {
  steps: TrakeStep[];
  active: number;
  onPick: (index: number) => void;
}) {
  return (
    <div className="step-strip" role="group" aria-label="Các bước trong chuỗi">
      {steps.map((step, index) => (
        <button
          key={`${step.step}-${step.frame_idx}`}
          type="button"
          className={index === active ? "step-chip is-active" : "step-chip"}
          onClick={() => onPick(index)}
          title={`Bước ${step.step} · frame ${step.frame_idx}${
            step.timestamp_sec != null ? ` · ${step.timestamp_sec.toFixed(2)}s` : ""
          }`}
        >
          <span className="step-chip-index tabular">{step.step}</span>
          <span className="step-chip-time tabular">
            {step.timestamp_sec != null ? `${step.timestamp_sec.toFixed(1)}s` : `#${step.frame_idx}`}
          </span>
        </button>
      ))}
    </div>
  );
}

export function SubmissionBoard({ apiConfig, task, kis, qa, trake, avs, results = [], onSelectSequence, onEditRows }: SubmissionBoardProps) {
  const [result, setResult] = useState<SubmissionBuildResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<Row[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [videoError, setVideoError] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Tick là tập ĐỘC LẬP với `selected`: `selected` chỉ quyết định panel xem
  // bên phải (một dòng), tick quyết định dòng nào chịu thao tác hàng loạt.
  // Gộp hai thứ này lại thì không sửa được 12 đáp án trong khi vẫn đang xem
  // video của dòng thứ 3.
  const [ticked, setTicked] = useState<Set<string>>(() => new Set());
  // Neo cho shift-click: tick dòng 3 rồi shift-tick dòng 9 = tick cả dải.
  const anchorRef = useRef<number | null>(null);
  const tickAllRef = useRef<HTMLInputElement>(null);
  const [bulkAnswer, setBulkAnswer] = useState("");
  const [bulkRank, setBulkRank] = useState("");
  // Hạng gõ tay: giữ ở dạng CHUỖI nháp, chỉ chốt khi Enter/blur. Sắp lại ngay
  // từng phím gõ thì dòng chạy khỏi con trỏ giữa chừng — gõ "12" thì chữ "1"
  // đã kịp đẩy dòng lên đầu bảng trước khi chữ "2" tới.
  const [rankDraft, setRankDraft] = useState<{ key: string; text: string } | null>(null);

  const sourceRows = useMemo(() => toRows(task, kis, qa, trake), [task, kis, qa, trake]);

  // Kết quả mới về thì thứ tự thủ công cũ không còn ý nghĩa — nạp lại từ đầu.
  useEffect(() => {
    setRows(sourceRows);
    setSelected(null);
    setResult(null);
    setTicked(new Set());
    setRankDraft(null);
    anchorRef.current = null;
  }, [sourceRows]);

  // Đáp án nguyên bản của model, để biết dòng nào đã bị sửa tay. So theo
  // `key` chứ không theo vị trí — bảng này sắp lại thứ tự liên tục.
  const sourceAnswers = useMemo(
    () => new Map(sourceRows.map((row) => [row.key, row.answer ?? ""])),
    [sourceRows]
  );
  const editedKeys = useMemo(
    () =>
      new Set(
        rows.filter((row) => (row.answer ?? "") !== (sourceAnswers.get(row.key) ?? "")).map((row) => row.key)
      ),
    [rows, sourceAnswers]
  );

  const tickedCount = useMemo(
    () => rows.reduce((total, row) => total + (ticked.has(row.key) ? 1 : 0), 0),
    [rows, ticked]
  );
  // Ô "tick tất cả" phải phân biệt được ba trạng thái. Chỉ checked/unchecked
  // thì tick 3/12 dòng trông y hệt không tick dòng nào, và người dùng bấm nó
  // để "chọn hết" lại hoá ra xoá sạch lựa chọn vừa làm.
  useEffect(() => {
    if (tickAllRef.current) tickAllRef.current.indeterminate = tickedCount > 0 && tickedCount < rows.length;
  }, [tickedCount, rows.length]);

  const hitByScene = useMemo(() => {
    const map = new Map<string, SearchHit>();
    for (const hit of results) if (!map.has(hit.scene_id)) map.set(hit.scene_id, hit);
    return map;
  }, [results]);

  const selectedRow = rows.find((row) => row.key === selected) ?? null;
  const selectedHit = selectedRow?.sceneId ? hitByScene.get(selectedRow.sceneId) ?? null : null;

  // TRAKE: một dòng là một CHUỖI. `stepIndex` là khoảnh khắc đang xem trong
  // chuỗi đó; các task khác luôn ở 0.
  const [stepIndex, setStepIndex] = useState(0);
  useEffect(() => setStepIndex(0), [selected]);
  const chainSteps = selectedRow?.steps ?? null;
  const activeStep: TrakeStep | null = chainSteps?.[stepIndex] ?? null;

  // Nguồn phát: ưu tiên cửa sổ của cả chuỗi (backend trải từ frame đầu tới
  // frame cuối), rồi tới cửa sổ của scene, cuối cùng là video thô.
  const window: PlaybackWindow | null = selectedRow?.playback ?? selectedHit?.playback ?? null;
  const videoSrc = window
    ? `${mediaUrl(apiConfig, window.media_path)}#t=${window.start_sec.toFixed(3)},${window.end_sec.toFixed(3)}`
    : selectedHit?.video_path
      ? mediaUrl(apiConfig, selectedHit.video_path)
      : null;

  const seekTo =
    activeStep?.timestamp_sec ??
    (selectedRow && selectedHit ? seekSecondsFor(selectedHit, selectedRow.frameIdx) : null);

  useEffect(() => setVideoError(false), [videoSrc]);

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
  }, [seekTo, videoSrc]);

  /** Chèn một dòng vào đúng hạng, ĐẨY các dòng khác xuống — không hoán đổi.
   *
   *  Hoán đổi là thao tác sai cho việc xếp hạng: đưa dòng 17 lên hạng 2 bằng
   *  hoán đổi sẽ ném dòng 2 xuống tận 17, phá thứ tự của 15 dòng ở giữa mà
   *  người dùng vừa sắp. Chèn giữ nguyên thứ tự tương đối của phần còn lại.
   */
  function moveRowTo(key: string, targetIndex: number) {
    setRows((current) => {
      const from = current.findIndex((row) => row.key === key);
      if (from < 0) return current;
      const to = Math.max(0, Math.min(targetIndex, current.length - 1));
      if (to === from) return current;
      const next = [...current];
      const [item] = next.splice(from, 1);
      next.splice(to, 0, item);
      return next;
    });
    setResult(null);
  }

  function move(index: number, delta: number) {
    const row = rows[index];
    if (row) moveRowTo(row.key, index + delta);
  }

  function untick(keys: Iterable<string>) {
    setTicked((current) => {
      const next = new Set(current);
      for (const key of keys) next.delete(key);
      return next.size === current.size ? current : next;
    });
  }

  function remove(index: number) {
    const key = rows[index]?.key;
    setRows((current) => current.filter((_, position) => position !== index));
    if (key) untick([key]);
    setResult(null);
  }

  /** Tick một dòng; `extend` (shift-click) tick cả dải từ neo tới đây. */
  function toggleTick(index: number, extend: boolean) {
    const row = rows[index];
    if (!row) return;
    // Đọc neo NGAY BÂY GIỜ, không đọc trong updater: updater của React chạy
    // ở pha render, tức là sau dòng `anchorRef.current = index` bên dưới —
    // đọc muộn thì neo luôn bằng chính dòng vừa bấm và shift-click không bao
    // giờ mở rộng được dải.
    const anchor = anchorRef.current;
    anchorRef.current = index;
    setTicked((current) => {
      const next = new Set(current);
      if (extend && anchor != null && anchor !== index && anchor < rows.length) {
        const [low, high] = anchor < index ? [anchor, index] : [index, anchor];
        const turnOn = !next.has(row.key);
        for (let position = low; position <= high; position += 1) {
          const target = rows[position];
          if (!target) continue;
          if (turnOn) next.add(target.key);
          else next.delete(target.key);
        }
      } else if (next.has(row.key)) {
        next.delete(row.key);
      } else {
        next.add(row.key);
      }
      return next;
    });
  }

  /** Một câu trả lời cho TẤT CẢ dòng đã tick.
   *
   *  Model hay trả cùng một sự thật bằng nhiều cách viết ("3 người", "ba
   *  người", "3 nguoi") trên các frame khác nhau. BTC chấm QA theo chuỗi
   *  answer, nên sửa từng dòng một vừa chậm vừa dễ sót một biến thể.
   */
  function applyBulkAnswer() {
    const value = bulkAnswer.trim();
    if (task !== "QA" || value === "" || ticked.size === 0) return;
    setRows((current) => current.map((row) => (ticked.has(row.key) ? { ...row, answer: value } : row)));
    setResult(null);
  }

  /** Đưa CẢ khối đã tick tới hạng N, giữ thứ tự tương đối bên trong khối. */
  function applyBulkRank() {
    const wanted = Number.parseInt(bulkRank, 10);
    if (!Number.isFinite(wanted) || ticked.size === 0) return;
    setRows((current) => {
      const block = current.filter((row) => ticked.has(row.key));
      if (block.length === 0) return current;
      const rest = current.filter((row) => !ticked.has(row.key));
      const at = Math.max(0, Math.min(wanted - 1, rest.length));
      return [...rest.slice(0, at), ...block, ...rest.slice(at)];
    });
    setResult(null);
  }

  function removeTicked() {
    if (ticked.size === 0) return;
    setRows((current) => current.filter((row) => !ticked.has(row.key)));
    setTicked(new Set());
    anchorRef.current = null;
    setResult(null);
  }

  /** Chốt hạng gõ tay. Bị gọi hai lần khi bấm Enter rồi rời ô, nên nháp đã
   *  xoá thì lần sau không được chạy lại — nếu không dòng bị đẩy đi hai lần. */
  function commitRank(key: string) {
    if (!rankDraft || rankDraft.key !== key) return;
    const wanted = Number.parseInt(rankDraft.text, 10);
    setRankDraft(null);
    if (!Number.isFinite(wanted)) return;
    moveRowTo(key, wanted - 1);
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

      // QA: `answer` sửa tay ở bảng này phải THẮNG câu của model — đó mới là
      // cột đi vào CSV. `canonical_answer` giữ nguyên vì nó chỉ là khoá truy
      // vết về candidate gốc, không có mặt trong bài nộp.
      const qaKey = (item: QaResultItem) => `${item.video_id}-${item.frame_idx}-${item.canonical_answer}`;
      const answerByKey = new Map(rows.map((row) => [row.key, row.answer ?? ""]));

      const body =
        task === "TEXTUAL_KIS"
          ? { task, kis: pick(kis, (i) => `${i.video_id}-${i.frame_idx}`) }
          : task === "QA"
            ? {
                task,
                qa: pick(qa, qaKey).map((item) => {
                  const edited = answerByKey.get(qaKey(item));
                  return edited != null && edited !== item.answer ? { ...item, answer: edited } : item;
                }),
              }
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
  const allTicked = rows.length > 0 && tickedCount === rows.length;

  return (
    <div className="submission-board">
      <div className="submission-toolbar">
        <span className="eyebrow">{rows.length} dòng</span>
        {reordered && <Badge tone="warning">đã sắp lại tay</Badge>}
        {editedKeys.size > 0 && <Badge tone="accent">{editedKeys.size} đáp án sửa tay</Badge>}
        {(reordered || editedKeys.size > 0) && (
          <IconButton
            icon={<RotateCcw size={14} />}
            label="Khôi phục thứ tự và đáp án của hệ thống"
            size="sm"
            variant="control"
            onClick={() => {
              setRows(sourceRows);
              setTicked(new Set());
              setRankDraft(null);
              anchorRef.current = null;
              setResult(null);
            }}
          />
        )}
        <div className="submission-zones">
          {["rank_1", "ranks_2_5", "ranks_6_20", "ranks_21_50", "ranks_51_100"].map((zone) => (
            <span key={zone} className="zone-chip">{zone.replace("ranks_", "").replace("rank_", "")}: {zoneCounts.get(zone) ?? 0}</span>
          ))}
        </div>
        {onEditRows && (
          <Button
            variant="secondary" size="sm" icon={<SlidersHorizontal size={13} />}
            disabled={rows.length === 0}
            onClick={() => onEditRows(toTunerRows(task, rows))}
          >
            Lưu &amp; chỉnh frame
          </Button>
        )}
        <Button variant="primary" size="sm" loading={loading} onClick={runBuild}>
          Build CSV
        </Button>
      </div>

      {/* Thanh hàng loạt: luôn hiện (kể cả khi chưa tick dòng nào) để ô "tick
          tất cả" có chỗ đứng cố định. Các nút bên trong tự khoá khi chưa có
          dòng nào được tick — thanh nhảy ra nhảy vào mỗi lần tick thì bảng
          bên dưới cũng nhảy theo, mất chỗ đang nhắm chuột. */}
      <div className="submission-bulk">
        <label className="submission-tick-all" title="Tick / bỏ tick tất cả các dòng">
          <input
            ref={tickAllRef}
            type="checkbox"
            className="submission-tick"
            checked={allTicked}
            aria-label="Tick tất cả các dòng"
            onChange={(event) => {
              setTicked(event.target.checked ? new Set(rows.map((row) => row.key)) : new Set());
              anchorRef.current = null;
            }}
          />
          <span className="tabular">{tickedCount}/{rows.length}</span>
        </label>

        {task === "QA" && (
          <div className="submission-bulk-group">
            <input
              type="text"
              className="submission-bulk-answer"
              value={bulkAnswer}
              placeholder="Câu trả lời thống nhất…"
              aria-label="Câu trả lời áp cho các dòng đã tick"
              disabled={tickedCount === 0}
              onChange={(event) => setBulkAnswer(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") applyBulkAnswer(); }}
            />
            <Button
              variant="secondary" size="sm" icon={<Pencil size={13} />}
              disabled={tickedCount === 0 || bulkAnswer.trim() === ""}
              onClick={applyBulkAnswer}
            >
              Áp cho {tickedCount} dòng
            </Button>
          </div>
        )}

        <div className="submission-bulk-group">
          <span className="eyebrow">Đưa tới hạng</span>
          <input
            type="number"
            className="submission-bulk-rank tabular"
            min={1}
            max={Math.max(rows.length, 1)}
            value={bulkRank}
            placeholder="1"
            aria-label="Hạng đích cho các dòng đã tick"
            disabled={tickedCount === 0}
            onChange={(event) => setBulkRank(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Enter") applyBulkRank(); }}
          />
          <Button
            variant="secondary" size="sm" icon={<ListOrdered size={13} />}
            disabled={tickedCount === 0 || bulkRank.trim() === ""}
            onClick={applyBulkRank}
          >
            Chèn
          </Button>
        </div>

        <IconButton
          icon={<Trash2 size={14} />}
          label={`Bỏ ${tickedCount} dòng đã tick khỏi bài nộp`}
          size="sm"
          variant="control"
          disabled={tickedCount === 0}
          onClick={removeTicked}
        />
      </div>
      <p className="muted small submission-hint">
        Gõ số vào ô hạng của một dòng rồi Enter để chèn nó vào đúng vị trí đó — các dòng khác dịch
        xuống, không bị hoán đổi. Giữ Shift khi tick để chọn cả dải.
      </p>

      <div className="submission-split">
        <ol className="submission-list scroll-y">
          {rows.map((row, index) => {
            const isTicked = ticked.has(row.key);
            return (
            <li
              key={row.key}
              className={[
                "submission-row",
                row.key === selected ? "is-selected" : "",
                isTicked ? "is-ticked" : "",
              ].filter(Boolean).join(" ")}
            >
              <input
                type="checkbox"
                className="submission-tick"
                checked={isTicked}
                aria-label={`Chọn dòng ${index + 1} — ${row.videoId}`}
                title="Tick để sửa/xếp hàng loạt (giữ Shift để chọn cả dải)"
                onChange={(event) => toggleTick(index, (event.nativeEvent as MouseEvent).shiftKey === true)}
              />
              <input
                type="number"
                className="submission-rank-input tabular"
                min={1}
                max={rows.length}
                value={rankDraft?.key === row.key ? rankDraft.text : index + 1}
                aria-label={`Hạng của ${row.videoId}, hiện là ${index + 1}`}
                title="Gõ hạng mới rồi Enter — các dòng khác dịch xuống"
                onFocus={(event) => event.currentTarget.select()}
                onChange={(event) => setRankDraft({ key: row.key, text: event.target.value })}
                onBlur={() => commitRank(row.key)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") commitRank(row.key);
                  if (event.key === "Escape") { setRankDraft(null); event.currentTarget.blur(); }
                }}
              />
              <button
                type="button"
                className="submission-main"
                onClick={() => {
                  const next = row.key === selected ? null : row.key;
                  setSelected(next);
                  if (next) onSelectSequence?.(row.sourceIndex);
                }}
                title="Bấm để xem đoạn video tại frame này"
              >
                <RowIcon task={task} row={row} />
                <span className="truncate">
                  {row.videoId} · frame <span className="tabular">{row.frameIdx}</span>
                  {row.answer && ` · ${row.answer}`}
                  {row.frameIds && row.frameIds.length > 1 && ` · ${row.frameIds.length} bước`}
                  {row.missingSteps && row.missingSteps.length > 0 &&
                    ` · thiếu bước ${row.missingSteps.join(", ")}`}
                </span>
                {editedKeys.has(row.key) && (
                  <span className="answer-edited" title="Đáp án đã sửa tay, khác câu của model">sửa</span>
                )}
              </button>
              <IconButton icon={<ArrowUp size={14} />} label="Lên một bậc" size="sm" variant="control"
                          disabled={index === 0} onClick={() => move(index, -1)} />
              <IconButton icon={<ArrowDown size={14} />} label="Xuống một bậc" size="sm" variant="control"
                          disabled={index === rows.length - 1} onClick={() => move(index, 1)} />
              <IconButton icon={<X size={14} />} label="Bỏ khỏi bài nộp" size="sm" variant="control"
                          onClick={() => remove(index)} />
            </li>
            );
          })}
        </ol>

        <div className="submission-preview">
          {selectedRow == null ? (
            <EmptyState
              size="sm"
              icon={<Play size={16} />}
              title="Chọn một dòng để xem"
              description="Video sẽ tua thẳng tới đúng frame sẽ nộp."
            />
          ) : videoSrc && !videoError ? (
            <>
              <video ref={videoRef} src={videoSrc} controls onError={() => setVideoError(true)} />
              {chainSteps && chainSteps.length > 1 && (
                <StepStrip steps={chainSteps} active={stepIndex} onPick={setStepIndex} />
              )}
              <p className="submission-preview-meta tabular">
                {selectedRow.videoId}
                {activeStep
                  ? ` · bước ${activeStep.step}/${chainSteps?.length ?? 1} · frame ${activeStep.frame_idx}`
                  : ` · frame ${selectedRow.frameIdx}`}
                {seekTo != null && ` · ${seekTo.toFixed(2)}s`}
                {window && ` · đoạn ${window.start_sec.toFixed(1)}–${window.end_sec.toFixed(1)}s`}
              </p>
            </>
          ) : activeStep?.image_path ?? selectedHit?.best_keyframe_path ? (
            <>
              <img
                src={mediaUrl(apiConfig, (activeStep?.image_path ?? selectedHit?.best_keyframe_path)!)}
                alt={`frame ${activeStep?.frame_idx ?? selectedRow.frameIdx}`}
              />
              {chainSteps && chainSteps.length > 1 && (
                <StepStrip steps={chainSteps} active={stepIndex} onPick={setStepIndex} />
              )}
              <p className="submission-preview-meta">
                Chưa có video nguồn — duyệt chuỗi bằng khung hình.
              </p>
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
