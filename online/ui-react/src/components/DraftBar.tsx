import { useCallback, useEffect, useState } from "react";
import { FolderOpen, RefreshCw, Save, Trash2, Users } from "lucide-react";
import type { ApiClientConfig } from "../api";
import { ApiError, deleteDraft, listDrafts, saveDraft } from "../api";
import type { DraftRow, SubmissionDraft, TaskType } from "../types";
import { Button, IconButton, InlineError } from "../ui";

const AUTHOR_KEY = "aic_draft_author";

function loadAuthor(): string {
  try {
    return localStorage.getItem(AUTHOR_KEY) ?? "";
  } catch {
    return "";
  }
}

export interface DraftBarProps {
  apiConfig: ApiClientConfig;
  task: TaskType;
  /** Truy vấn đang chạy — lưu kèm để người nạp lại biết phải chạy lại câu nào. */
  query: string;
  /** Các dòng đang sắp trên bảng nộp, theo đúng thứ tự hiện tại. */
  rows: DraftRow[];
  /** Nạp một bản nháp về bảng nộp. Component này không biết cách khớp dòng —
   *  đó là việc của bảng nộp, nơi có `sourceRows` của truy vấn hiện tại. */
  onLoad: (draft: SubmissionDraft) => void;
}

/** Lưu / nạp bản nháp sắp xếp, dùng chung cả đội.
 *
 * Nằm ở server chứ không localStorage: cả đội trỏ vào cùng một backend nên
 * lưu ở đó là tự động thấy được của nhau, và bản soát tay sống qua restart.
 * Tên người lưu thì giữ ở localStorage — nó là thuộc tính của cái máy đang
 * ngồi, không phải của dữ liệu.
 */
export function DraftBar({ apiConfig, task, query, rows, onLoad }: DraftBarProps) {
  const [drafts, setDrafts] = useState<SubmissionDraft[]>([]);
  const [author, setAuthor] = useState(loadAuthor);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDrafts(await listDrafts(apiConfig));
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  }, [apiConfig]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function handleSave() {
    if (!name.trim() || rows.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      // Trùng tên + trùng người = ghi đè. Bấm Lưu năm lần trong một buổi soát
      // không được đẻ ra năm bản cùng tên để rồi không ai biết bản nào mới.
      const existing = drafts.find(
        (draft) => draft.name === name.trim() && draft.author === author.trim()
      );
      const saved = await saveDraft(apiConfig, {
        name: name.trim(),
        author: author.trim(),
        task,
        query,
        rows,
        draft_id: existing?.draft_id ?? null,
      });
      setSelected(saved.draft_id);
      setNote(`Đã lưu "${saved.name}" · ${saved.rows.length} dòng`);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    const draft = drafts.find((item) => item.draft_id === selected);
    if (!draft) return;
    setBusy(true);
    setError(null);
    try {
      await deleteDraft(apiConfig, draft.draft_id);
      setSelected("");
      setNote(`Đã xoá "${draft.name}"`);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const chosen = drafts.find((item) => item.draft_id === selected) ?? null;

  return (
    <div className="draft-bar">
      <div className="draft-bar-row">
        <Users size={13} />
        <input
          type="text"
          className="draft-input"
          value={author}
          placeholder="tên bạn"
          aria-label="Tên người lưu bản nháp"
          onChange={(event) => {
            setAuthor(event.target.value);
            try {
              localStorage.setItem(AUTHOR_KEY, event.target.value);
            } catch {
              // Trình duyệt chặn storage thì vẫn dùng được, chỉ là lần sau
              // phải gõ lại tên — không đáng để chặn cả tính năng.
            }
          }}
        />
        <input
          type="text"
          className="draft-input draft-input-wide"
          value={name}
          placeholder="tên bản nháp (vd: câu 7 — thứ tự của Nhân)"
          aria-label="Tên bản nháp"
          onChange={(event) => setName(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void handleSave();
          }}
        />
        <Button
          variant="secondary" size="sm" icon={<Save size={13} />}
          disabled={busy || !name.trim() || rows.length === 0}
          onClick={() => void handleSave()}
        >
          Lưu nháp
        </Button>
      </div>

      <div className="draft-bar-row">
        <FolderOpen size={13} />
        <select
          className="draft-select"
          value={selected}
          aria-label="Bản nháp của cả đội"
          onChange={(event) => setSelected(event.target.value)}
        >
          <option value="">
            {drafts.length === 0 ? "chưa có bản nháp nào" : `${drafts.length} bản nháp của cả đội`}
          </option>
          {drafts.map((draft) => (
            <option key={draft.draft_id} value={draft.draft_id}>
              {draft.name} · {draft.author || "?"} · {draft.task} · {draft.rows.length} dòng
            </option>
          ))}
        </select>
        <Button
          variant="secondary" size="sm"
          disabled={busy || !chosen}
          onClick={() => chosen && onLoad(chosen)}
        >
          Nạp
        </Button>
        <IconButton
          icon={<RefreshCw size={13} />}
          label="Tải lại danh sách bản nháp"
          size="sm" variant="control"
          disabled={busy}
          onClick={() => void refresh()}
        />
        <IconButton
          icon={<Trash2 size={13} />}
          label={chosen ? `Xoá bản nháp "${chosen.name}"` : "Xoá bản nháp đang chọn"}
          size="sm" variant="control"
          disabled={busy || !chosen}
          onClick={() => void handleDelete()}
        />
      </div>

      {chosen && chosen.task !== task && (
        <p className="warning-text">
          Bản nháp này thuộc task {chosen.task}, bạn đang ở {task} — nạp vào sẽ không khớp dòng nào.
        </p>
      )}
      {chosen?.query && (
        <p className="muted small truncate" title={chosen.query}>
          Truy vấn của bản nháp: “{chosen.query}”
        </p>
      )}
      {note && <p className="muted small">{note}</p>}
      {error && <InlineError message={`Bản nháp: ${error}`} />}
    </div>
  );
}
