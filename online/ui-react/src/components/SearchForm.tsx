import type { FormEvent } from "react";
import type { Task } from "../types";

export interface SearchFormProps {
  apiBase: string;
  onApiBaseChange: (value: string) => void;
  apiToken: string;
  onApiTokenChange: (value: string) => void;
  task: Task;
  onTaskChange: (value: Task) => void;
  query: string;
  onQueryChange: (value: string) => void;
  topK: number;
  onTopKChange: (value: number) => void;
  debug: boolean;
  onDebugChange: (value: boolean) => void;
  onSubmit: () => void;
  onHealthCheck: () => void;
}

export function SearchForm({
  apiBase,
  onApiBaseChange,
  apiToken,
  onApiTokenChange,
  task,
  onTaskChange,
  query,
  onQueryChange,
  topK,
  onTopKChange,
  debug,
  onDebugChange,
  onSubmit,
  onHealthCheck,
}: SearchFormProps) {
  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit();
  }

  return (
    <form id="search-form" onSubmit={handleSubmit}>
      <label>
        Backend Vast.ai
        <input
          id="api-base"
          type="url"
          placeholder="https://your-vast-host:port"
          autoComplete="url"
          value={apiBase}
          onChange={(e) => onApiBaseChange(e.target.value)}
        />
      </label>
      <label>
        API token
        <input
          type="password"
          placeholder="để trống nếu không bật"
          value={apiToken}
          onChange={(e) => onApiTokenChange(e.target.value)}
        />
      </label>
      <label>
        Loại nhiệm vụ
        <select id="task" value={task} onChange={(e) => onTaskChange(e.target.value as Task)}>
          <option value="kis">KIS</option>
          <option value="avs">AVS</option>
          <option value="sequence">Sequence</option>
          <option value="vqa">VQA</option>
        </select>
      </label>
      <label className="query-label wide">
        Truy vấn
        <textarea
          id="query"
          rows={4}
          required
          placeholder='Ví dụ: Người cào muối, sau đó đoàn người vẫy tay...'
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
        />
      </label>
      <label>
        Top K
        <input type="number" min={1} max={100} value={topK} onChange={(e) => onTopKChange(Number(e.target.value))} />
      </label>
      <label className="check">
        <input type="checkbox" checked={debug} onChange={(e) => onDebugChange(e.target.checked)} /> Hiện query plan
      </label>
      <button type="submit">Tìm kiếm</button>
      <button type="button" onClick={onHealthCheck}>
        Kiểm tra server
      </button>
    </form>
  );
}
