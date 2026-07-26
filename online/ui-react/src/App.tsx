import { useMemo, useRef, useState } from "react";
import { ApiError, health, search, vqa as vqaCall } from "./api";
import { DebugPanel } from "./components/DebugPanel";
import { ResultsView } from "./components/ResultsView";
import { SearchForm } from "./components/SearchForm";
import { SelectionTray } from "./components/SelectionTray";
import { loadApiBase, loadApiToken, loadSelection, loadTrayAnswer, saveApiBase, saveApiToken, saveSelection, saveTrayAnswer } from "./storage";
import type { SceneDocument, SearchHit, SearchResponse, Task, TrayItem, VQAResponse } from "./types";

function App() {
  const [apiBase, setApiBase] = useState(loadApiBase);
  const [apiToken, setApiToken] = useState(loadApiToken);
  const [task, setTask] = useState<Task>("kis");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [debug, setDebug] = useState(false);
  const [status, setStatus] = useState("");
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [vqaResult, setVqaResult] = useState<VQAResponse | null>(null);
  const [selection, setSelection] = useState<Map<string, TrayItem>>(loadSelection);
  const [trayAnswer, setTrayAnswer] = useState(loadTrayAnswer);
  const sceneDetailCache = useRef(new Map<string, Promise<SceneDocument>>()).current;

  const apiConfig = useMemo(() => ({ base: apiBase, token: apiToken }), [apiBase, apiToken]);

  function persistApiBase(value: string) {
    setApiBase(value);
    saveApiBase(value);
  }
  function persistApiToken(value: string) {
    setApiToken(value);
    saveApiToken(value);
  }
  function persistTrayAnswer(value: string) {
    setTrayAnswer(value);
    saveTrayAnswer(value);
  }
  function persistSelection(next: Map<string, TrayItem>) {
    setSelection(next);
    saveSelection(next);
  }

  function toggleSelect(hit: SearchHit, checked: boolean) {
    const next = new Map(selection);
    if (checked) {
      next.set(hit.scene_id, {
        scene_id: hit.scene_id,
        video_id: hit.video_id,
        score: hit.score,
        best_keyframe_id: hit.best_keyframe_id,
        best_timestamp_sec: hit.best_timestamp_sec,
        start_sec: hit.start_sec,
      });
    } else {
      next.delete(hit.scene_id);
    }
    persistSelection(next);
  }

  function removeFromTray(sceneId: string) {
    const next = new Map(selection);
    next.delete(sceneId);
    persistSelection(next);
  }

  function clearTray() {
    persistSelection(new Map());
  }

  async function runSearch(overrides: { filters?: { video_ids: string[] } } = {}) {
    setStatus(
      overrides.filters ? `Đang tìm lại trong ${overrides.filters.video_ids.length} video đã chọn…` : "Đang tìm kiếm…"
    );
    setSearchResult(null);
    setVqaResult(null);
    sceneDetailCache.clear();
    try {
      if (task === "vqa") {
        const response = await vqaCall(apiConfig, {
          question: query,
          top_k_evidence: topK,
          debug,
          ...(overrides.filters ? { filters: overrides.filters } : {}),
        });
        setVqaResult(response);
        setStatus(`Hoàn tất trong ${response.took_ms.toFixed(1)} ms`);
      } else {
        const response = await search(apiConfig, task, {
          query,
          top_k: topK,
          debug,
          ...(overrides.filters ? { filters: overrides.filters } : {}),
        });
        setSearchResult(response);
        const count = task === "sequence" ? response.sequences.length : response.results.length;
        const label = task === "sequence" ? "chuỗi" : "kết quả";
        setStatus(`${count} ${label} · ${response.took_ms.toFixed(1)} ms`);
      }
    } catch (error) {
      setStatus(`Lỗi: ${error instanceof ApiError ? error.message : String(error)}`);
    }
  }

  function refineToSelection() {
    const videoIds = [...new Set([...selection.values()].map((item) => item.video_id))];
    runSearch({ filters: { video_ids: videoIds } });
  }

  async function checkHealth() {
    setStatus("Đang kiểm tra server…");
    try {
      const data = await health(apiConfig);
      setStatus(`Server OK · ${data.backend} · ${data.scene_count} scenes`);
    } catch (error) {
      setStatus(`Không kết nối được: ${error instanceof ApiError ? error.message : String(error)}`);
    }
  }

  return (
    <main>
      <header>
        <p className="eyebrow">AIC 2026 · Online V1</p>
        <h1>Multimodal Video Search</h1>
        <p className="subtitle">KIS, AVS, chuỗi sự kiện và VQA trên cùng một luồng retrieval.</p>
      </header>
      <SearchForm
        apiBase={apiBase}
        onApiBaseChange={persistApiBase}
        apiToken={apiToken}
        onApiTokenChange={persistApiToken}
        task={task}
        onTaskChange={setTask}
        query={query}
        onQueryChange={setQuery}
        topK={topK}
        onTopKChange={setTopK}
        debug={debug}
        onDebugChange={setDebug}
        onSubmit={() => runSearch()}
        onHealthCheck={checkHealth}
      />
      <section id="status" aria-live="polite">
        {status}
      </section>
      <div className="layout">
        <section id="results">
          <ResultsView
            task={task}
            searchResult={searchResult}
            vqaResult={vqaResult}
            selection={selection}
            onToggleSelect={toggleSelect}
            apiConfig={apiConfig}
            sceneDetailCache={sceneDetailCache}
          />
        </section>
        <SelectionTray
          task={task}
          selection={selection}
          onRemove={removeFromTray}
          onClear={clearTray}
          onRefine={refineToSelection}
          trayAnswer={trayAnswer}
          onTrayAnswerChange={persistTrayAnswer}
          onStatus={setStatus}
        />
      </div>
      <DebugPanel queryPlan={searchResult?.query_plan ?? null} />
    </main>
  );
}

export default App;
