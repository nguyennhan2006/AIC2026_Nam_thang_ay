import { useMemo, useRef, useState } from "react";
import { ApiError, health, search, searchStream, unifiedSearch } from "./api";
import { BranchStatusPanel } from "./components/BranchStatusPanel";
import { CompareLab } from "./components/CompareLab";
import { EvidenceInspector } from "./components/EvidenceInspector";
import { HealthDrawer } from "./components/HealthDrawer";
import { MixingConsole } from "./components/MixingConsole";
import { QueryStudio } from "./components/QueryStudio";
import { ResultsExplorer } from "./components/ResultsExplorer";
import { StreamLog } from "./components/StreamLog";
import { SubmissionBoard } from "./components/SubmissionBoard";
import { AvsWorkspace, KisWorkspace, QaWorkspace, TrakeWorkspace } from "./components/TaskWorkspaces";
import { loadApiBase, loadApiToken, saveApiBase, saveApiToken } from "./storage";
import type { SearchOptions, SearchResponse, StreamEvent, TaskType } from "./types";

const TABS = [
  "query",
  "results",
  "kis",
  "qa",
  "trake",
  "avs",
  "evidence",
  "submission",
  "compare",
  "health",
] as const;
type Tab = (typeof TABS)[number];

const TAB_LABELS: Record<Tab, string> = {
  query: "Query Studio",
  results: "Results Explorer",
  kis: "KIS Safe Frame",
  qa: "QA Evidence",
  trake: "TRAKE Alignment",
  avs: "AVS Relevance",
  evidence: "Evidence Inspector",
  submission: "Submission Board",
  compare: "Compare Lab",
  health: "Health",
};

function App() {
  const [apiBase, setApiBase] = useState(loadApiBase);
  const [apiToken, setApiToken] = useState(loadApiToken);
  const [tab, setTab] = useState<Tab>("query");
  const [task, setTask] = useState<TaskType>("TEXTUAL_KIS");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(20);
  const [debug, setDebug] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>({});
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [evidenceCandidateId, setEvidenceCandidateId] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const apiConfig = useMemo(() => ({ base: apiBase, token: apiToken }), [apiBase, apiToken]);

  function persistApiBase(value: string) {
    setApiBase(value);
    saveApiBase(value);
  }
  function persistApiToken(value: string) {
    setApiToken(value);
    saveApiToken(value);
  }

  async function runSearch() {
    setSubmitting(true);
    setStatus(streaming ? "Đang stream…" : "Đang tìm kiếm…");
    setResult(null);
    setStreamEvents([]);
    const body = { query, task, top_k: topK, debug, search_options: searchOptions };
    try {
      if (streaming) {
        abortRef.current?.abort();
        const controller = new AbortController();
        abortRef.current = controller;
        await searchStream(
          apiConfig,
          body,
          (event) => {
            setStreamEvents((prev) => [...prev, event]);
            if (event.type === "search_completed") {
              setResult(event.response);
              setStatus(`${event.response.results.length || event.response.kis.length || event.response.qa.length || event.response.trake.length || event.response.avs.length} kết quả · ${event.response.took_ms.toFixed(1)} ms (stream)`);
            } else if (event.type === "error") {
              setStatus(`Lỗi: ${event.message}`);
            }
          },
          controller.signal
        );
      } else {
        const response = await unifiedSearch(apiConfig, body).catch(() => search(apiConfig, task, body));
        setResult(response);
        const count = response.results.length || response.kis.length || response.qa.length || response.trake.length || response.avs.length;
        setStatus(
          `${count} kết quả · ${response.took_ms.toFixed(1)} ms` +
            (response.status === "COMPLETED_WITH_WARNINGS" ? " · có cảnh báo" : "")
        );
        setTab("results");
      }
    } catch (error) {
      setStatus(`Lỗi: ${error instanceof ApiError ? error.message : String(error)}`);
    } finally {
      setSubmitting(false);
    }
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
        <p className="eyebrow">AIC 2026 · Competition Retrieval Studio</p>
        <h1>Universal Multimodal Retrieval Engine</h1>
        <p className="subtitle">
          Một lõi retrieval dùng chung + bốn task processor chuyên biệt (Textual KIS, Q&amp;A, TRAKE, AVS).
        </p>
      </header>

      <nav className="tab-bar">
        {TABS.map((item) => (
          <button key={item} type="button" className={tab === item ? "tab active" : "tab"} onClick={() => setTab(item)}>
            {TAB_LABELS[item]}
          </button>
        ))}
      </nav>

      <section id="status" aria-live="polite">
        {status}
      </section>

      {tab === "query" && (
        <>
          <QueryStudio
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
            streaming={streaming}
            onStreamingChange={setStreaming}
            onSubmit={runSearch}
            onHealthCheck={checkHealth}
            submitting={submitting}
          />
          <MixingConsole apiConfig={apiConfig} searchOptions={searchOptions} onChange={setSearchOptions} />
          {streaming && <StreamLog events={streamEvents} />}
          {result && <BranchStatusPanel statuses={result.branch_status} />}
        </>
      )}

      {tab === "results" && (
        <>
          {result && <BranchStatusPanel statuses={result.branch_status} />}
          <ResultsExplorer results={result?.results ?? []} sequences={result?.sequences ?? []} apiConfig={apiConfig} />
        </>
      )}

      {tab === "kis" && <KisWorkspace items={result?.kis ?? []} />}
      {tab === "qa" && <QaWorkspace items={result?.qa ?? []} />}
      {tab === "trake" && <TrakeWorkspace items={result?.trake ?? []} />}
      {tab === "avs" && <AvsWorkspace items={result?.avs ?? []} />}

      {tab === "evidence" && (
        <div>
          <label>
            candidate_id (scene_id hoặc keyframe_id)
            <input value={evidenceCandidateId} onChange={(e) => setEvidenceCandidateId(e.target.value)} placeholder="L21_V001_S0012" />
          </label>
          <EvidenceInspector apiConfig={apiConfig} candidateId={evidenceCandidateId || null} />
        </div>
      )}

      {tab === "submission" && (
        <SubmissionBoard
          apiConfig={apiConfig}
          task={task}
          kis={result?.kis ?? []}
          qa={result?.qa ?? []}
          trake={result?.trake ?? []}
          avs={result?.avs ?? []}
        />
      )}

      {tab === "compare" && <CompareLab apiConfig={apiConfig} />}
      {tab === "health" && <HealthDrawer apiConfig={apiConfig} />}
    </main>
  );
}

export default App;
