import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, health as fetchHealth, search, searchStream, unifiedSearch } from "./api";
import { AppFooter } from "./app/AppFooter";
import { AppShell } from "./app/AppShell";
import { LeftRail } from "./app/LeftRail";
import type { AppPage } from "./app/TopNavigation";
import { TopNavigation } from "./app/TopNavigation";
import { CompareLab } from "./components/CompareLab";
import { HealthDrawer } from "./components/HealthDrawer";
import { QueryStudio } from "./components/QueryStudio";
import { ResultsExplorer } from "./components/ResultsExplorer";
import { StreamLog } from "./components/StreamLog";
import { SubmissionBoard } from "./components/SubmissionBoard";
import { AvsWorkspace, KisWorkspace, QaWorkspace, TrakeWorkspace } from "./components/TaskWorkspaces";
import { DatasetStats } from "./features/search/DatasetStats";
import { PreviewPanel } from "./features/inspector/PreviewPanel";
import { WeightPanel } from "./features/weights/WeightPanel";
import { loadApiBase, loadApiToken, saveApiBase, saveApiToken } from "./storage";
import type { HealthResponse, SearchOptions, SearchResponse, StreamEvent, TaskType } from "./types";

// "query"/"evidence" không còn là tab riêng: QueryStudio+WeightPanel luôn
// hiện ở đầu trang search, Evidence chuyển vào tab của PreviewPanel (docs
// §4 "Workbench ba cột" — center chỉ còn kết quả theo task).
const TABS = ["results", "kis", "qa", "trake", "avs", "submission", "compare", "health"] as const;
type Tab = (typeof TABS)[number];

const TAB_LABELS: Record<Tab, string> = {
  results: "Kết quả",
  kis: "KIS",
  qa: "QA",
  trake: "TRAKE",
  avs: "AVS",
  submission: "Submission Board",
  compare: "Compare Lab",
  health: "Health",
};

function App() {
  const [apiBase, setApiBase] = useState(loadApiBase);
  const [apiToken, setApiToken] = useState(loadApiToken);
  const [page, setPage] = useState<AppPage>("search");
  const [healthState, setHealthState] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<"checking" | "ok" | "error">("checking");
  const [tab, setTab] = useState<Tab>("results");
  const [task, setTask] = useState<TaskType>("TEXTUAL_KIS");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(20);
  const [debug, setDebug] = useState(false);
  const [streaming, setStreaming] = useState(false);
  // Draft/applied: kéo slider trong WeightPanel chỉ đổi draftOptions; chỉ khi
  // search thực sự chạy (runSearch) thì draft mới trở thành "applied" —
  // hasUnsavedChanges cho biết còn thay đổi chưa gửi lên server (docs §14).
  const [draftOptions, setDraftOptions] = useState<SearchOptions>({});
  const [appliedOptions, setAppliedOptions] = useState<SearchOptions>({});
  const hasUnsavedChanges = JSON.stringify(draftOptions) !== JSON.stringify(appliedOptions);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [selectedSequenceIndex, setSelectedSequenceIndex] = useState(0);
  const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [submitting, setSubmitting] = useState(false);
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
    setSelectedSequenceIndex(0);
    setActiveStepIndex(null);
    // Apply & Search: draft chỉ thực sự áp dụng khi search chạy thật.
    setAppliedOptions(draftOptions);
    const body = { query, task, top_k: topK, debug, search_options: draftOptions };
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
    setHealthStatus("checking");
    try {
      const data = await fetchHealth(apiConfig);
      setHealthState(data);
      setHealthStatus("ok");
      setStatus(`Server OK · ${data.backend} · ${data.scene_count} scenes`);
    } catch (error) {
      setHealthStatus("error");
      setStatus(`Không kết nối được: ${error instanceof ApiError ? error.message : String(error)}`);
    }
  }

  // Nạp health/dataset stats ngay khi mở app hoặc đổi API base — footer và
  // dataset stats cards cần dữ liệu thật ngay cả khi người dùng chưa bấm
  // "Kiểm tra server" thủ công.
  useEffect(() => {
    let cancelled = false;
    setHealthStatus("checking");
    fetchHealth(apiConfig)
      .then((data) => {
        if (cancelled) return;
        setHealthState(data);
        setHealthStatus("ok");
      })
      .catch(() => {
        if (!cancelled) setHealthStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [apiConfig]);

  return (
    <AppShell
      topNav={<TopNavigation page={page} onPageChange={setPage} backendStatus={healthStatus} backendLabel={healthState?.backend ?? apiBase} />}
      leftRail={<LeftRail page={page} onPageChange={setPage} />}
      footer={<AppFooter health={healthState} />}
    >
      {page === "search" && (
        <>
          {/* Query section: controls (trái) + dataset stats (phải), cùng
              hàng trên desktop — bỏ header lặp lại thương hiệu (đã có ở
              TopNavigation) để gọn hơn (docs §4.2, yêu cầu "tối giản"). */}
          <div className="query-section">
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
              parsedEvents={result?.query_plan?.events ?? []}
            />
            <DatasetStats health={healthState} />
          </div>

          {status && (
            <section id="status" aria-live="polite">
              {status}
            </section>
          )}
          {streaming && <StreamLog events={streamEvents} />}

          {/* Workbench ba cột: Weights (trái) | kết quả theo task (giữa) |
              Preview & Details (phải, sticky) — docs §4.3. */}
          <div className="search-workbench">
            <aside className="workbench-left">
              <WeightPanel
                apiConfig={apiConfig}
                task={task}
                draftOptions={draftOptions}
                onDraftChange={setDraftOptions}
                hasUnsavedChanges={hasUnsavedChanges}
                parsedEvents={result?.query_plan?.events ?? []}
              />
            </aside>

            <section className="workbench-center">
              <nav className="tab-bar">
                {TABS.map((item) => (
                  <button key={item} type="button" className={tab === item ? "tab active" : "tab"} onClick={() => setTab(item)}>
                    {TAB_LABELS[item]}
                  </button>
                ))}
              </nav>

              {tab === "results" && (
                <ResultsExplorer results={result?.results ?? []} sequences={result?.sequences ?? []} apiConfig={apiConfig} />
              )}
              {tab === "kis" && <KisWorkspace items={result?.kis ?? []} />}
              {tab === "qa" && <QaWorkspace items={result?.qa ?? []} />}
              {tab === "trake" && (
                <TrakeWorkspace
                  items={result?.trake ?? []}
                  stepQueries={(result?.query_plan?.events ?? []).map((event) => event.text)}
                  apiConfig={apiConfig}
                  selectedIndex={selectedSequenceIndex}
                  onSelectSequence={(index) => {
                    setSelectedSequenceIndex(index);
                    setActiveStepIndex(null);
                  }}
                  activeStepIndex={activeStepIndex}
                  onSelectStep={setActiveStepIndex}
                />
              )}
              {tab === "avs" && <AvsWorkspace items={result?.avs ?? []} />}
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
            </section>

            <aside className="workbench-right">
              <PreviewPanel
                apiConfig={apiConfig}
                result={result}
                selectedSequence={result?.trake?.[selectedSequenceIndex] ?? null}
                activeStepIndex={activeStepIndex}
              />
            </aside>
          </div>
        </>
      )}

      {page === "history" && <CompareLab apiConfig={apiConfig} />}

      {page === "dataset" && (
        <>
          <h2>Dataset</h2>
          <DatasetStats health={healthState} />
          <p className="muted">
            Dataset: {healthState?.dataset ?? "—"} · build {healthState?.dataset_version ?? "—"}
          </p>
        </>
      )}

      {page === "submission" && (
        <SubmissionBoard
          apiConfig={apiConfig}
          task={task}
          kis={result?.kis ?? []}
          qa={result?.qa ?? []}
          trake={result?.trake ?? []}
          avs={result?.avs ?? []}
        />
      )}

      {page === "system" && <HealthDrawer apiConfig={apiConfig} />}
    </AppShell>
  );
}

export default App;
