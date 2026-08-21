import { useEffect, useMemo, useRef, useState } from "react";
import { LayoutGrid, ListChecks, SlidersHorizontal, Upload } from "lucide-react";
import { ApiError, health as fetchHealth, search, searchStream, unifiedSearch } from "./api";
import { AppFooter } from "./app/AppFooter";
import { AppShell } from "./app/AppShell";
import { LeftRail } from "./app/LeftRail";
import type { AppPage } from "./app/TopNavigation";
import { TopNavigation } from "./app/TopNavigation";
import { CompareLab } from "./components/CompareLab";
import type { TunerRow } from "./components/FrameTuner";
import { FrameTuner } from "./components/FrameTuner";
import { HealthDrawer } from "./components/HealthDrawer";
import { QueryStudio } from "./components/QueryStudio";
import { ResultsExplorer } from "./components/ResultsExplorer";
import { StreamLog } from "./components/StreamLog";
import { SubmissionBoard } from "./components/SubmissionBoard";
import { AvsWorkspace, KisWorkspace, QaWorkspace, TrakeWorkspace } from "./components/TaskWorkspaces";
import { PreviewPanel } from "./features/inspector/PreviewPanel";
import { DatasetStats } from "./features/search/DatasetStats";
import { WeightPanel } from "./features/weights/WeightPanel";
import { loadApiBase, loadApiToken, saveApiBase, saveApiToken } from "./storage";
import type { HealthResponse, SearchOptions, SearchResponse, StreamEvent, TaskType } from "./types";
import { PanelBody, PanelHeader, Surface, Tabs } from "./ui";
import type { TabItem } from "./ui";

type ResultTab = "results" | "task" | "submission" | "tuner";

/** Demo mode (`?demo=1`): tự chạy MỘT search THẬT lên backend ngay khi mở, để
 * người xem thấy giao diện ở trạng thái có dữ liệu mà không phải gõ gì.
 * KHÔNG có fixture giả nào — nếu backend không chạy thì vẫn ra empty state
 * đúng như production, và nav hiện badge "demo" để không ai nhầm. */
const DEMO_QUERY = "cảnh báo sạt lở nguy hiểm ven sông";

function isDemoMode(): boolean {
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).get("demo") === "1";
}

/** Nhãn tab "task" đổi theo task đang chọn — một tab duy nhất thay vì bốn tab
 * KIS/QA/TRAKE/AVS luôn hiện (ba trong bốn luôn rỗng, chỉ tạo nhiễu). */
const TASK_TAB_LABEL: Record<TaskType, string> = {
  TEXTUAL_KIS: "KIS frames",
  QA: "QA answers",
  TRAKE: "Sequences",
  AVS: "Segments",
};

function App() {
  const [apiBase, setApiBase] = useState(loadApiBase);
  const [apiToken, setApiToken] = useState(loadApiToken);
  const [page, setPage] = useState<AppPage>("search");
  // Dòng do người dùng đẩy sang từ bảng nộp ("Lưu & chỉnh frame"). `null` =
  // chưa đẩy gì, tab chỉnh frame nạp từ kết quả tìm kiếm như trước.
  const [tunerHandoff, setTunerHandoff] = useState<TunerRow[] | null>(null);
  // Tăng mỗi lần bấm "Lưu & chỉnh frame". FrameTuner nạp khi số này đổi, nên
  // bấm lại lần nữa vẫn nạp lại được dù danh sách giống hệt.
  const [tunerHandoffKey, setTunerHandoffKey] = useState(0);
  const [healthState, setHealthState] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<"checking" | "ok" | "error">("checking");
  const [resultTab, setResultTab] = useState<ResultTab>("results");
  const [task, setTask] = useState<TaskType>("TEXTUAL_KIS");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(20);
  const [debug, setDebug] = useState(false);
  const [streaming, setStreaming] = useState(false);

  // Draft/applied: kéo slider chỉ đổi draftOptions; chỉ khi search thật chạy
  // thì draft mới thành "applied" — nên KHÔNG có search nào bị kích hoạt
  // trong lúc người dùng đang kéo.
  const [draftOptions, setDraftOptions] = useState<SearchOptions>({});
  const [appliedOptions, setAppliedOptions] = useState<SearchOptions>({});
  const hasUnsavedChanges = JSON.stringify(draftOptions) !== JSON.stringify(appliedOptions);

  const [status, setStatus] = useState("");
  const [statusIsError, setStatusIsError] = useState(false);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [selectedSequenceIndex, setSelectedSequenceIndex] = useState(0);
  const [activeStepIndex, setActiveStepIndex] = useState<number | null>(null);
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const demo = useRef(isDemoMode()).current;
  const demoRan = useRef(false);

  const apiConfig = useMemo(() => ({ base: apiBase, token: apiToken }), [apiBase, apiToken]);

  // Phải tìm trong CẢ `sequences`, không chỉ `results`: endpoint TRAKE trả
  // `results: []` (đo được: 0 phần tử) và toàn bộ candidate nằm trong
  // `sequences[].scenes`. Bản cũ chỉ tra `results` nên với TRAKE `selectedHit`
  // luôn là null, và panel xem không bao giờ theo được thẻ vừa bấm.
  const selectedHit = useMemo(() => {
    if (!selectedCandidateId) return null;
    const direct = result?.results.find((hit) => hit.candidate_id === selectedCandidateId);
    if (direct) return direct;
    for (const sequence of result?.sequences ?? []) {
      const scene = sequence.scenes.find((hit) => hit.candidate_id === selectedCandidateId);
      if (scene) return scene;
    }
    return null;
  }, [result, selectedCandidateId]);

  function persistApiBase(value: string) {
    setApiBase(value);
    saveApiBase(value);
  }
  function persistApiToken(value: string) {
    setApiToken(value);
    saveApiToken(value);
  }

  function resultCount(response: SearchResponse): number {
    return (
      response.results.length ||
      response.kis.length ||
      response.qa.length ||
      response.trake.length ||
      response.avs.length
    );
  }

  async function runSearch(queryOverride?: string) {
    // Chỉ nhận string: nếu hàm này lỡ bị gắn thẳng làm event handler thì tham
    // số sẽ là một SyntheticEvent, và nó phải bị bỏ qua chứ không được đi
    // tiếp vào request body.
    const effectiveQuery = typeof queryOverride === "string" ? queryOverride : query;
    setSubmitting(true);
    setStatusIsError(false);
    setStatus(streaming ? "Đang stream…" : "Đang tìm kiếm…");
    setResult(null);
    setStreamEvents([]);
    setSelectedCandidateId(null);
    setSelectedSequenceIndex(0);
    setActiveStepIndex(null);
    setAppliedOptions(draftOptions);
    setHasSearched(true);

    const body = { query: effectiveQuery, task, top_k: topK, debug, search_options: draftOptions };
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
              setSelectedCandidateId(event.response.results[0]?.candidate_id ?? null);
              setStatus(`${resultCount(event.response)} kết quả · ${event.response.took_ms.toFixed(0)} ms · stream`);
            } else if (event.type === "error") {
              setStatusIsError(true);
              setStatus(event.message);
            }
          },
          controller.signal
        );
      } else {
        const response = await unifiedSearch(apiConfig, body).catch(() => search(apiConfig, task, body));
        setResult(response);
        // Tự chọn kết quả đầu: rail phải luôn có nội dung thật ngay sau khi
        // search, thay vì để một cột rỗng bắt người dùng đoán phải bấm gì.
        // TRAKE không dùng `results` mà dùng `trake` → chọn bước 1 của chuỗi
        // đứng đầu, nếu không Preview sẽ rỗng đúng ở task cần nó nhất.
        setSelectedCandidateId(response.results[0]?.candidate_id ?? null);
        if (response.trake.length > 0) setActiveStepIndex(0);
        setStatus(
          `${resultCount(response)} kết quả · ${response.took_ms.toFixed(0)} ms` +
            (response.status === "COMPLETED_WITH_WARNINGS" ? ` · ${response.warnings.length} cảnh báo` : "")
        );
        setResultTab("results");
      }
    } catch (error) {
      setStatusIsError(true);
      setStatus(error instanceof ApiError ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  async function checkHealth() {
    setHealthStatus("checking");
    try {
      const data = await fetchHealth(apiConfig);
      setHealthState(data);
      setHealthStatus("ok");
      setStatusIsError(false);
      setStatus(`Server OK · ${data.backend} · ${data.scene_count} scenes`);
    } catch (error) {
      setHealthStatus("error");
      setStatusIsError(true);
      setStatus(`Không kết nối được: ${error instanceof ApiError ? error.message : String(error)}`);
    }
  }

  // Nạp health ngay khi mở app — footer/stat cards cần dữ liệu thật kể cả khi
  // người dùng chưa bấm "Kiểm tra server".
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

  // Demo mode: chờ health OK rồi chạy đúng một search thật, một lần duy nhất.
  useEffect(() => {
    if (!demo || demoRan.current || healthStatus !== "ok") return;
    demoRan.current = true;
    setQuery(DEMO_QUERY);
    void runSearch(DEMO_QUERY);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [demo, healthStatus]);

  const taskItemCount =
    task === "TEXTUAL_KIS"
      ? result?.kis.length
      : task === "QA"
        ? result?.qa.length
        : task === "TRAKE"
          ? result?.trake.length
          : result?.avs.length;

  // Dòng nạp sẵn cho tab chỉnh frame. TRAKE tách MỖI BƯỚC thành một dòng —
  // người chấm soát từng khoảnh khắc, không soát cả chuỗi một lượt.
  // Truy vấn mới thì bản đã đẩy sang không còn đúng nữa. Không xoá thì người
  // dùng chỉnh danh sách của truy vấn TRƯỚC mà không có gì báo.
  useEffect(() => { setTunerHandoff(null); }, [result]);

  const tunerSeed = useMemo<TunerRow[]>(() => {
    if (!result) return [];
    if (task === "TEXTUAL_KIS") {
      return result.kis.map((item, index) => ({
        id: `kis-${index}`, videoId: item.video_id,
        originalFrame: item.frame_idx, frame: item.frame_idx,
      }));
    }
    if (task === "QA") {
      return result.qa.map((item, index) => ({
        id: `qa-${index}`, videoId: item.video_id,
        originalFrame: item.frame_idx, frame: item.frame_idx, answer: item.answer,
      }));
    }
    if (task === "TRAKE") {
      // Đánh số bước theo `steps[].step` của backend, KHÔNG theo vị trí trong
      // `frame_ids`: chuỗi được phép thiếu bước ở giữa, nên phần tử thứ 3 của
      // mảng không còn chắc chắn là bước 3. Đánh theo vị trí là gán nhầm frame
      // cho bước — sai lặng lẽ, và người dùng chỉnh đúng frame vào nhầm ô.
      return result.trake.flatMap((item, chain) => {
        const known = new Map(item.steps.map((entry) => [entry.step, entry]));
        const total = Math.max(
          0,
          ...item.steps.map((entry) => entry.step),
          ...(item.missing_steps ?? []),
        );
        const rows: TunerRow[] = [];
        for (let step = 1; step <= total; step += 1) {
          const entry = known.get(step);
          if (entry) {
            rows.push({
              id: `trake-${chain}-${step}`, videoId: item.video_id,
              originalFrame: entry.frame_idx, frame: entry.frame_idx,
              chain: chain + 1, step,
              placeholder: entry.refinement === "interpolated",
            });
            continue;
          }
          // Bước không tìm được gì: dựng sẵn một dòng ở ĐIỂM GIỮA hai mốc lân
          // cận để người dùng có chỗ bám mà kéo, thay vì phải tự nhớ là chuỗi
          // thiếu bước nào rồi gõ số từ đầu.
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
          rows.push({
            id: `trake-${chain}-${step}`, videoId: item.video_id,
            originalFrame: guess, frame: guess, chain: chain + 1, step,
            placeholder: true,
          });
        }
        return rows;
      });
    }
    return [];
  }, [result, task]);

  const resultTabs: TabItem<ResultTab>[] = [
    { value: "results", label: "Lưới ảnh", count: result?.results.length, icon: <LayoutGrid size={13} /> },
    { value: "task", label: TASK_TAB_LABEL[task], count: taskItemCount, icon: <ListChecks size={13} /> },
    { value: "submission", label: "Submission", icon: <Upload size={13} /> },
    { value: "tuner", label: "Chỉnh frame", count: tunerSeed.length || undefined, icon: <SlidersHorizontal size={13} /> },
  ];

  return (
    <AppShell
      nav={
        <TopNavigation
          page={page}
          onPageChange={setPage}
          backendStatus={healthStatus}
          backendLabel={healthState?.backend ?? apiBase}
          demo={demo}
        />
      }
      rail={<LeftRail page={page} onPageChange={setPage} />}
      footer={<AppFooter health={healthState} />}
    >
      {page === "search" && (
        <div className="studio">
            <QueryStudio
              aside={<DatasetStats health={healthState} loading={healthStatus === "checking" && !healthState} />}
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
              // Bọc trong arrow: truyền thẳng `runSearch` sẽ khiến React đưa
              // MouseEvent vào tham số queryOverride (JSON.stringify vòng lặp).
              onSubmit={() => runSearch()}
              onHealthCheck={checkHealth}
              submitting={submitting}
              parsedEvents={result?.query_plan?.events ?? []}
            />

            {status && (
              <div className={statusIsError ? "status-strip is-error" : "status-strip"} role="status" aria-live="polite">
                {status}
              </div>
            )}

            {streaming && streamEvents.length > 0 && <StreamLog events={streamEvents} />}

            <div className="studio-body">
              <WeightPanel
                apiConfig={apiConfig}
                task={task}
                draftOptions={draftOptions}
                onDraftChange={setDraftOptions}
                hasUnsavedChanges={hasUnsavedChanges}
                parsedEvents={result?.query_plan?.events ?? []}
              />

              <Surface fill className="results-panel">
                <PanelHeader
                  title="Kết quả"
                  meta={result ? `${result.took_ms.toFixed(0)} ms` : undefined}
                />
                <Tabs ariaLabel="Chế độ xem kết quả" value={resultTab} onChange={setResultTab} items={resultTabs} />

                {resultTab === "results" && (
                  <ResultsExplorer
                    results={result?.results ?? []}
                    sequences={result?.sequences ?? []}
                    apiConfig={apiConfig}
                    selectedCandidateId={selectedCandidateId}
                    onSelect={setSelectedCandidateId}
                    onSelectSequenceStep={(sequenceIndex, stepIndex) => {
                      setSelectedSequenceIndex(sequenceIndex);
                      setActiveStepIndex(stepIndex);
                    }}
                    pristine={!hasSearched}
                    topK={topK}
                    perVideoCapSet={appliedOptions.fusion?.max_results_per_video != null}
                  />
                )}

                {resultTab === "task" && task === "TEXTUAL_KIS" && <KisWorkspace items={result?.kis ?? []} />}
                {resultTab === "task" && task === "QA" && <QaWorkspace items={result?.qa ?? []} />}
                {resultTab === "task" && task === "TRAKE" && (
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
                {resultTab === "task" && task === "AVS" && <AvsWorkspace items={result?.avs ?? []} />}

                {resultTab === "tuner" && (
                  <FrameTuner
                    apiConfig={apiConfig}
                    task={task}
                    seedRows={tunerHandoff ?? tunerSeed}
                    autoLoadKey={tunerHandoff ? tunerHandoffKey : undefined}
                  />
                )}

                {resultTab === "submission" && (
                  <PanelBody>
                    <SubmissionBoard
                      apiConfig={apiConfig}
                      task={task}
                      kis={result?.kis ?? []}
                      qa={result?.qa ?? []}
                      trake={result?.trake ?? []}
                      avs={result?.avs ?? []}
                      results={result?.results ?? []}
                      onSelectSequence={(index) => {
                        setSelectedSequenceIndex(index);
                        setActiveStepIndex(null);
                      }}
                      onEditRows={(rows) => {
                        setTunerHandoff(rows);
                        setTunerHandoffKey((value) => value + 1);
                        setResultTab("tuner");
                      }}
                    />
                  </PanelBody>
                )}
              </Surface>

              <PreviewPanel
                apiConfig={apiConfig}
                result={result}
                selectedSequence={result?.trake?.[selectedSequenceIndex] ?? null}
                activeStepIndex={activeStepIndex}
                selectedHit={selectedHit}
                onSelectStep={setActiveStepIndex}
              />
            </div>
        </div>
      )}

      {page === "analytics" && (
        <div className="page-simple">
          <header className="page-simple-head">
            <h2 className="page-simple-title">Session Analytics</h2>
            <p className="page-simple-sub">So sánh trace của hai lần search (thường là bản gốc và bản replay).</p>
          </header>
          <div className="page-simple-body">
            <Surface fill>
              <PanelBody>
                <CompareLab apiConfig={apiConfig} />
              </PanelBody>
            </Surface>
          </div>
        </div>
      )}

      {page === "dataset" && (
        <div className="page-simple">
          <header className="page-simple-head">
            <h2 className="page-simple-title">Dataset</h2>
            <p className="page-simple-sub">
              {healthState?.dataset ?? "—"} · build {healthState?.dataset_version ?? "—"}
            </p>
          </header>
          <DatasetStats health={healthState} loading={healthStatus === "checking" && !healthState} />
          <div className="page-simple-body">
            <Surface fill>
              <PanelHeader title="Trạng thái hệ thống" />
              <PanelBody>
                <HealthDrawer apiConfig={apiConfig} />
              </PanelBody>
            </Surface>
          </div>
        </div>
      )}

      {page === "submission" && (
        <div className="page-simple">
          <header className="page-simple-head">
            <h2 className="page-simple-title">Submission Board</h2>
            <p className="page-simple-sub">Build và validate CSV qua backend (POST /v1/submissions/*).</p>
          </header>
          <div className="page-simple-body">
            <Surface fill>
              <PanelBody>
                <SubmissionBoard
                  apiConfig={apiConfig}
                  task={task}
                  kis={result?.kis ?? []}
                  qa={result?.qa ?? []}
                  trake={result?.trake ?? []}
                  avs={result?.avs ?? []}
                  results={result?.results ?? []}
                  onEditRows={(rows) => {
                    setTunerHandoff(rows);
                    setTunerHandoffKey((value) => value + 1);
                    setPage("search");
                    setResultTab("tuner");
                  }}
                />
              </PanelBody>
            </Surface>
          </div>
        </div>
      )}

      {page === "system" && (
        <div className="page-simple">
          <header className="page-simple-head">
            <h2 className="page-simple-title">System</h2>
            <p className="page-simple-sub">Health, branch đang đăng ký và tầng rerank thật sự có sẵn.</p>
          </header>
          <div className="page-simple-body">
            <Surface fill>
              <PanelBody>
                <HealthDrawer apiConfig={apiConfig} />
              </PanelBody>
            </Surface>
          </div>
        </div>
      )}
    </AppShell>
  );
}

export default App;
