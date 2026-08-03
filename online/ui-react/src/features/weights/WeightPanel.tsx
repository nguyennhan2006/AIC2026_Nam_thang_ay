import { useEffect, useMemo, useState } from "react";
import type { ApiClientConfig } from "../../api";
import { ApiError, getCapabilities } from "../../api";
import type { CapabilitiesResponse, QueryPlan, SearchOptions, TaskType } from "../../types";
import { normalizeStepColumn, normalizeWeights } from "./weightMath";
import type { WeightControlValue } from "./weightMath";
import {
  allPresets,
  deleteUserPreset,
  newPresetId,
  upsertUserPreset,
} from "./presets";
import type { SearchPreset } from "./presets";

export interface WeightPanelProps {
  apiConfig: ApiClientConfig;
  task: TaskType;
  draftOptions: SearchOptions;
  onDraftChange: (options: SearchOptions) => void;
  hasUnsavedChanges: boolean;
  parsedEvents: QueryPlan["events"];
}

const MODALITY_LABELS: Record<string, string> = {
  visual: "Visual (CLIP)",
  caption: "Caption",
  ocr: "OCR",
  asr: "ASR",
  keyword: "Keyword",
  object: "Objects / Actions",
  action: "Objects / Actions",
  color: "Color",
  event: "Event Metadata",
};

function branchWeight(options: SearchOptions, branchId: string): number {
  return options.branches?.[branchId]?.weight ?? 1;
}

function branchEnabled(options: SearchOptions, branchId: string): boolean {
  return options.branches?.[branchId]?.enabled ?? true;
}

/** Weight Lab — trọng tâm của lần chỉnh UI này (docs §9). Mọi branch/control
 * đọc từ GET /v1/search/capabilities, không hard-code — branch không có
 * trong danh sách thì không render control nào cho nó. */
export function WeightPanel({ apiConfig, task, draftOptions, onDraftChange, hasUnsavedChanges, parsedEvents }: WeightPanelProps) {
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stepTab, setStepTab] = useState<"global" | "per_step">("global");
  const [presetName, setPresetName] = useState("");
  const [presets, setPresets] = useState<SearchPreset[]>(() => allPresets());
  const [normalizeError, setNormalizeError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getCapabilities(apiConfig)
      .then((result) => {
        if (!cancelled) setCapabilities(result);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [apiConfig]);

  const branchLocks = useMemo(() => new Set<string>(), []);
  const [locked, setLocked] = useState<Set<string>>(branchLocks);

  function updateBranch(branchId: string, patch: { enabled?: boolean; weight?: number; top_k?: number; timeout_ms?: number }) {
    const branches = { ...(draftOptions.branches ?? {}) };
    branches[branchId] = { ...branches[branchId], ...patch };
    onDraftChange({ ...draftOptions, branches });
  }

  function toggleLock(branchId: string) {
    const next = new Set(locked);
    if (next.has(branchId)) next.delete(branchId);
    else next.add(branchId);
    setLocked(next);
  }

  function handleNormalize() {
    if (!capabilities) return;
    setNormalizeError(null);
    const values: WeightControlValue[] = capabilities.branches.map((branch) => ({
      branchId: branch.branch_id,
      label: MODALITY_LABELS[branch.modality ?? ""] ?? branch.branch_id,
      enabled: branchEnabled(draftOptions, branch.branch_id),
      weight: branchWeight(draftOptions, branch.branch_id),
      locked: locked.has(branch.branch_id),
      available: branch.available,
      colorToken: "",
    }));
    try {
      const normalized = normalizeWeights(values);
      const branches = { ...(draftOptions.branches ?? {}) };
      for (const item of normalized) {
        branches[item.branchId] = { ...branches[item.branchId], weight: item.weight };
      }
      onDraftChange({ ...draftOptions, branches });
    } catch (err) {
      setNormalizeError(err instanceof Error ? err.message : String(err));
    }
  }

  function updateFusion(patch: Partial<NonNullable<SearchOptions["fusion"]>>) {
    onDraftChange({ ...draftOptions, fusion: { ...draftOptions.fusion, ...patch } });
  }

  function updateRerankText(patch: Partial<NonNullable<NonNullable<SearchOptions["rerank"]>["text"]>>) {
    onDraftChange({
      ...draftOptions,
      rerank: { ...draftOptions.rerank, text: { ...draftOptions.rerank?.text, ...patch } },
    });
  }

  function updateRerankVlm(patch: Partial<NonNullable<NonNullable<SearchOptions["rerank"]>["vlm"]>>) {
    onDraftChange({
      ...draftOptions,
      rerank: { ...draftOptions.rerank, vlm: { ...draftOptions.rerank?.vlm, ...patch } },
    });
  }

  function updateTemporal(patch: Partial<NonNullable<SearchOptions["temporal"]>>) {
    onDraftChange({ ...draftOptions, temporal: { ...draftOptions.temporal, ...patch } });
  }

  function updateStepWeight(stepIndex: number, modality: string, value: number) {
    const current = draftOptions.temporal?.step_modality_weights ?? [];
    const next = [...current];
    while (next.length <= stepIndex) next.push({});
    next[stepIndex] = { ...next[stepIndex], [modality]: value };
    updateTemporal({ step_modality_weights: next });
  }

  function normalizeStep(stepIndex: number) {
    const current = draftOptions.temporal?.step_modality_weights ?? [];
    if (!current[stepIndex]) return;
    const next = [...current];
    next[stepIndex] = normalizeStepColumn(next[stepIndex]);
    updateTemporal({ step_modality_weights: next });
  }

  function copyGlobalToAllSteps() {
    if (!capabilities) return;
    const globalRow: Record<string, number> = {};
    for (const branch of capabilities.branches) {
      if (!branch.modality) continue;
      globalRow[branch.modality] = branchWeight(draftOptions, branch.branch_id);
    }
    const next = parsedEvents.map(() => ({ ...globalRow }));
    updateTemporal({ step_modality_weights: next });
  }

  function applyPreset(preset: SearchPreset) {
    onDraftChange(preset.searchOptions);
  }

  function saveCurrentAsPreset() {
    if (!presetName.trim()) return;
    const preset: SearchPreset = {
      id: newPresetId(),
      name: presetName.trim(),
      task,
      version: "1",
      searchOptions: draftOptions,
      createdAt: new Date().toISOString(),
      source: "user",
    };
    upsertUserPreset(preset);
    setPresets(allPresets());
    setPresetName("");
  }

  function removePreset(id: string) {
    deleteUserPreset(id);
    setPresets(allPresets());
  }

  if (error) return <p className="muted">Không tải được capabilities: {error}</p>;
  if (!capabilities) return <p className="muted">Đang tải danh sách branch…</p>;

  const modalityBranches = capabilities.branches.filter((branch) => branch.modality);

  return (
    <div className="weight-panel">
      <div className="weight-panel-head">
        <h3>Trọng số (Weights)</h3>
        {hasUnsavedChanges && <span className="unsaved-badge">chưa áp dụng</span>}
      </div>

      <section className="weight-group">
        <h4>Nhóm Modalities</h4>
        {modalityBranches.map((branch) => {
          const weight = branchWeight(draftOptions, branch.branch_id);
          const enabled = branchEnabled(draftOptions, branch.branch_id);
          const isLocked = locked.has(branch.branch_id);
          return (
            <div key={branch.branch_id} className="weight-slider-row" title={branch.degraded ? branch.degraded_reason ?? "" : ""}>
              <label className="weight-row-label">
                <input
                  type="checkbox"
                  checked={enabled}
                  disabled={!branch.supported_controls.includes("enabled") && !branch.available}
                  onChange={(e) => updateBranch(branch.branch_id, { enabled: e.target.checked })}
                />
                {MODALITY_LABELS[branch.modality ?? ""] ?? branch.branch_id}
                {branch.degraded && <span className="degraded-badge">degraded</span>}
              </label>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={Math.min(weight, 1)}
                disabled={!enabled || !branch.available}
                onChange={(e) => updateBranch(branch.branch_id, { weight: Number(e.target.value) })}
              />
              <input
                type="number"
                min={0}
                max={10}
                step={0.01}
                className="narrow"
                value={weight}
                disabled={!enabled || !branch.available}
                onChange={(e) => updateBranch(branch.branch_id, { weight: Number(e.target.value) })}
              />
              <button
                type="button"
                className={isLocked ? "lock-btn locked" : "lock-btn"}
                title={isLocked ? "Đang khóa — bấm để mở" : "Khóa giá trị này khi Normalize"}
                onClick={() => toggleLock(branch.branch_id)}
              >
                {isLocked ? "🔒" : "🔓"}
              </button>
            </div>
          );
        })}
        <div className="weight-actions">
          <button type="button" onClick={handleNormalize}>
            Normalize
          </button>
          {normalizeError && <span className="warning-text">{normalizeError}</span>}
        </div>
      </section>

      <section className="weight-group">
        <h4>Fusion &amp; Ranking</h4>
        <label className="weight-field">
          Fusion method
          <select
            value={draftOptions.fusion?.method ?? "rrf"}
            onChange={(e) => updateFusion({ method: e.target.value as NonNullable<SearchOptions["fusion"]>["method"] })}
          >
            {capabilities.fusion_methods.map((method) => (
              <option key={method} value={method}>
                {method}
              </option>
            ))}
          </select>
        </label>
        <label className="weight-field">
          RRF k
          <input
            type="number"
            min={1}
            max={500}
            value={draftOptions.fusion?.rrf_k ?? 60}
            onChange={(e) => updateFusion({ rrf_k: Number(e.target.value) })}
          />
        </label>
        <label className="weight-field">
          Dedup scope
          <select
            value={draftOptions.fusion?.dedup_scope ?? "scene"}
            onChange={(e) => updateFusion({ dedup_scope: e.target.value as NonNullable<SearchOptions["fusion"]>["dedup_scope"] })}
          >
            <option value="none">none</option>
            <option value="frame">frame</option>
            <option value="scene">scene</option>
            <option value="event">event</option>
          </select>
        </label>
        <label className="weight-field" title="Số branch tối thiểu phải cùng thấy một candidate (fusion.minimum_matching_branches)">
          Min matching branches
          <input
            type="number"
            min={1}
            value={draftOptions.fusion?.minimum_matching_branches ?? 1}
            onChange={(e) => updateFusion({ minimum_matching_branches: Number(e.target.value) })}
          />
        </label>
        <label
          className="weight-field"
          title="KIS mặc định giới hạn 5 kết quả/video (dedup chống một video chiếm hết top-K khi dataset có nhiều video) — với dataset chỉ 1 video, giới hạn này khiến top_k cao hơn 5 không có tác dụng. Để trống = dùng mặc định của task."
        >
          Max results / video
          <input
            type="number"
            min={1}
            className="narrow"
            placeholder="mặc định theo task"
            value={draftOptions.fusion?.max_results_per_video ?? ""}
            onChange={(e) =>
              updateFusion({ max_results_per_video: e.target.value === "" ? null : Number(e.target.value) })
            }
          />
        </label>
      </section>

      <section className="weight-group">
        <h4>Rerank</h4>
        <label className="weight-row-label">
          <input
            type="checkbox"
            checked={draftOptions.rerank?.text?.enabled ?? capabilities.rerank.text}
            disabled={!capabilities.rerank.text}
            title={!capabilities.rerank.text ? "Chưa cấu hình text reranker (AIC_RERANK_TEXT_URL)" : ""}
            onChange={(e) => updateRerankText({ enabled: e.target.checked })}
          />
          Text rerank (cross-encoder)
        </label>
        {capabilities.rerank.text && (
          <div className="weight-inline-fields">
            <label>
              input top-k
              <input
                type="number"
                min={1}
                className="narrow"
                value={draftOptions.rerank?.text?.input_top_k ?? 300}
                onChange={(e) => updateRerankText({ input_top_k: Number(e.target.value) })}
              />
            </label>
            <label>
              output top-k
              <input
                type="number"
                min={1}
                className="narrow"
                value={draftOptions.rerank?.text?.output_top_k ?? 80}
                onChange={(e) => updateRerankText({ output_top_k: Number(e.target.value) })}
              />
            </label>
          </div>
        )}
        <label className="weight-row-label">
          <input
            type="checkbox"
            checked={draftOptions.rerank?.vlm?.enabled ?? capabilities.rerank.vlm}
            disabled={!capabilities.rerank.vlm}
            title={!capabilities.rerank.vlm ? "Chưa cấu hình VLM reranker (AIC_RERANK_VLM_URL)" : ""}
            onChange={(e) => updateRerankVlm({ enabled: e.target.checked })}
          />
          VLM rerank
        </label>
        {capabilities.rerank.vlm && (
          <div className="weight-inline-fields">
            <label>
              input top-k
              <input
                type="number"
                min={1}
                className="narrow"
                value={draftOptions.rerank?.vlm?.input_top_k ?? 20}
                onChange={(e) => updateRerankVlm({ input_top_k: Number(e.target.value) })}
              />
            </label>
          </div>
        )}
        {Object.keys(capabilities.unsupported_options).length > 0 && (
          <details className="unsupported-options">
            <summary>{Object.keys(capabilities.unsupported_options).length} option chưa chạy thật</summary>
            <ul>
              {Object.entries(capabilities.unsupported_options).map(([path, reason]) => (
                <li key={path}>
                  <code>{path}</code>: {reason}
                </li>
              ))}
            </ul>
          </details>
        )}
      </section>

      {task === "TRAKE" && (
        <section className="weight-group">
          <h4>TRAKE Alignment</h4>
          <label className="weight-field" title="Thưởng chuỗi đúng thứ tự (VideoRetrieverConfig.ordering_weight)">
            Order Consistency
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={draftOptions.temporal?.order_weight ?? 0.6}
              onChange={(e) => updateTemporal({ order_weight: Number(e.target.value) })}
            />
            <output>{(draftOptions.temporal?.order_weight ?? 0.6).toFixed(2)}</output>
          </label>
          <label className="weight-field" title="Phạt theo khoảng cách giây giữa hai step liên tiếp (SequenceConfig.gap_penalty_per_sec)">
            Gap Penalty
            <input
              type="range"
              min={0}
              max={0.1}
              step={0.002}
              value={draftOptions.temporal?.gap_penalty_per_sec ?? 0.002}
              onChange={(e) => updateTemporal({ gap_penalty_per_sec: Number(e.target.value) })}
            />
            <output>{(draftOptions.temporal?.gap_penalty_per_sec ?? 0.002).toFixed(3)}</output>
          </label>
          <label className="weight-field" title="Phạt khi một step không có candidate (SequenceConfig.missing_step_penalty)">
            Missing Step Penalty
            <input
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={draftOptions.temporal?.missing_step_penalty ?? 0.5}
              onChange={(e) => updateTemporal({ missing_step_penalty: Number(e.target.value) })}
            />
            <output>{(draftOptions.temporal?.missing_step_penalty ?? 0.5).toFixed(2)}</output>
          </label>
          <label className="weight-field disabled-field" title="Chưa có thuật toán làm mượt cục bộ trong SequenceConfig — không phải option chết ẩn, hiển thị rõ để không tạo cảm giác đã bật">
            Temporal Smoothness
            <input type="range" min={0} max={1} step={0.05} value={0} disabled />
            <output>chưa cài đặt</output>
          </label>
          <label className="weight-field" title="Khoảng cách tối đa giữa hai step (giây) — SequenceConfig.max_gap_sec">
            Max gap (sec)
            <input
              type="number"
              min={1}
              className="narrow"
              value={draftOptions.temporal?.maximum_gap_sec ?? 300}
              onChange={(e) => updateTemporal({ maximum_gap_sec: Number(e.target.value) })}
            />
          </label>
          <label className="weight-row-label">
            <input
              type="checkbox"
              checked={draftOptions.temporal?.allow_missing_optional_step ?? true}
              onChange={(e) => updateTemporal({ allow_missing_optional_step: e.target.checked })}
            />
            Cho phép bỏ qua step thiếu candidate
          </label>

          <div className="weight-tabs">
            <button type="button" className={stepTab === "global" ? "weight-tab active" : "weight-tab"} onClick={() => setStepTab("global")}>
              Global
            </button>
            <button
              type="button"
              className={stepTab === "per_step" ? "weight-tab active" : "weight-tab"}
              onClick={() => setStepTab("per_step")}
              disabled={parsedEvents.length < 2}
              title={parsedEvents.length < 2 ? "Query chưa parse được >= 2 step — chạy search với Debug bật để xem" : ""}
            >
              Per Step
            </button>
          </div>

          {stepTab === "per_step" && parsedEvents.length >= 2 && (
            <div className="per-step-table-wrap">
              <div className="per-step-actions">
                <button type="button" onClick={copyGlobalToAllSteps}>
                  Copy global → all steps
                </button>
              </div>
              <table className="per-step-table">
                <thead>
                  <tr>
                    <th>Branch</th>
                    {parsedEvents.map((event) => (
                      <th key={event.event_idx}>Step {event.event_idx + 1}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(MODALITY_LABELS)
                    .filter((key, index, arr) => arr.indexOf(key) === index)
                    .map((modality) => (
                      <tr key={modality}>
                        <td>{MODALITY_LABELS[modality]}</td>
                        {parsedEvents.map((event) => (
                          <td key={event.event_idx}>
                            <input
                              type="number"
                              min={0}
                              step={0.01}
                              className="narrow"
                              value={draftOptions.temporal?.step_modality_weights?.[event.event_idx]?.[modality] ?? ""}
                              placeholder="auto"
                              onChange={(e) => updateStepWeight(event.event_idx, modality, Number(e.target.value))}
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                </tbody>
              </table>
              <div className="per-step-actions">
                {parsedEvents.map((event) => (
                  <button key={event.event_idx} type="button" onClick={() => normalizeStep(event.event_idx)}>
                    Normalize step {event.event_idx + 1}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      <section className="weight-group">
        <h4>Presets</h4>
        <select
          onChange={(e) => {
            const preset = presets.find((item) => item.id === e.target.value);
            if (preset) applyPreset(preset);
          }}
          defaultValue=""
        >
          <option value="" disabled>
            Chọn preset…
          </option>
          {presets.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.name} {preset.source === "user" ? "(tự lưu)" : ""}
            </option>
          ))}
        </select>
        <div className="preset-save-row">
          <input value={presetName} onChange={(e) => setPresetName(e.target.value)} placeholder="Tên preset" />
          <button type="button" onClick={saveCurrentAsPreset} disabled={!presetName.trim()}>
            Lưu preset hiện tại
          </button>
        </div>
        {presets.some((p) => p.source === "user") && (
          <ul className="preset-user-list">
            {presets
              .filter((p) => p.source === "user")
              .map((p) => (
                <li key={p.id}>
                  {p.name}
                  <button type="button" onClick={() => removePreset(p.id)} title="Xoá preset">
                    ✕
                  </button>
                </li>
              ))}
          </ul>
        )}
      </section>
    </div>
  );
}
