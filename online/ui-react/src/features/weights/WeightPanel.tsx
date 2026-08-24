import { useEffect, useMemo, useState } from "react";
import { Headphones, RotateCcw, Save, Scale, Trash2 } from "lucide-react";
import type { ApiClientConfig } from "../../api";
import { ApiError, getCapabilities } from "../../api";
import type { CapabilitiesResponse, QueryPlan, SearchOptions, TaskType } from "../../types";
import {
  applySolo, isSoloActive, normalizeStepColumn, normalizeWeights, runningBranches, toggleSolo,
} from "./weightMath";
import type { WeightControlValue } from "./weightMath";
import { allPresets, deleteUserPreset, newPresetId, upsertUserPreset } from "./presets";
import type { SearchPreset } from "./presets";
import {
  Badge,
  Button,
  Checkbox,
  IconButton,
  InlineError,
  NumericInput,
  PanelBody,
  PanelHeader,
  SegmentedControl,
  SelectField,
  Section,
  SkeletonRows,
  Surface,
  WeightRow,
} from "../../ui";

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
  object: "Objects",
  action: "Actions",
  color: "Color",
  event: "Event meta",
};

/** Màu track riêng cho từng modality — cùng bảng màu với badge điểm ở card
 * kết quả, để mắt nối được "trọng số này" với "đóng góp kia". */
const MODALITY_TONES: Record<string, string> = {
  visual: "var(--accent)",
  caption: "var(--accent-cyan)",
  ocr: "var(--accent-amber)",
  asr: "var(--accent-pink)",
  keyword: "var(--accent-purple)",
  object: "var(--accent-green)",
  action: "var(--accent-green)",
  color: "var(--accent-cyan)",
  event: "var(--accent-purple)",
};

function branchWeight(options: SearchOptions, branchId: string): number {
  return options.branches?.[branchId]?.weight ?? 1;
}
function branchEnabled(options: SearchOptions, branchId: string): boolean {
  return options.branches?.[branchId]?.enabled ?? true;
}

export function WeightPanel({ apiConfig, task, draftOptions, onDraftChange, hasUnsavedChanges, parsedEvents }: WeightPanelProps) {
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stepMode, setStepMode] = useState<"global" | "per_step">("global");
  const [presetName, setPresetName] = useState("");
  const [appliedPresetId, setAppliedPresetId] = useState("");
  const [presets, setPresets] = useState<SearchPreset[]>(() => allPresets());
  const [normalizeError, setNormalizeError] = useState<string | null>(null);
  const [locked, setLocked] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    let cancelled = false;
    setError(null);
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

  const modalityBranches = useMemo(
    () => (capabilities?.branches ?? []).filter((branch) => branch.modality),
    [capabilities]
  );

  // Solo chỉ thao tác trên nhánh CÓ THẬT và đang khả dụng: gán
  // `enabled: false` cho một nhánh server không đăng ký thì
  // `/v1/search/capabilities` trả 422 và mọi truy vấn hỏng.
  const soloableIds = useMemo(
    () => modalityBranches.filter((branch) => branch.available).map((branch) => branch.branch_id),
    [modalityBranches]
  );
  const soloActive = isSoloActive(draftOptions, soloableIds);
  const running = runningBranches(draftOptions, soloableIds);
  const labelFor = (branchId: string) => {
    const branch = modalityBranches.find((item) => item.branch_id === branchId);
    return MODALITY_LABELS[branch?.modality ?? ""] ?? branchId;
  };

  function updateBranch(branchId: string, patch: { enabled?: boolean; weight?: number }) {
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
      const branches = { ...(draftOptions.branches ?? {}) };
      for (const item of normalizeWeights(values)) {
        branches[item.branchId] = { ...branches[item.branchId], weight: item.weight };
      }
      onDraftChange({ ...draftOptions, branches });
    } catch (err) {
      setNormalizeError(err instanceof Error ? err.message : String(err));
    }
  }

  const updateFusion = (patch: Partial<NonNullable<SearchOptions["fusion"]>>) =>
    onDraftChange({ ...draftOptions, fusion: { ...draftOptions.fusion, ...patch } });
  const updateRerankText = (patch: Partial<NonNullable<NonNullable<SearchOptions["rerank"]>["text"]>>) =>
    onDraftChange({ ...draftOptions, rerank: { ...draftOptions.rerank, text: { ...draftOptions.rerank?.text, ...patch } } });
  const updateRerankVlm = (patch: Partial<NonNullable<NonNullable<SearchOptions["rerank"]>["vlm"]>>) =>
    onDraftChange({ ...draftOptions, rerank: { ...draftOptions.rerank, vlm: { ...draftOptions.rerank?.vlm, ...patch } } });
  const updateTemporal = (patch: Partial<NonNullable<SearchOptions["temporal"]>>) =>
    onDraftChange({ ...draftOptions, temporal: { ...draftOptions.temporal, ...patch } });

  function updateStepWeight(stepIndex: number, modality: string, value: number | null) {
    const current = draftOptions.temporal?.step_modality_weights ?? [];
    const next = [...current];
    while (next.length <= stepIndex) next.push({});
    next[stepIndex] = { ...next[stepIndex], [modality]: value ?? 0 };
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
    const globalRow: Record<string, number> = {};
    for (const branch of modalityBranches) {
      if (branch.modality) globalRow[branch.modality] = branchWeight(draftOptions, branch.branch_id);
    }
    updateTemporal({ step_modality_weights: parsedEvents.map(() => ({ ...globalRow })) });
  }

  function saveCurrentAsPreset() {
    if (!presetName.trim()) return;
    upsertUserPreset({
      id: newPresetId(),
      name: presetName.trim(),
      task,
      version: "1",
      searchOptions: draftOptions,
      createdAt: new Date().toISOString(),
      source: "user",
    });
    setPresets(allPresets());
    setPresetName("");
  }

  const header = (
    <PanelHeader
      title="Trọng số"
      icon={<Scale size={14} />}
      meta={hasUnsavedChanges ? undefined : `${modalityBranches.length} nhánh`}
      actions={
        <>
          {hasUnsavedChanges && <Badge tone="warning">chưa áp dụng</Badge>}
          <IconButton
            icon={<RotateCcw size={13} />}
            label="Đặt lại toàn bộ trọng số về mặc định"
            size="sm"
            onClick={() => onDraftChange({})}
          />
        </>
      }
    />
  );

  if (error) {
    return (
      <Surface fill className="weight-panel">
        {header}
        <PanelBody>
          <InlineError message={`Không tải được capabilities: ${error}`} />
        </PanelBody>
      </Surface>
    );
  }

  if (!capabilities) {
    return (
      <Surface fill className="weight-panel">
        {header}
        <PanelBody>
          <SkeletonRows rows={6} />
        </PanelBody>
      </Surface>
    );
  }

  const unsupportedCount = Object.keys(capabilities.unsupported_options).length;

  return (
    <Surface fill className="weight-panel">
      {header}
      <PanelBody>
        <Section
          title="Modalities"
          actions={
            <Button size="sm" variant="ghost" onClick={handleNormalize}>
              Normalize
            </Button>
          }
        >
          {soloActive && (
            <div className="solo-banner">
              <Headphones size={12} />
              <span>Chỉ chạy:</span>
              {running.map((branchId) => (
                <span key={branchId} className="solo-banner-branch">{labelFor(branchId)}</span>
              ))}
              <Button
                size="sm" variant="ghost"
                onClick={() => onDraftChange(applySolo(draftOptions, soloableIds, []))}
              >
                Bật lại tất cả
              </Button>
            </div>
          )}
          {modalityBranches.map((branch) => (
            <WeightRow
              key={branch.branch_id}
              label={MODALITY_LABELS[branch.modality ?? ""] ?? branch.branch_id}
              value={branchWeight(draftOptions, branch.branch_id)}
              onValueChange={(value) => updateBranch(branch.branch_id, { weight: value })}
              enabled={branchEnabled(draftOptions, branch.branch_id)}
              onEnabledChange={(enabled) => updateBranch(branch.branch_id, { enabled })}
              locked={locked.has(branch.branch_id)}
              onLockToggle={() => toggleLock(branch.branch_id)}
              soloed={soloActive && running.includes(branch.branch_id)}
              onSoloToggle={
                branch.available
                  ? () => onDraftChange(toggleSolo(draftOptions, soloableIds, branch.branch_id))
                  : undefined
              }
              disabled={!branch.available}
              badge={branch.degraded ? "degraded" : undefined}
              title={branch.degraded ? branch.degraded_reason ?? undefined : branch.branch_id}
              tone={MODALITY_TONES[branch.modality ?? ""]}
            />
          ))}
          {normalizeError && <InlineError message={normalizeError} />}
        </Section>

        <Section title="Fusion & Ranking">
          <SelectField
            label="Method"
            value={draftOptions.fusion?.method ?? "rrf"}
            onChange={(value) => updateFusion({ method: value as NonNullable<SearchOptions["fusion"]>["method"] })}
          >
            {capabilities.fusion_methods.map((method) => (
              <option key={method} value={method}>
                {method}
              </option>
            ))}
          </SelectField>

          <div className="field field-inline">
            <label className="field-label" htmlFor="rrf-k">
              RRF k
            </label>
            <NumericInput
              value={draftOptions.fusion?.rrf_k ?? 60}
              min={1}
              max={500}
              ariaLabel="RRF k"
              width="64px"
              onChange={(value) => updateFusion({ rrf_k: value ?? 60 })}
            />
          </div>

          <SelectField
            label="Dedup scope"
            value={draftOptions.fusion?.dedup_scope ?? "scene"}
            onChange={(value) => updateFusion({ dedup_scope: value as NonNullable<SearchOptions["fusion"]>["dedup_scope"] })}
          >
            <option value="none">none</option>
            <option value="frame">frame</option>
            <option value="scene">scene</option>
            <option value="event">event</option>
          </SelectField>

          <div
            className="field field-inline"
            title="KIS mặc định chỉ giữ 5 kết quả/video (dedup chống một video chiếm hết top-K). Dataset 1 video → giới hạn này khiến Top-K > 5 không có tác dụng. Để trống = mặc định của task."
          >
            <label className="field-label">Max / video</label>
            <NumericInput
              value={draftOptions.fusion?.max_results_per_video ?? ""}
              min={1}
              ariaLabel="Số kết quả tối đa mỗi video"
              width="64px"
              placeholder="auto"
              onChange={(value) => updateFusion({ max_results_per_video: value })}
            />
          </div>
        </Section>

        <Section title="Rerank">
          <Checkbox
            checked={draftOptions.rerank?.text?.enabled ?? capabilities.rerank.text}
            disabled={!capabilities.rerank.text}
            title={capabilities.rerank.text ? undefined : "Server chưa cấu hình text reranker"}
            onChange={(checked) => updateRerankText({ enabled: checked })}
            label="Text rerank (cross-encoder)"
          />
          {capabilities.rerank.text && (
            <div className="field field-inline">
              <label className="field-label">Input → output top-k</label>
              <span className="field-pair">
                <NumericInput
                  value={draftOptions.rerank?.text?.input_top_k ?? 300}
                  min={1}
                  ariaLabel="Text rerank input top-k"
                  width="56px"
                  onChange={(value) => updateRerankText({ input_top_k: value ?? 300 })}
                />
                <NumericInput
                  value={draftOptions.rerank?.text?.output_top_k ?? 80}
                  min={1}
                  ariaLabel="Text rerank output top-k"
                  width="56px"
                  onChange={(value) => updateRerankText({ output_top_k: value ?? 80 })}
                />
              </span>
            </div>
          )}

          <Checkbox
            checked={draftOptions.rerank?.vlm?.enabled ?? capabilities.rerank.vlm}
            disabled={!capabilities.rerank.vlm}
            title={capabilities.rerank.vlm ? undefined : "Server chưa cấu hình VLM reranker"}
            onChange={(checked) => updateRerankVlm({ enabled: checked })}
            label="VLM rerank"
          />
          {capabilities.rerank.vlm && (
            <div className="field field-inline">
              <label className="field-label">Input top-k</label>
              <NumericInput
                value={draftOptions.rerank?.vlm?.input_top_k ?? 20}
                min={1}
                ariaLabel="VLM rerank input top-k"
                width="56px"
                onChange={(value) => updateRerankVlm({ input_top_k: value ?? 20 })}
              />
            </div>
          )}

          {unsupportedCount > 0 && (
            <details className="disclosure">
              <summary>{unsupportedCount} option server chưa chạy thật</summary>
              <ul className="disclosure-list">
                {Object.entries(capabilities.unsupported_options).map(([path, reason]) => (
                  <li key={path}>
                    <code>{path}</code>
                    <span>{reason}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </Section>

        {task === "TRAKE" && (
          <Section
            title="TRAKE alignment"
            actions={
              <SegmentedControl
                size="sm"
                ariaLabel="Phạm vi trọng số TRAKE"
                value={stepMode}
                onChange={setStepMode}
                options={[
                  { value: "global", label: "Global" },
                  {
                    value: "per_step",
                    label: "Per step",
                    disabled: parsedEvents.length < 2,
                    title: parsedEvents.length < 2 ? "Cần query parse được ≥ 2 bước (bật Debug rồi search)" : undefined,
                  },
                ]}
              />
            }
          >
            {stepMode === "global" ? (
              <>
                <WeightRow
                  label="Order"
                  value={draftOptions.temporal?.order_weight ?? 0.6}
                  min={0}
                  max={2}
                  step={0.05}
                  enabled
                  onValueChange={(value) => updateTemporal({ order_weight: value })}
                  title="Thưởng chuỗi đúng thứ tự (VideoRetrieverConfig.ordering_weight)"
                />
                <WeightRow
                  label="Gap penalty"
                  value={draftOptions.temporal?.gap_penalty_per_sec ?? 0.002}
                  min={0}
                  max={0.1}
                  step={0.002}
                  enabled
                  onValueChange={(value) => updateTemporal({ gap_penalty_per_sec: value })}
                  title="Phạt theo khoảng cách giây giữa hai bước (SequenceConfig.gap_penalty_per_sec)"
                  tone="var(--accent-amber)"
                />
                <WeightRow
                  label="Missing step"
                  value={draftOptions.temporal?.missing_step_penalty ?? 0.5}
                  min={0}
                  max={2}
                  step={0.05}
                  enabled
                  onValueChange={(value) => updateTemporal({ missing_step_penalty: value })}
                  title="Phạt khi một bước không có candidate (SequenceConfig.missing_step_penalty)"
                  tone="var(--accent-pink)"
                />
                <div className="field field-inline">
                  <label className="field-label">Max gap (giây)</label>
                  <NumericInput
                    value={draftOptions.temporal?.maximum_gap_sec ?? 300}
                    min={1}
                    ariaLabel="Khoảng cách tối đa giữa hai bước"
                    width="64px"
                    onChange={(value) => updateTemporal({ maximum_gap_sec: value ?? 300 })}
                  />
                </div>
                <Checkbox
                  checked={draftOptions.temporal?.allow_missing_optional_step ?? true}
                  onChange={(checked) => updateTemporal({ allow_missing_optional_step: checked })}
                  label="Cho phép bỏ qua bước thiếu candidate"
                />
              </>
            ) : (
              <div className="per-step">
                <Button size="sm" variant="ghost" block onClick={copyGlobalToAllSteps}>
                  Copy global → tất cả bước
                </Button>
                <div className="per-step-scroll scroll-x">
                  <table className="per-step-table">
                    <thead>
                      <tr>
                        <th scope="col">Branch</th>
                        {parsedEvents.map((event) => (
                          <th key={event.event_idx} scope="col">
                            #{event.event_idx + 1}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {modalityBranches.map((branch) => (
                        <tr key={branch.branch_id}>
                          <th scope="row" className="truncate">
                            {MODALITY_LABELS[branch.modality ?? ""] ?? branch.branch_id}
                          </th>
                          {parsedEvents.map((event) => (
                            <td key={event.event_idx}>
                              <NumericInput
                                value={draftOptions.temporal?.step_modality_weights?.[event.event_idx]?.[branch.modality ?? ""] ?? ""}
                                min={0}
                                step={0.01}
                                placeholder="auto"
                                width="100%"
                                ariaLabel={`Trọng số ${branch.modality} bước ${event.event_idx + 1}`}
                                onChange={(value) => updateStepWeight(event.event_idx, branch.modality ?? "", value)}
                              />
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="per-step-actions">
                  {parsedEvents.map((event) => (
                    <Button key={event.event_idx} size="sm" variant="ghost" onClick={() => normalizeStep(event.event_idx)}>
                      Norm #{event.event_idx + 1}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </Section>
        )}

        <Section title="Cách chỉnh cho ra điểm cao">
          {/* Ba điều này rút từ chính số đo của hệ, không phải lời khuyên chung
              chung. Đặt TRƯỚC bảng trọng số vì đó là thứ quyết định người dùng
              có cần động vào thanh trượt hay không. */}
          <ol className="tuning-guide">
            <li>
              <strong>Tìm bằng bản Mặc định trước.</strong> Nó là cấu hình đã đo:
              KIS <span className="tabular">R@20 = 1.000</span>, tức đáp án gần như
              luôn nằm đâu đó trong 20 dòng đầu. Hãy xem hết 20 dòng trước khi kết
              luận là không tìm thấy.
            </li>
            <li>
              <strong>Không thấy thì đổi theo LOẠI MANH MỐI</strong>, đừng kéo bừa:
              chữ trên màn hình → bản “Tìm chữ”; lời người ta nói → bản “ASR”;
              mô tả cảnh → bản “Hình ảnh”.
            </li>
            <li>
              <strong>Đổi một thứ mỗi lần.</strong> Kéo hai thanh cùng lúc thì
              không biết cái nào giúp. Bấm <em>Đặt lại</em> trước khi thử hướng khác.
            </li>
          </ol>
          <p className="tuning-note">
            Trọng số <span className="tabular">0.25</span> gần như không đổi được thứ
            hạng — muốn một nhánh thật sự thắng thì cần <span className="tabular">1.0</span>
            trở lên. Đo được: truy vấn tra tên ở 0.25 không vào nổi top-20, ở 1.0 lên hạng 1.
          </p>
        </Section>

        <Section title="Bản chấm dựng sẵn">
          {/* Người chưa quen chỉnh thông số cần biết ba điều trước khi động vào
              thanh trượt: nên bắt đầu ở đâu, khi nào cần đổi, và đổi rồi thì
              tin được tới mức nào. Ba điều đó hiện ngay dưới ô chọn. */}
          <p className="preset-intro">
            Chưa biết chỉnh gì thì cứ để <strong>Mặc định</strong> rồi tìm một lượt.
            Chỉ đổi khi đã xem hết 20 kết quả mà vẫn chưa thấy đáp án — và đổi theo
            <em> loại manh mối bạn đang có</em>, không đổi bừa.
          </p>

          <SelectField
            label="Áp dụng"
            value={appliedPresetId}
            onChange={(value) => {
              setAppliedPresetId(value);
              const preset = presets.find((item) => item.id === value);
              if (preset) onDraftChange(preset.searchOptions);
            }}
          >
            <option value="">Chọn bản chấm…</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
                {preset.source === "user" ? " ·" : ""}
              </option>
            ))}
          </SelectField>

          {(() => {
            const active = presets.find((item) => item.id === appliedPresetId);
            if (!active) return null;
            return (
              <div className="preset-guide">
                {active.description && <p>{active.description}</p>}
                {active.whenToUse && (
                  <p><strong>Khi nào dùng:</strong> {active.whenToUse}</p>
                )}
                {active.evidence ? (
                  <p className="preset-evidence"><strong>Số đo:</strong> {active.evidence}</p>
                ) : (
                  <p className="preset-evidence">
                    <strong>Chưa đo</strong> — bản chấm này chưa có số liệu ủng hộ.
                  </p>
                )}
              </div>
            );
          })()}

          <div className="preset-save">
            <input
              className="text-input"
              value={presetName}
              placeholder="Tên preset mới"
              aria-label="Tên preset mới"
              onChange={(event) => setPresetName(event.target.value)}
            />
            <Button size="sm" icon={<Save size={13} />} disabled={!presetName.trim()} onClick={saveCurrentAsPreset}>
              Lưu
            </Button>
          </div>

          {presets.some((preset) => preset.source === "user") && (
            <ul className="preset-list">
              {presets
                .filter((preset) => preset.source === "user")
                .map((preset) => (
                  <li key={preset.id}>
                    <span className="truncate">{preset.name}</span>
                    <IconButton
                      icon={<Trash2 size={12} />}
                      label={`Xoá preset ${preset.name}`}
                      size="sm"
                      onClick={() => {
                        deleteUserPreset(preset.id);
                        setPresets(allPresets());
                      }}
                    />
                  </li>
                ))}
            </ul>
          )}
        </Section>
      </PanelBody>
    </Surface>
  );
}
