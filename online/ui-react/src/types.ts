// Khớp 1:1 với online/domain/*.py sau PR-01..PR-09 — đổi backend thì đổi ở đây
// trước. UI không được tự suy đoán field nào backend có: mọi field optional
// ở đây phản ánh đúng field optional bên Pydantic.

export type TaskType = "TEXTUAL_KIS" | "QA" | "TRAKE" | "AVS";

export interface SearchFilters {
  video_ids?: string[];
  scene_ids?: string[];
  has_ocr?: boolean | null;
  has_asr?: boolean | null;
  start_sec_gte?: number | null;
  end_sec_lte?: number | null;
}

// ---- Search Mixing Console (SearchOptions) — PR-04 ------------------------

export interface BranchRuntimeOptions {
  enabled?: boolean;
  weight?: number;
  top_k?: number;
  min_score?: number | null;
  threshold_space?: "raw" | "normalized" | "percentile";
  threshold_policy?: "hard" | "soft";
  timeout_ms?: number;
}

export interface FusionOptions {
  method?: "rrf" | "weighted_sum" | "max_score" | "intersection" | "union";
  rrf_k?: number;
  minimum_matching_branches?: number;
  dedup_scope?: "none" | "frame" | "scene" | "event";
  max_results_per_video?: number | null;
}

export interface RerankStageOptions {
  enabled?: boolean;
}

export interface RerankOptions {
  enable_rules?: boolean;
  text?: RerankStageOptions;
  vlm?: RerankStageOptions;
}

export interface ResultOptions {
  display_top_k?: number;
  sort_by?: "final_score" | "visual_score" | "caption_score" | "ocr_score" | "asr_score" | "time";
}

export interface SearchOptions {
  branches?: Record<string, BranchRuntimeOptions>;
  fusion?: FusionOptions;
  rerank?: RerankOptions;
  results?: ResultOptions;
}

// ---- Capabilities (GET /v1/search/capabilities) — PR-03/04 ----------------

export interface BranchCapabilities {
  branch_id: string;
  execution_ids: string[];
  modality: string | null;
  backend_kind: string;
  available: boolean;
  degraded: boolean;
  degraded_reason: string | null;
  model_id: string | null;
  index_id: string | null;
  supported_controls: string[];
}

export interface CapabilitiesResponse {
  task_types: string[];
  branches: BranchCapabilities[];
  fusion_methods: string[];
  unsupported_options: Record<string, string>;
  rerank: { rules: boolean; text: boolean; vlm: boolean };
  events_available: boolean;
}

// ---- Candidate / SearchHit (frame contract) — PR-01 ------------------------

export interface FrameEvidence {
  keyframe_id: string;
  video_id: string;
  scene_id: string;
  frame_idx: number;
  timestamp_sec: number;
  image_path: string;
  captions: string[];
  ocr_texts: string[];
  object_labels: string[];
  action_tags: string[];
  dominant_colors: string[];
}

export interface Evidence {
  modality: string;
  text: string;
  score: number;
}

export interface SearchHit {
  rank: number;
  candidate_id: string;
  scene_id: string;
  video_id: string;
  video_path: string | null;
  event_id: string | null;
  scene_idx: number;
  start_frame: number;
  end_frame_exclusive: number;
  start_sec: number;
  end_sec: number;
  best_frame_idx: number;
  best_keyframe_id: string | null;
  best_keyframe_path: string | null;
  best_timestamp_sec: number | null;
  safe_frame_score: number | null;
  score: number;
  keyframes: FrameEvidence[];
  matched_modalities: string[];
  matched_branches: string[];
  evidence: Evidence[];
  component_scores: Record<string, number>;
  branch_contributions: Record<string, number>;
  warnings: string[];
}

export interface SequenceHit {
  video_id: string;
  score: number;
  scenes: SearchHit[];
  frame_ids: number[];
}

// ---- Task-specific result items — PR-07 ------------------------------------

export interface KisResultItem {
  rank: number;
  video_id: string;
  frame_idx: number;
  scene_id: string | null;
  event_id: string | null;
  score: number;
  safe_frame_score: number | null;
  must_match_coverage: number | null;
}

export interface QaResultItem {
  rank: number;
  video_id: string;
  frame_idx: number;
  answer: string;
  canonical_answer: string;
  answer_type: string;
  joint_score: number;
  verifier_status: "SUPPORTED" | "PARTIAL" | "CONTRADICTED" | "INSUFFICIENT";
  scene_id: string | null;
  evidence_ids: string[];
}

export interface TrakeStep {
  step: number;
  frame_idx: number;
  scene_id: string | null;
  confidence: number;
  refinement: "keyframe_only" | "dense_window";
}

export interface TrakeResultItem {
  rank: number;
  video_id: string;
  frame_ids: number[];
  sequence_score: number;
  steps: TrakeStep[];
  step_coverage: number;
  ordering_score: number;
}

export interface AvsResultItem {
  rank: number;
  video_id: string;
  segment_id: string;
  start_frame: number;
  end_frame: number;
  relevance_grade: number;
  score: number;
  cluster_id: string | null;
  best_frame_idx: number | null;
}

// ---- Branch execution status — PR-03 --------------------------------------

export type BranchState = "success" | "disabled" | "unavailable" | "timeout" | "failed" | "empty";

export interface BranchStatus {
  execution_id: string;
  branch_id: string;
  state: BranchState;
  latency_ms: number;
  candidate_count: number;
  warning: string | null;
}

// ---- QueryPlan / SearchRequest / SearchResponse ----------------------------

export interface QueryPlan {
  task: TaskType;
  original_query: string;
  normalized_query: string;
  events: { event_idx: number; text: string; exact_phrases: string[] }[];
  modality_weights: Record<string, number>;
  filters: SearchFilters;
}

export interface SearchRequestBody {
  query: string;
  task?: TaskType;
  top_k?: number;
  filters?: SearchFilters;
  debug?: boolean;
  search_options?: SearchOptions;
}

export type PipelineStatus = "COMPLETED" | "COMPLETED_WITH_WARNINGS";

export interface SearchResponse {
  query_id: string;
  task: TaskType;
  took_ms: number;
  status: PipelineStatus;
  results: SearchHit[];
  sequences: SequenceHit[];
  kis: KisResultItem[];
  qa: QaResultItem[];
  trake: TrakeResultItem[];
  avs: AvsResultItem[];
  branch_status: BranchStatus[];
  query_plan: QueryPlan | null;
  warnings: string[];
  replayed_from: string | null;
}

// ---- Evidence pack (GET /v1/evidence/{id}) — PR-06 -------------------------

export interface NeighborContext {
  scene_id: string;
  start_frame: number;
  end_frame_exclusive: number;
  start_sec: number;
  end_sec: number;
  caption: string | null;
  ocr_text: string | null;
}

export interface EvidencePack {
  candidate_id: string;
  video_id: string;
  scene_id: string | null;
  event_id: string | null;
  start_frame: number;
  end_frame_exclusive: number;
  start_sec: number;
  end_sec: number;
  keyframes: FrameEvidence[];
  best_frame_idx: number | null;
  asr_window: string | null;
  caption_text: string | null;
  ocr_text: string | null;
  previous_context: NeighborContext | null;
  next_context: NeighborContext | null;
  branch_contributions: Record<string, number>;
  rule_adjustments: { rule: string; delta: number; detail: string | null }[];
  model_versions: Record<string, string>;
  dataset_version: string | null;
}

// ---- Search session trace (PR-09) ------------------------------------------

export interface SearchExecutionTrace {
  session_id: string;
  task: TaskType;
  raw_request: SearchRequestBody;
  branch_status: BranchStatus[];
  status: PipelineStatus;
  warnings: string[];
  took_ms: number;
  dataset_version: string | null;
  model_versions: Record<string, string>;
  created_at: string;
  replay_count: number;
}

// ---- SSE stream events (PR-09) ---------------------------------------------

export type StreamEvent =
  | { type: "search_started"; query_id: string; task: string }
  | { type: "query_prepared"; query_id: string; normalized_query: string; modality_weights: Record<string, number> }
  | { type: "branch_started"; query_id: string; branch_id: string; execution_id: string }
  | {
      type: "branch_completed" | "branch_failed";
      query_id: string;
      branch_id: string;
      execution_id: string;
      state: BranchState;
      latency_ms: number;
      candidate_count: number;
      warning: string | null;
    }
  | { type: "fusion_completed"; query_id: string; candidate_count: number }
  | { type: "rerank_completed"; query_id: string; stages: { stage: string; applied: boolean; warning: string | null }[] }
  | { type: "evidence_ready"; query_id: string; count: number }
  | { type: "alignment_completed"; query_id: string; sequence_count: number; note?: string }
  | { type: "search_completed"; query_id: string; response: SearchResponse }
  | { type: "error"; message: string };

// ---- Submission (PR-08) ----------------------------------------------------

export interface SubmissionIssue {
  severity: "error" | "warning";
  code: string;
  message: string;
  row_index: number | null;
}

export interface SubmissionBuildResponse {
  task: TaskType;
  item_count: number;
  csv: string;
  has_errors: boolean;
  issues: SubmissionIssue[];
}

// ---- Misc -------------------------------------------------------------------

export interface SceneDocument {
  scene_id: string;
  video_id: string;
  video_path: string | null;
  scene_idx: number;
  start_frame: number;
  end_frame_exclusive: number;
  start_sec: number;
  end_sec: number;
  keyframes: FrameEvidence[];
  object_labels: string[];
  captions: string[];
  ocr_texts: string[];
  asr_texts: string[];
  keywords: string[];
}

export interface HealthResponse {
  status: string;
  backend: string;
  scene_count: number;
  dataset: string;
}

/** Vùng ranking chiến thuật (docs 01082026 §17 / online/competition/rules.py). */
export const RANKING_ZONES = [
  { name: "rank_1", low: 1, high: 1 },
  { name: "ranks_2_5", low: 2, high: 5 },
  { name: "ranks_6_20", low: 6, high: 20 },
  { name: "ranks_21_50", low: 21, high: 50 },
  { name: "ranks_51_100", low: 51, high: 100 },
] as const;

export function zoneForRank(rank: number): string {
  const match = RANKING_ZONES.find((zone) => rank >= zone.low && rank <= zone.high);
  return match?.name ?? "beyond_100";
}
