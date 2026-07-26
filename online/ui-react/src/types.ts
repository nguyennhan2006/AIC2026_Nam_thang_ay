// Khớp 1:1 với online/domain/models.py — đổi backend thì đổi ở đây trước.

export type Task = "kis" | "avs" | "sequence" | "vqa";

export interface SearchFilters {
  video_ids?: string[];
  scene_ids?: string[];
  has_ocr?: boolean | null;
  has_asr?: boolean | null;
  start_sec_gte?: number | null;
  end_sec_lte?: number | null;
}

export interface Evidence {
  modality: string;
  text: string;
  score: number;
}

export interface SearchHit {
  scene_id: string;
  video_id: string;
  video_path: string | null;
  scene_idx: number;
  start_sec: number;
  end_sec: number;
  score: number;
  keyframe_ids: string[];
  keyframe_paths: string[];
  keyframe_timestamps: number[];
  best_keyframe_id: string | null;
  best_keyframe_path: string | null;
  best_timestamp_sec: number | null;
  matched_modalities: string[];
  evidence: Evidence[];
  component_scores: Record<string, number>;
}

export interface SequenceHit {
  video_id: string;
  score: number;
  scenes: SearchHit[];
}

export interface QueryPlan {
  task: Task;
  original_query: string;
  normalized_query: string;
  events: { event_idx: number; text: string; exact_phrases: string[] }[];
  modality_weights: Record<string, number>;
  filters: SearchFilters;
}

export interface SearchResponse {
  query_id: string;
  task: Task;
  took_ms: number;
  results: SearchHit[];
  sequences: SequenceHit[];
  query_plan: QueryPlan | null;
}

export interface VQAResponse {
  query_id: string;
  answer: string;
  confidence: number | null;
  evidence: SearchHit[];
  took_ms: number;
}

export interface SceneDocument {
  scene_id: string;
  video_id: string;
  video_path: string | null;
  scene_idx: number;
  start_sec: number;
  end_sec: number;
  keyframe_ids: string[];
  keyframe_paths: string[];
  keyframe_timestamps: number[];
  object_labels: string[];
  keyframe_evidence: Record<string, unknown>[];
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

// Item lưu trong khay chọn — subset của SearchHit đủ dùng cho export CSV + hiển thị.
export interface TrayItem {
  scene_id: string;
  video_id: string;
  score: number;
  best_keyframe_id: string | null;
  best_timestamp_sec: number | null;
  start_sec: number;
}
