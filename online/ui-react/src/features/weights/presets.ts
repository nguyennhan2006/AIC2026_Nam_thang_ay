// Preset lưu client-side (localStorage) — KHÔNG có endpoint GET /v1/search/
// presets ở backend (đã audit trước khi build UI), nên toàn bộ vòng đời
// preset (save/load/duplicate/rename/delete/export/import) chỉ chạy ở đây.

import type { SearchOptions, TaskType } from "../../types";

export interface SearchPreset {
  id: string;
  name: string;
  task: TaskType;
  version: string;
  searchOptions: SearchOptions;
  createdAt: string;
  source: "built_in" | "user";
}

const STORAGE_KEY = "aic_search_presets_v2";
const PRESET_VERSION = "1";

export const BUILT_IN_PRESETS: SearchPreset[] = [
  {
    id: "builtin_balanced",
    name: "Balanced",
    task: "TEXTUAL_KIS",
    version: PRESET_VERSION,
    searchOptions: {},
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
  {
    id: "builtin_visual_heavy",
    name: "Visual Heavy",
    task: "TEXTUAL_KIS",
    version: PRESET_VERSION,
    searchOptions: { branches: { dense_visual: { weight: 3 } } },
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
  {
    id: "builtin_ocr_heavy",
    name: "OCR Heavy",
    task: "TEXTUAL_KIS",
    version: PRESET_VERSION,
    searchOptions: { branches: { bm25_ocr: { weight: 3 } } },
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
  {
    id: "builtin_qa_evidence",
    name: "QA Evidence",
    task: "QA",
    version: PRESET_VERSION,
    searchOptions: { rerank: { text: { enabled: true } } },
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
  {
    id: "builtin_trake_balanced",
    name: "TRAKE Balanced",
    task: "TRAKE",
    version: PRESET_VERSION,
    searchOptions: {},
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
  {
    id: "builtin_trake_ocr_final",
    name: "TRAKE OCR Final Step",
    task: "TRAKE",
    version: PRESET_VERSION,
    searchOptions: { temporal: { step_modality_weights: [] } },
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
  {
    id: "builtin_avs_diversity",
    name: "AVS Diversity",
    task: "AVS",
    version: PRESET_VERSION,
    searchOptions: { fusion: { dedup_scope: "event" } },
    createdAt: "1970-01-01T00:00:00Z",
    source: "built_in",
  },
];

export function loadUserPresets(): SearchPreset[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveUserPresets(presets: SearchPreset[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

export function upsertUserPreset(preset: SearchPreset): SearchPreset[] {
  const existing = loadUserPresets();
  const withoutOld = existing.filter((item) => item.id !== preset.id);
  const next = [...withoutOld, preset];
  saveUserPresets(next);
  return next;
}

export function deleteUserPreset(id: string): SearchPreset[] {
  const next = loadUserPresets().filter((item) => item.id !== id);
  saveUserPresets(next);
  return next;
}

export function allPresets(): SearchPreset[] {
  return [...BUILT_IN_PRESETS, ...loadUserPresets()];
}

export function newPresetId(): string {
  return `user_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
