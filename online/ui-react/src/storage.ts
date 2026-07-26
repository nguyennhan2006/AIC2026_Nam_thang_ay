// Port 1:1 các key localStorage từ online/ui/app.js — đổi tên key ở đây sẽ làm mất
// cấu hình/khay chọn của người đã dùng bản UI cũ (app.js dùng đúng các key này).
import type { TrayItem } from "./types";

const KEYS = {
  apiBase: "aic_api_base",
  apiToken: "aic_api_token",
  trayAnswer: "aic_tray_answer",
  selection: "aic_selection",
} as const;

export function loadApiBase(): string {
  return localStorage.getItem(KEYS.apiBase) || "http://localhost:8000";
}
export function saveApiBase(value: string): void {
  localStorage.setItem(KEYS.apiBase, value);
}

export function loadApiToken(): string {
  return localStorage.getItem(KEYS.apiToken) || "";
}
export function saveApiToken(value: string): void {
  localStorage.setItem(KEYS.apiToken, value);
}

export function loadTrayAnswer(): string {
  return localStorage.getItem(KEYS.trayAnswer) || "";
}
export function saveTrayAnswer(value: string): void {
  localStorage.setItem(KEYS.trayAnswer, value);
}

/** app.js lưu selection dạng `[...Map.entries()]` (mảng [scene_id, item][]) — giữ
 * đúng định dạng đó để 2 bản UI đọc/ghi lẫn nhau được (dùng chung 1 trình duyệt). */
export function loadSelection(): Map<string, TrayItem> {
  try {
    const raw = JSON.parse(localStorage.getItem(KEYS.selection) || "[]") as [string, TrayItem][];
    return new Map(raw);
  } catch {
    return new Map();
  }
}
export function saveSelection(selection: Map<string, TrayItem>): void {
  localStorage.setItem(KEYS.selection, JSON.stringify([...selection.entries()]));
}
