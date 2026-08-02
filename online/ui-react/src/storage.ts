// Port key localStorage từ bản UI trước (vanilla + React cũ) — đổi tên key ở
// đây sẽ làm mất cấu hình đã lưu của người dùng cũ.
// `selection`/`trayAnswer` của bản VQA-only cũ đã bỏ: submission giờ build từ
// chính response.kis/qa/trake (server-side, /v1/submissions/build — PR-08),
// không còn "khay chọn tay từng scene" phía client.
const KEYS = {
  apiBase: "aic_api_base",
  apiToken: "aic_api_token",
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
