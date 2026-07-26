import type { HealthResponse, SceneDocument, SearchFilters, SearchResponse, Task, VQAResponse } from "./types";

export class ApiError extends Error {}

export interface ApiClientConfig {
  base: string;
  token: string;
}

function headers(config: ApiClientConfig): HeadersInit {
  const value: Record<string, string> = { "Content-Type": "application/json" };
  if (config.token) value.Authorization = `Bearer ${config.token}`;
  return value;
}

function normalizedBase(config: ApiClientConfig): string {
  return config.base.trim().replace(/\/$/, "");
}

export function mediaUrl(config: ApiClientConfig, path: string): string {
  const encoded = path.split("/").map(encodeURIComponent).join("/");
  return `${normalizedBase(config)}/v1/media/${encoded}`;
}

async function parseOrThrow<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data?.detail || data?.error?.message || "Request failed";
    throw new ApiError(message);
  }
  return data as T;
}

export async function search(
  config: ApiClientConfig,
  task: Exclude<Task, "vqa">,
  body: { query: string; top_k: number; debug: boolean; filters?: SearchFilters }
): Promise<SearchResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/search/${task}`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
  });
  return parseOrThrow<SearchResponse>(response);
}

export async function vqa(
  config: ApiClientConfig,
  body: { question: string; top_k_evidence: number; debug: boolean; filters?: SearchFilters }
): Promise<VQAResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/vqa`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
  });
  return parseOrThrow<VQAResponse>(response);
}

export async function getSceneDetail(config: ApiClientConfig, sceneId: string): Promise<SceneDocument> {
  const response = await fetch(`${normalizedBase(config)}/v1/scenes/${encodeURIComponent(sceneId)}`, {
    headers: headers(config),
  });
  return parseOrThrow<SceneDocument>(response);
}

export async function health(config: ApiClientConfig): Promise<HealthResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/health`);
  return parseOrThrow<HealthResponse>(response);
}
