import type {
  CapabilitiesResponse,
  EvidencePack,
  HealthResponse,
  SceneDocument,
  SearchExecutionTrace,
  SearchRequestBody,
  SearchResponse,
  StreamEvent,
  SubmissionBuildResponse,
  VideoFrame,
  VideoMeta,
  SubmissionIssue,
  TaskType,
} from "./types";

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
    const message = data?.detail || data?.error?.message || `Request failed (${response.status})`;
    throw new ApiError(message);
  }
  return data as T;
}

// ---- Search (convenience + unified) — PR-01/09 -----------------------------

const TASK_PATH: Record<TaskType, string> = {
  TEXTUAL_KIS: "kis",
  QA: "qa",
  TRAKE: "trake",
  AVS: "avs",
};

export async function search(
  config: ApiClientConfig,
  task: TaskType,
  body: Omit<SearchRequestBody, "task">
): Promise<SearchResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/search/${TASK_PATH[task]}`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
  });
  return parseOrThrow<SearchResponse>(response);
}

/** Endpoint thống nhất — task bắt buộc phải có trong body (PR-09). */
export async function unifiedSearch(config: ApiClientConfig, body: SearchRequestBody): Promise<SearchResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/search`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
  });
  return parseOrThrow<SearchResponse>(response);
}

/**
 * SSE thật qua fetch + ReadableStream (không dùng EventSource vì cần POST
 * body). Gọi `onEvent` cho từng sự kiện ngay khi dòng `data: ...\n\n` đầy đủ
 * tới — không đợi toàn bộ response như polling giả lập.
 */
export async function searchStream(
  config: ApiClientConfig,
  body: SearchRequestBody,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${normalizedBase(config)}/v1/search/stream`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => ({}));
    throw new ApiError(data?.detail || `Stream request failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.split("\n").find((item) => item.startsWith("data: "));
      if (!line) continue;
      onEvent(JSON.parse(line.slice("data: ".length)) as StreamEvent);
    }
  }
}

// ---- Capabilities / evidence / scenes — PR-03/06 ---------------------------

export async function getCapabilities(config: ApiClientConfig): Promise<CapabilitiesResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/search/capabilities`, { headers: headers(config) });
  return parseOrThrow<CapabilitiesResponse>(response);
}

export async function getEvidence(config: ApiClientConfig, candidateId: string): Promise<EvidencePack> {
  const response = await fetch(`${normalizedBase(config)}/v1/evidence/${encodeURIComponent(candidateId)}`, {
    headers: headers(config),
  });
  return parseOrThrow<EvidencePack>(response);
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

// ---- Search sessions — PR-09 ------------------------------------------------

export async function getSearchSession(config: ApiClientConfig, sessionId: string): Promise<SearchExecutionTrace> {
  const response = await fetch(`${normalizedBase(config)}/v1/search-sessions/${encodeURIComponent(sessionId)}`, {
    headers: headers(config),
  });
  return parseOrThrow<SearchExecutionTrace>(response);
}

export async function replaySearchSession(config: ApiClientConfig, sessionId: string): Promise<SearchResponse> {
  const response = await fetch(
    `${normalizedBase(config)}/v1/search-sessions/${encodeURIComponent(sessionId)}/replay`,
    { method: "POST", headers: headers(config) }
  );
  return parseOrThrow<SearchResponse>(response);
}

// ---- Submissions — PR-08 ----------------------------------------------------

export interface SubmissionBuildBody {
  task: TaskType;
  kis?: SearchResponse["kis"];
  qa?: SearchResponse["qa"];
  trake?: SearchResponse["trake"];
}

export async function buildSubmission(
  config: ApiClientConfig,
  body: SubmissionBuildBody
): Promise<SubmissionBuildResponse> {
  const response = await fetch(`${normalizedBase(config)}/v1/submissions/build`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
  });
  return parseOrThrow<SubmissionBuildResponse>(response);
}

export async function validateSubmission(
  config: ApiClientConfig,
  body: SubmissionBuildBody
): Promise<SubmissionIssue[]> {
  const response = await fetch(`${normalizedBase(config)}/v1/submissions/validate`, {
    method: "POST",
    headers: headers(config),
    body: JSON.stringify(body),
  });
  return parseOrThrow<SubmissionIssue[]>(response);
}

/** Metadata mức video — UI cần `fps` THẬT để quy đổi frame <-> giây.
 *
 * Đo trên corpus hiện tại: V001/V002 chạy 30 fps nhưng **V003 chạy 25 fps**.
 * Giả định 30 cho tất cả thì tua lệch 20% trên V003.
 */
export async function listVideos(config: ApiClientConfig): Promise<VideoMeta[]> {
  const response = await fetch(`${normalizedBase(config)}/v1/videos`, { headers: headers(config) });
  const data = await parseOrThrow<{ videos: VideoMeta[] }>(response);
  return data.videos;
}

/** Keyframe của một video — dùng làm ảnh thay thế khi thiếu file mp4.
 *
 * `storage/raw/videos/` hiện chỉ có L21_V001.mp4, nhưng ảnh keyframe đủ cho cả
 * ba video, nên người chấm vẫn soát được nội dung ở mật độ keyframe.
 */
export async function listVideoFrames(
  config: ApiClientConfig,
  videoId: string
): Promise<VideoFrame[]> {
  const response = await fetch(
    `${normalizedBase(config)}/v1/videos/${encodeURIComponent(videoId)}/frames`,
    { headers: headers(config) }
  );
  const data = await parseOrThrow<{ frames: VideoFrame[] }>(response);
  return data.frames;
}
