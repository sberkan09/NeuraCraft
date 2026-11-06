// Talking to the engine (`NeuraCraft serve`, see Includes/Server.h).

export interface WorkflowField {
  name: string;
  type: string; // int, float, bool, string, text, enum, clip, clips, mesh, model, loras, file, dir, strings, ints, keyframes
  doc: string;
  required: boolean;
  default?: unknown;
  model?: string;
  options?: string[];
  min?: number;
  max?: number;
}

export interface WorkflowOp {
  name: string;
  title: string;
  group: string;
  doc: string;
  output: boolean;
  makes?: string; // "clip", "mesh" or "any" (load: by the file)
  fields: WorkflowField[];
}

export interface Schema {
  version: number;
  ops: WorkflowOp[];
}

export interface ModelInfo {
  path: string;
  kind: string;
  name: string;
  bytes: number;
}

export interface EngineInfo {
  name: string;
  api: number;
  device: string;
  gpu: string;
  gpuAttention: boolean;
  threads: number;
  dataDir: string;
  baseDir: string;
}

export interface JobOutput {
  step: number;
  op: string;
  clip: string;
  frames: number;
  width: number;
  height: number;
  seconds: number;
  url: string;
}

export type JobState = "queued" | "running" | "done" | "failed" | "cancelled";

export interface Job {
  id: number;
  state: JobState;
  step: number;
  steps: number;
  op: string;
  stage: string;
  fraction: number;
  error: string;
  queued: number;
  started: number;
  finished: number;
  previews: number;
  clipsKept: boolean;
  outputs: JobOutput[];
}

export type Step = { op: string; out?: string; ui?: { x: number; y: number }; label?: string; [field: string]: unknown };
export interface Workflow {
  steps: Step[];
  cache?: boolean;
  [key: string]: unknown;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { error: text };
  }
  if (!res.ok) {
    const b = body as { error?: string; problems?: string[] } | null;
    const err = new Error(b?.error ?? `${res.status} ${res.statusText}`) as Error & { problems?: string[] };
    err.problems = b?.problems;
    throw err;
  }
  return body as T;
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export interface CacheInfo {
  enabled: boolean;
  entries: number;
  bytes: number;
  hits: number;
}

export const api = {
  info: () => request<EngineInfo>("/api/info"),
  schema: () => request<Schema>("/api/ops"),
  models: () => request<ModelInfo[]>("/api/models"),
  jobs: () => request<Job[]>("/api/jobs"),
  check: (wf: Workflow) => post<{ problems: string[] }>("/api/check", wf),
  run: (wf: Workflow) => post<{ job: number }>("/api/run", wf),
  cancel: (id: number) => post<unknown>(`/api/jobs/${id}/cancel`, {}),
  // Step results the engine keeps for reuse.
  cache: () => request<CacheInfo>("/api/cache"),
  clearCache: () => post<CacheInfo>("/api/cache/clear", {}),
  // A picture the user picked, as PNG, saved by the engine; returns its path for workflows.
  upload: (name: string, pngBase64: string) => post<{ path: string }>("/api/upload", { name, data: pngBase64 }),
};

export const frameUrl = (job: number, clip: string, frame: number, max = 0) =>
  `/api/jobs/${job}/clips/${encodeURIComponent(clip)}/${frame}.jpg${max ? `?max=${max}` : ""}`;

export type EngineEvent =
  | { type: "job"; data: Job }
  | { type: "progress"; data: { job: number; step: number; steps: number; op: string; stage: string; fraction: number } }
  | { type: "preview"; data: { job: number; step: number; url: string } }
  | { type: "output"; data: { job: number; step: number; op: string; clip: string; frames: number; seconds: number; url: string } }
  | { type: "log"; data: { job: number; line: string } };

// Server-sent events; reconnects on its own (EventSource), reports connection state.
export function subscribe(onEvent: (e: EngineEvent) => void, onState: (connected: boolean) => void): () => void {
  const es = new EventSource("/api/events");
  es.onopen = () => onState(true);
  es.onerror = () => onState(false);
  for (const type of ["job", "progress", "preview", "output", "log"] as const) {
    es.addEventListener(type, (m) => {
      try {
        onEvent({ type, data: JSON.parse((m as MessageEvent).data) } as EngineEvent);
      } catch {
        /* ignore malformed events */
      }
    });
  }
  return () => es.close();
}

// Any picture the browser can show (JPEG, WebP, ...) -> PNG base64 (the engine reads PNG).
export async function fileToPngBase64(file: Blob, maxSide = 2048): Promise<{ data: string; width: number; height: number }> {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale)), h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, w, h);
  return { data: canvas.toDataURL("image/png").split(",")[1], width: w, height: h };
}

export const formatBytes = (n: number) => (n >= 1 << 30 ? `${(n / (1 << 30)).toFixed(1)} GB` : `${Math.round(n / (1 << 20))} MB`);
