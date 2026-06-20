import type { GenerateResult, JobStatus, PresetMap, RunDetail, RunLog, RunSummary } from "./types";

// Auth token (only required when the server binds a non-loopback address).
// Accept it once via ?token=… in the URL, persist it, then strip it from the
// address bar so it is not left in history or copied links.
const AUTH_TOKEN = ((): string => {
  try {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get("token");
    if (fromUrl) {
      localStorage.setItem("picochat.token", fromUrl);
      url.searchParams.delete("token");
      window.history.replaceState({}, "", url.toString());
      return fromUrl;
    }
    return localStorage.getItem("picochat.token") || "";
  } catch {
    return "";
  }
})();

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(AUTH_TOKEN ? { "X-Picochat-Token": AUTH_TOKEN } : {}),
      ...(init?.headers || {})
    }
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload?.ok === false || payload?.error) {
    throw new Error(payload?.error?.message || payload?.message || `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function listRuns(): Promise<{ runs: RunSummary[] }> {
  return jsonRequest("/api/runs");
}

export function loadRun(name: string): Promise<RunDetail> {
  return jsonRequest(`/api/run?name=${encodeURIComponent(name)}`);
}

export function loadStatus(job?: string): Promise<{ jobs: JobStatus[]; job: JobStatus | null }> {
  const suffix = job ? `?job=${encodeURIComponent(job)}` : "";
  return jsonRequest(`/api/run/status${suffix}`);
}

export function loadRunLog(args: { run?: string; job?: string; limit?: number }): Promise<RunLog> {
  const params = new URLSearchParams();
  if (args.run) params.set("run", args.run);
  if (args.job) params.set("job", args.job);
  if (args.limit) params.set("limit", String(args.limit));
  return jsonRequest(`/api/run/log?${params.toString()}`);
}

// Server-sent-events log stream. EventSource can't set headers, so the token
// (when present) is passed as a query param.
export function runLogStreamUrl(args: { run?: string; job?: string; limit?: number }): string {
  const params = new URLSearchParams();
  if (args.run) params.set("run", args.run);
  if (args.job) params.set("job", args.job);
  if (args.limit) params.set("limit", String(args.limit));
  if (AUTH_TOKEN) params.set("token", AUTH_TOKEN);
  return `/api/run/log/stream?${params.toString()}`;
}

export function startRun(payload: Record<string, any>): Promise<{ job: JobStatus }> {
  return jsonRequest("/api/run/start", { method: "POST", body: JSON.stringify(payload) });
}

export function cancelRun(jobId: string): Promise<{ jobs: JobStatus[]; job: JobStatus | null }> {
  return jsonRequest("/api/run/cancel", { method: "POST", body: JSON.stringify({ job_id: jobId }) });
}

export function archiveRuns(runNames: string[]): Promise<Record<string, any>> {
  return jsonRequest("/api/run/archive", { method: "POST", body: JSON.stringify({ run_names: runNames }) });
}

export function loadPresets(): Promise<{ presets: PresetMap }> {
  return jsonRequest("/api/run/presets");
}

export function importHf(payload: Record<string, unknown>): Promise<Record<string, any>> {
  return jsonRequest("/api/hf/import", { method: "POST", body: JSON.stringify(payload) });
}

export function inspectTuning(payload: Record<string, unknown>): Promise<Record<string, any>> {
  return jsonRequest("/api/tuning/inspect", { method: "POST", body: JSON.stringify(payload) });
}

export function generateSftStarter(payload: Record<string, unknown>): Promise<Record<string, any>> {
  return jsonRequest("/api/sft/starter", { method: "POST", body: JSON.stringify(payload) });
}

export function generateEvalStarter(payload: Record<string, unknown>): Promise<Record<string, any>> {
  return jsonRequest("/api/eval/starter", { method: "POST", body: JSON.stringify(payload) });
}

export function trainHfSft(payload: Record<string, unknown>): Promise<{ job: JobStatus }> {
  return jsonRequest("/api/train/hf-sft", { method: "POST", body: JSON.stringify(payload) });
}

export function serveStatus(): Promise<{ servers: Array<Record<string, any>>; server: Record<string, any> | null }> {
  return jsonRequest("/api/serve/status");
}

export function serveStart(run: string): Promise<{ servers: Array<Record<string, any>>; server: Record<string, any> | null }> {
  return jsonRequest("/api/serve/start", { method: "POST", body: JSON.stringify({ run }) });
}

export function serveStop(run: string): Promise<{ stopped: boolean; run: string }> {
  return jsonRequest("/api/serve/stop", { method: "POST", body: JSON.stringify({ run }) });
}

export function remoteStatus(): Promise<{ modal_available: boolean; modal_script: boolean }> {
  return jsonRequest("/api/remote/status");
}

export function remoteModalStart(payload: Record<string, unknown>): Promise<{ job: JobStatus }> {
  return jsonRequest("/api/remote/modal/start", { method: "POST", body: JSON.stringify(payload) });
}

export function generateText(payload: {
  run: string;
  prompt: string;
  checkpoint?: "base" | "sft";
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  seed?: number;
}): Promise<GenerateResult> {
  return jsonRequest("/api/generate", { method: "POST", body: JSON.stringify(payload) });
}
