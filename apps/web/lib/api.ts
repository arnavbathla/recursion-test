import type {
  EvalJob,
  ModelRecord,
  RunDetail,
  SolveResponse,
  TaskDetail,
  TaskSummary,
  TrainingHistory,
} from "./types";

// Server components read API_BASE_URL; the browser reads NEXT_PUBLIC_API_BASE_URL.
export const API_BASE =
  (typeof window === "undefined"
    ? process.env.API_BASE_URL
    : process.env.NEXT_PUBLIC_API_BASE_URL) ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "http://localhost:8080";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => http<{ ok: boolean }>("/healthz"),
  listTasks: (split?: string, limit = 60, offset = 0) =>
    http<{ total: number; tasks: TaskSummary[] }>(
      `/v1/tasks?limit=${limit}&offset=${offset}${split ? `&split=${split}` : ""}`
    ),
  getTask: (taskId: string) => http<TaskDetail>(`/v1/tasks/${taskId}`),
  listModels: () => http<{ models: ModelRecord[] }>("/v1/models"),
  getTrainingHistory: (modelId: string) =>
    http<TrainingHistory>(`/v1/models/${modelId}/training`),
  solve: (body: {
    task_id: string;
    model_id?: string;
    recursion_steps: number;
    return_trace: boolean;
  }) => http<SolveResponse>("/v1/solve", { method: "POST", body: JSON.stringify(body) }),
  listEvals: () => http<{ evals: EvalJob[] }>("/v1/evals"),
  getRun: (runId: string) => http<RunDetail>(`/v1/runs/${runId}`),
};
