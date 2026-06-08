"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { ModelRecord, SolveResponse, TaskDetail } from "@/lib/types";
import { ArcGrid } from "./ArcGrid";
import { MetricsCard } from "./MetricsCard";
import { RecursionTrace } from "./RecursionTrace";

const DEPTHS = [1, 2, 4, 8, 16, 32, 64];

export function SolvePanel({ task, models }: { task: TaskDetail; models: ModelRecord[] }) {
  const [modelId, setModelId] = useState(models[0]?.model_id ?? "");
  const [depth, setDepth] = useState(16);
  const [returnTrace, setReturnTrace] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResponse | null>(null);

  const target = task.test[0].output;

  async function runSolve() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.solve({
        task_id: task.task_id,
        model_id: modelId || undefined,
        recursion_steps: depth,
        return_trace: returnTrace,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="panel p-4 flex flex-wrap items-end gap-4">
        <label className="text-sm">
          <div className="text-[var(--muted)] mb-1">Model</div>
          <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {models.length === 0 && <option value="">(no models registered)</option>}
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_id}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm">
          <div className="text-[var(--muted)] mb-1">Recursion depth</div>
          <select value={depth} onChange={(e) => setDepth(Number(e.target.value))}>
            {DEPTHS.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm flex items-center gap-2 pb-1">
          <input
            type="checkbox"
            checked={returnTrace}
            onChange={(e) => setReturnTrace(e.target.checked)}
          />
          return trace
        </label>
        <button className="btn" onClick={runSolve} disabled={loading || !modelId}>
          {loading ? "Solving..." : "Solve"}
        </button>
      </div>

      {error && (
        <div className="panel p-4 border-red-500 text-red-400 text-sm">Error: {error}</div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="flex flex-wrap gap-6 items-start">
            <div className="panel p-3">
              <ArcGrid grid={result.prediction} label="prediction" />
            </div>
            {target && (
              <div className="panel p-3">
                <ArcGrid grid={target} label="target" />
              </div>
            )}
            <MetricsCard result={result} />
          </div>
          <RecursionTrace trace={result.trace} />
          <div className="text-xs text-[var(--muted)]">
            Stored as run{" "}
            <a className="underline" href={`/runs/${result.run_id}`}>
              {result.run_id}
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
