import Link from "next/link";
import { api } from "@/lib/api";
import { ArcGrid } from "@/components/ArcGrid";
import { RecursionTrace } from "@/components/RecursionTrace";
import type { RunDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function RunPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = await params;
  let run: RunDetail | null = null;
  let error: string | null = null;
  try {
    run = await api.getRun(runId);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  if (error || !run) {
    return <div className="panel p-6 text-sm text-red-400">Could not load run {runId}: {error}</div>;
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold font-mono">{run.run_id}</h1>
        <div className="text-sm text-[var(--muted)]">
          task{" "}
          <Link href={`/tasks/${run.task_id}`} className="underline">
            {run.task_id}
          </Link>{" "}
          · model {run.model_id} · depth {run.recursion_steps}
        </div>
      </div>

      {run.error ? (
        <div className="panel p-6 text-sm text-red-400">This run errored: {run.error}</div>
      ) : (
        <>
          <div className="flex flex-wrap gap-6">
            {run.prediction && (
              <div className="panel p-3">
                <ArcGrid grid={run.prediction} label="prediction" />
              </div>
            )}
            <div className="panel p-4 text-sm">
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2">
                <dt className="text-[var(--muted)]">exact_match</dt>
                <dd>{String(run.exact_match)}</dd>
                <dt className="text-[var(--muted)]">pixel_accuracy</dt>
                <dd>{run.pixel_accuracy ?? "n/a"}</dd>
                <dt className="text-[var(--muted)]">shape_accuracy</dt>
                <dd>{String(run.shape_accuracy)}</dd>
                <dt className="text-[var(--muted)]">latency_ms</dt>
                <dd>{run.latency_ms ?? "n/a"}</dd>
                <dt className="text-[var(--muted)]">created_at</dt>
                <dd>{run.created_at}</dd>
              </dl>
            </div>
          </div>
          {run.trace && <RecursionTrace trace={run.trace} />}
        </>
      )}
    </div>
  );
}
