import { api } from "@/lib/api";
import { EvalChart, type Series } from "@/components/EvalChart";

export const dynamic = "force-dynamic";

export default async function EvalsPage() {
  let evals: Awaited<ReturnType<typeof api.listEvals>>["evals"] = [];
  let error: string | null = null;
  try {
    evals = (await api.listEvals()).evals;
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  const withDepths = evals.filter((e) => e.metrics?.depths);
  const accuracySeries: Series[] = withDepths.map((e) => ({
    name: `${e.model_id} (${e.split})`,
    depths: e.metrics!.depths!,
  }));

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Evaluations</h1>

      {error && (
        <div className="panel p-6 text-sm text-red-400">Could not load evals: {error}</div>
      )}

      {accuracySeries.length > 0 ? (
        <div className="grid md:grid-cols-2 gap-6">
          <EvalChart series={accuracySeries} metric="exact_match" title="Exact match vs depth" />
          <EvalChart series={accuracySeries} metric="pixel_accuracy" title="Pixel accuracy vs depth" />
          <EvalChart series={accuracySeries} metric="mean_latency_ms" title="Latency (ms) vs depth" />
          <EvalChart series={accuracySeries} metric="shape_accuracy" title="Shape accuracy vs depth" />
        </div>
      ) : (
        !error && (
          <div className="panel p-6 text-sm text-[var(--muted)]">
            No completed evaluations yet. Trigger one with:
            <pre className="mt-2 text-xs">
              curl -X POST $API/v1/evaluate -d &apos;{"{"}&quot;model_id&quot;:&quot;trm_arc_v1&quot;,&quot;split&quot;:&quot;train_holdout&quot;,&quot;depths&quot;:[1,2,4,8],&quot;limit&quot;:30{"}"}&apos;
            </pre>
          </div>
        )
      )}

      <div>
        <h2 className="text-lg font-semibold mb-3">Eval jobs</h2>
        <div className="panel p-4 text-sm">
          {evals.length === 0 ? (
            <div className="text-[var(--muted)]">none</div>
          ) : (
            <table className="w-full text-left">
              <thead className="text-[var(--muted)]">
                <tr>
                  <th className="py-1">job</th>
                  <th>model</th>
                  <th>split</th>
                  <th>status</th>
                </tr>
              </thead>
              <tbody>
                {evals.map((e) => (
                  <tr key={e.eval_job_id} className="border-t border-[var(--border)]">
                    <td className="py-1 font-mono text-xs">{e.eval_job_id}</td>
                    <td>{e.model_id}</td>
                    <td>{e.split}</td>
                    <td>{e.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
