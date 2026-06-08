import Link from "next/link";
import { api } from "@/lib/api";
import { EvalChart, type Series } from "@/components/EvalChart";

export const dynamic = "force-dynamic";

export default async function Home() {
  let series: Series[] = [];
  let evalError: string | null = null;
  try {
    const { evals } = await api.listEvals();
    series = evals
      .filter((e) => e.metrics?.depths)
      .slice(0, 4)
      .map((e) => ({ name: `${e.model_id} (${e.split})`, depths: e.metrics!.depths! }));
  } catch (e) {
    evalError = e instanceof Error ? e.message : String(e);
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-2">Recursive ARC Engine</h1>
        <p className="text-[var(--muted)] max-w-2xl">
          A small neural model with persistent latent state and repeated recursive
          refinement, trained on real ARC-AGI-2 tasks and served through a real API.
          The same weights are applied repeatedly while a hidden reasoning state and an
          answer state are carried forward, then read out into a pixel-exact grid.
        </p>
        <div className="mt-4 flex gap-3">
          <Link href="/tasks" className="btn">
            Browse tasks
          </Link>
          <Link href="/evals" className="btn">
            View evals
          </Link>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">Accuracy vs recursion depth</h2>
        {series.length > 0 ? (
          <EvalChart series={series} metric="exact_match" title="Exact match vs depth" />
        ) : (
          <div className="panel p-6 text-sm text-[var(--muted)]">
            {evalError
              ? `Could not reach the API (${evalError}). Start it with: uv run uvicorn apps.api.main:app --port 8080`
              : "No evaluations recorded yet. Run an evaluation from the Evals page or via POST /v1/evaluate."}
          </div>
        )}
      </section>

      <section className="text-xs text-[var(--muted)] max-w-2xl border-t border-[var(--border)] pt-4">
        Not AGI. Not RSI. Not a hand-coded ARC solver. Exact-match scores depend on
        training quality. The artifact is a reproducible recursive-reasoning lab.
      </section>
    </div>
  );
}
