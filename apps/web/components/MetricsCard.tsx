import type { SolveResponse } from "@/lib/types";

function fmt(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined) return "n/a";
  return v.toFixed(digits);
}

export function MetricsCard({ result }: { result: SolveResponse }) {
  const items: [string, string][] = [
    ["Exact match", result.exact_match === null ? "n/a (no target)" : String(result.exact_match)],
    ["Pixel accuracy", fmt(result.pixel_accuracy)],
    ["Shape accuracy", result.shape_accuracy === null ? "n/a" : String(result.shape_accuracy)],
    ["Latency (ms)", fmt(result.latency_ms, 1)],
    ["Recursion steps", String(result.recursion_steps)],
    ["Model", result.model_id],
    ["Checkpoint", result.checkpoint_hash.slice(0, 12)],
    ["Run id", result.run_id],
  ];
  return (
    <div className="panel p-4">
      <h3 className="text-sm font-semibold mb-3 text-[var(--muted)]">Metrics</h3>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
        {items.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <dt className="text-[var(--muted)]">{k}</dt>
            <dd className="font-mono">{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
