import type { TraceItem } from "@/lib/types";
import { ArcGrid } from "./ArcGrid";

export function RecursionTrace({ trace }: { trace: TraceItem[] }) {
  if (!trace || trace.length === 0) {
    return (
      <div className="text-xs text-[var(--muted)]">
        No trace returned (enable &quot;return trace&quot; before solving).
      </div>
    );
  }
  return (
    <div>
      <h3 className="text-sm font-semibold mb-3 text-[var(--muted)]">
        Recursion trace (intermediate answer states)
      </h3>
      <div className="flex flex-wrap gap-4">
        {trace.map((item) => (
          <div key={item.step} className="panel p-3">
            <ArcGrid grid={item.grid} cell={12} label={`step ${item.step} (${item.height}x${item.width})`} />
          </div>
        ))}
      </div>
    </div>
  );
}
