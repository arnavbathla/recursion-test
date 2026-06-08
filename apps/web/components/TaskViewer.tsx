import type { TaskDetail } from "@/lib/types";
import { ArcGrid } from "./ArcGrid";

export function TaskViewer({ task }: { task: TaskDetail }) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-semibold mb-3 text-[var(--muted)]">
          Training examples ({task.train.length})
        </h3>
        <div className="flex flex-wrap gap-6">
          {task.train.map((pair, i) => (
            <div key={i} className="panel p-3 flex items-end gap-3">
              <ArcGrid grid={pair.input} label={`in #${i + 1}`} />
              <span className="text-[var(--muted)] pb-4">to</span>
              {pair.output && <ArcGrid grid={pair.output} label={`out #${i + 1}`} />}
            </div>
          ))}
        </div>
      </div>
      <div>
        <h3 className="text-sm font-semibold mb-3 text-[var(--muted)]">Test input</h3>
        <div className="panel p-3 inline-block">
          <ArcGrid grid={task.test[0].input} label="test input" />
        </div>
      </div>
    </div>
  );
}
