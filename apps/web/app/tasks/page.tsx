import Link from "next/link";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

const SPLITS = ["training", "evaluation", "train", "validation"];

export default async function TasksPage({
  searchParams,
}: {
  searchParams: Promise<{ split?: string }>;
}) {
  const { split } = await searchParams;
  let tasks: { task_id: string; split: string; num_train_pairs: number; num_test_pairs: number }[] = [];
  let total = 0;
  let error: string | null = null;
  try {
    const res = await api.listTasks(split, 100);
    tasks = res.tasks;
    total = res.total;
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Tasks {total > 0 && <span className="text-[var(--muted)] text-base">({total})</span>}</h1>
        <div className="flex gap-2 text-sm">
          <Link href="/tasks" className={!split ? "btn" : "text-[var(--muted)] px-2 py-2"}>
            all
          </Link>
          {SPLITS.map((s) => (
            <Link
              key={s}
              href={`/tasks?split=${s}`}
              className={split === s ? "btn" : "text-[var(--muted)] px-2 py-2"}
            >
              {s}
            </Link>
          ))}
        </div>
      </div>

      {error ? (
        <div className="panel p-6 text-sm text-red-400">
          Could not load tasks: {error}. Is the API running and ARC data synced?
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
          {tasks.map((t) => (
            <Link
              key={t.task_id}
              href={`/tasks/${t.task_id}`}
              className="panel p-3 hover:border-white transition-colors"
            >
              <div className="font-mono text-sm">{t.task_id}</div>
              <div className="text-xs text-[var(--muted)] mt-1">
                {t.split} · {t.num_train_pairs} train · {t.num_test_pairs} test
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
