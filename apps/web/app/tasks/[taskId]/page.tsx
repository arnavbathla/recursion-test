import Link from "next/link";
import { api } from "@/lib/api";
import { SolvePanel } from "@/components/SolvePanel";
import { TaskViewer } from "@/components/TaskViewer";
import type { ModelRecord, TaskDetail } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function TaskPage({ params }: { params: Promise<{ taskId: string }> }) {
  const { taskId } = await params;
  let task: TaskDetail | null = null;
  let models: ModelRecord[] = [];
  let error: string | null = null;
  try {
    task = await api.getTask(taskId);
    models = (await api.listModels()).models;
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  if (error || !task) {
    return (
      <div className="panel p-6 text-sm text-red-400">
        Could not load task {taskId}: {error}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <Link href="/tasks" className="text-sm text-[var(--muted)] hover:text-white">
          back to tasks
        </Link>
        <h1 className="text-2xl font-bold font-mono mt-1">{task.task_id}</h1>
        <div className="text-sm text-[var(--muted)]">split: {task.split}</div>
      </div>
      <TaskViewer task={task} />
      <div>
        <h2 className="text-lg font-semibold mb-3">Solve</h2>
        <SolvePanel task={task} models={models} />
      </div>
    </div>
  );
}
