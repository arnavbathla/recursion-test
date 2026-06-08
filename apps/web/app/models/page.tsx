import { TrainingChart } from "@/components/TrainingChart";
import { api } from "@/lib/api";
import type { ModelRecord, TrainingHistory } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function ModelsPage() {
  let models: ModelRecord[] = [];
  let error: string | null = null;
  try {
    models = (await api.listModels()).models;
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }

  const trainingByModel: Record<string, TrainingHistory> = {};
  await Promise.all(
    models.map(async (m) => {
      try {
        trainingByModel[m.model_id] = await api.getTrainingHistory(m.model_id);
      } catch {
        /* no training history for this model */
      }
    })
  );

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Models</h1>
      {error ? (
        <div className="panel p-6 text-sm text-red-400">Could not load models: {error}</div>
      ) : models.length === 0 ? (
        <div className="panel p-6 text-sm text-[var(--muted)]">
          No models registered. Train one: <code>bash scripts/smoke_train.sh</code>
        </div>
      ) : (
        <div className="space-y-3">
          {models.map((m) => (
            <div key={m.model_id} className="panel p-4 text-sm">
              <div className="font-semibold text-base">{m.model_id}</div>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-1 mt-2 font-mono text-xs text-[var(--muted)]">
                <div>checkpoint: {m.checkpoint_path}</div>
                <div>checkpoint_sha256: {m.checkpoint_sha256?.slice(0, 16) ?? "n/a"}</div>
                <div>config_hash: {m.config_hash?.slice(0, 16) ?? "n/a"}</div>
                <div>dataset_hash: {m.dataset_manifest_hash?.slice(0, 16) ?? "n/a"}</div>
                <div>git_commit: {m.git_commit?.slice(0, 12) ?? "n/a"}</div>
                <div>created: {m.created_at}</div>
              </dl>
              {trainingByModel[m.model_id] && (
                <div className="mt-4">
                  <TrainingChart training={trainingByModel[m.model_id]} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
