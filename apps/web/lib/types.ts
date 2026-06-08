export type Grid = number[][];

export interface TaskSummary {
  task_id: string;
  split: string;
  num_train_pairs: number;
  num_test_pairs: number;
}

export interface TaskPair {
  input: Grid;
  output: Grid | null;
}

export interface TaskDetail {
  task_id: string;
  split: string;
  num_train_pairs: number;
  num_test_pairs: number;
  train: TaskPair[];
  test: TaskPair[];
}

export interface TraceItem {
  step: number;
  height: number;
  width: number;
  grid: Grid;
}

export interface SolveResponse {
  run_id: string;
  task_id: string;
  model_id: string;
  recursion_steps: number;
  prediction: Grid;
  trace: TraceItem[];
  exact_match: boolean | null;
  pixel_accuracy: number | null;
  shape_accuracy: boolean | null;
  latency_ms: number;
  checkpoint_hash: string;
  dataset_hash: string | null;
}

export interface ModelRecord {
  model_id: string;
  checkpoint_path: string;
  checkpoint_sha256: string | null;
  config_hash: string | null;
  dataset_manifest_hash: string | null;
  git_commit: string | null;
  created_at: string;
  metrics?: Record<string, unknown> | null;
}

export interface DepthMetrics {
  exact_match: number;
  pixel_accuracy: number;
  shape_accuracy: number;
  mean_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  num_tasks: number;
}

export interface EvalJob {
  eval_job_id: string;
  model_id: string;
  split: string;
  depths: number[] | null;
  status: string;
  metrics: { depths?: Record<string, DepthMetrics> } | null;
  error: string | null;
  created_at: string | null;
}

export interface TrainingEpoch {
  epoch: number;
  train_loss: number;
  grad_norm?: number;
  weight_update_l2?: number;
  param_global_norm?: number;
  steps: number;
  seconds?: number;
  val_exact_match?: number;
  val_pixel_accuracy?: number;
}

export interface TrainingHistory {
  model_id: string;
  best_metric?: number | null;
  total_weight_movement_l2?: number;
  weights_changed?: boolean;
  total_steps?: number;
  history: TrainingEpoch[];
}

export interface RunDetail {
  run_id: string;
  task_id: string;
  model_id: string;
  recursion_steps: number;
  prediction: Grid | null;
  trace: TraceItem[] | null;
  exact_match: boolean | null;
  pixel_accuracy: number | null;
  shape_accuracy: boolean | null;
  latency_ms: number | null;
  error: string | null;
  created_at: string | null;
}
