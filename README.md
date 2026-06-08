# Recursive ARC Engine

A production-grade, end-to-end system that **trains and serves a TRM/HRM-inspired
latent-recursive neural model on real ARC-AGI-2 data**. The model repeatedly applies
the *same weights* while carrying a persistent latent reasoning state and a
persistent answer state forward — recursion in latent space, not chain-of-thought in
token space. Predictions are checked with a pixel-exact verifier and served through a
real FastAPI + Next.js stack with run logging and recursion-depth evaluation.

---

## 1. What this is

- A real neural ARC solver: `packages/model/trm_arc.py` (`TRMARCModel`) is first-class
  production code, unit-tested, trained on real data, and served for real inference.
- A reproducible lab: data ingestion with hashed manifests, deterministic splits with
  leakage guards, deep-supervision training, recursion-depth evaluation + ablations,
  a model registry, Postgres/SQLite run logging, a web UI, Docker Compose, and tests.

## 2. What this is NOT

- **Not AGI. Not RSI.** It is a small supervised model.
- **Not a hand-coded ARC solver.** There are no symbolic rules, no task-id special
  casing, and no inspection of evaluation answers.
- **Not a GPT/Claude/LLM wrapper.** There is no LLM API dependency for solving ARC.
- **Not a fake demo.** The UI calls the real API; `/v1/solve` runs the real PyTorch
  model; a missing checkpoint returns a real `503`, never a fake success.
- Exact-match scores depend entirely on training quality. The important production
  artifact is the **reproducible recursive-reasoning lab**, not a leaderboard number.

## 3. Architecture

```
official ARC-AGI-2 JSON
  -> ingestion + schema validation (packages/data)
  -> dataset manifest + sha256 hashing (artifacts/)
  -> tokenizer/packer  -> PyTorch Dataset/DataLoader
  -> TRMARCModel (latent + answer state, shared recursive cell)   [+ non-recursive baseline]
  -> training loop (simple | deep-supervision state-carry)
  -> checkpoint registry (artifacts/model_registry.json + DB)
  -> evaluator / verifier (exact-match, pixel, shape, latency, ablations)
  -> FastAPI backend  -> cached CPU/GPU ModelRuntime
  -> Postgres/SQLite run logging   + Redis/RQ async eval worker
  -> Next.js UI (tasks, solve panel, recursion trace, eval dashboard)
```

Primary flow: open UI -> pick a real task -> choose recursion depth (1..64) ->
`POST /v1/solve` -> backend loads the cached model -> returns final grid + intermediate
trace grids + objective metrics -> run is persisted -> UI renders everything.

## 4. Data source & leakage policy

Data is the **official ARC-AGI-2** public set, fetched (not vendored) from
<https://github.com/arcprize/ARC-AGI-2>:

```
data/raw/ARC-AGI-2/data/training      (1000 tasks)
data/raw/ARC-AGI-2/data/evaluation    (120 tasks)
```

Leakage guards (`packages/data/splits.py`):
- Official **training** tasks are split deterministically into `train` (85%) and
  `validation`/`train_holdout` (15%) with `seed=42`.
- The official **evaluation** split is reserved for final evaluation only. Training
  code that requests it raises `RuntimeError` unless `allow_official_eval=True`
  (which the training configs never set).
- No stable identifier (task id, file name) is ever fed to the model. `task_id` is used
  only for logging/DB keys. Model tokens are integers in `[0, 18]`.

## 5. Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+ (for the web UI),
Docker (optional, for the full stack).

```bash
uv sync --extra dev          # create .venv and install Python deps (incl. PyTorch CPU)
cp .env.example .env         # optional; sensible defaults work out of the box
```

Without `DATABASE_URL`, the app uses a local SQLite file (`arc_engine.db`). Docker
Compose points it at Postgres.

## 6. Data sync

```bash
bash scripts/sync_arc_agi_2.sh
```

Clones/pulls ARC-AGI-2, prints file counts, and writes a validated manifest with
per-file sha256 to `artifacts/arc_agi_2_manifest.json`. You can re-run the validator
directly:

```bash
uv run python -m packages.data.ingest_arc data/raw/ARC-AGI-2
```

## 7. Smoke training (real, tiny, fast)

```bash
bash scripts/smoke_train.sh
```

Trains a **tiny** model (`d_model=64`) on **real** ARC tasks for a handful of steps,
saves `checkpoints/smoke.pt`, and registers it. This validates the full pipeline; it is
**not** the claimed final model and will have near-zero exact match.

## 8. Full training

```bash
bash scripts/full_train.sh                      # configs/trm_arc_v1.yaml -> trm_arc_v1
bash scripts/full_train.sh configs/baseline_v1.yaml baseline_v1   # non-recursive baseline
```

Full training (200 epochs, `d_model=256`, deep supervision) is GPU-intensive. It runs on
CPU/MPS but slowly; tune `epochs`/`batch_size` in the config for your hardware. On CUDA,
bf16 mixed precision is enabled automatically.

## 9. Evaluation

```bash
# depth sweep + Markdown report
bash scripts/eval_depths.sh trm_arc_v1 configs/trm_arc_v1.yaml train_holdout

# or directly
uv run python -m packages.eval.evaluate \
  --checkpoint checkpoints/smoke.pt --config configs/smoke.yaml \
  --split train_holdout --depths 1 2 4 8 --out reports/smoke_eval.json
uv run python -m packages.eval.report --eval-json reports/smoke_eval.json --out reports/smoke_report.md

# ablations
uv run python -m packages.eval.ablations \
  --checkpoint checkpoints/smoke.pt --config configs/smoke.yaml --limit 30 \
  --out reports/smoke_ablations.json
```

Per-depth metrics: `exact_match`, `pixel_accuracy`, `shape_accuracy`,
`mean/p50/p95_latency_ms`, `num_tasks`.

## 10. Running the API

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8080
# or: bash scripts/serve_api.sh
```

## 11. Running the web UI

```bash
cd apps/web && npm install && npm run dev      # http://localhost:3000
# or: bash scripts/run_web.sh
```

Set `NEXT_PUBLIC_API_BASE_URL` if the API is not on `http://localhost:8080`.

## 12. Docker Compose

```bash
docker compose up --build
```

Starts **postgres**, **redis**, **api** (`:8080`), **web** (`:3000`), and a CPU
**worker**. The API/worker mount `./data`, `./checkpoints`, and `./artifacts` from the
host, so sync data and train a checkpoint first. GPU workers: see
`infra/Dockerfile.worker` (swap to a CUDA base + CUDA torch wheel, run with `--gpus all`
or k8s `nvidia.com/gpu`).

## 13. API examples

```bash
curl localhost:8080/healthz
curl "localhost:8080/v1/tasks?split=training&limit=5"
curl localhost:8080/v1/tasks/00576224
curl localhost:8080/v1/models

curl -X POST localhost:8080/v1/solve -H 'content-type: application/json' -d '{
  "task_id": "00576224", "model_id": "smoke", "recursion_steps": 16, "return_trace": true
}'

curl -X POST localhost:8080/v1/evaluate -H 'content-type: application/json' -d '{
  "model_id": "smoke", "split": "train_holdout", "depths": [1,2,4,8], "limit": 30
}'

curl localhost:8080/v1/runs/<run_id>
curl localhost:8080/v1/evals
```

Error contract: `404` (unknown task), `400` (recursion_steps out of range), `503`
(checkpoint missing / async queue unavailable). Every solve attempt is logged to `runs`.

## 14. Model architecture

`TRMARCModel` (`packages/model/trm_arc.py`):
- Embeddings: token + row + col + segment + position.
- Two `RecursiveCell`s (RMSNorm + MultiheadAttention + SwiGLU): one for the latent
  reasoning state over the packed task, one for the answer state over the 30x30 canvas.
- `answer_to_latent` feeds an answer summary back into the latent input each step.
- Heads: `height` (30-way), `width` (30-way), `cell` (10 colors per answer slot).
- The answer canvas is a 30x30 row-major grid; the output grid occupies its top-left
  corner. Decoding selects `argmax(height)+1 x argmax(width)+1`.

## 15. Recursion & deep supervision

- **Recursion**: a single shared `RecursiveCell` is applied `recursion_steps` times,
  carrying `latent_state` and `answer_state` forward. Trace readouts are emitted at
  steps {1,2,4,8,16,32,64} ≤ the requested depth.
- **Deep supervision (Mode B)**: the model runs for `num_supervision_steps` rounds; after
  each round the loss is accumulated and `latent_state`/`answer_state` are **detached**
  and carried into the next round. Backprop happens once on the summed loss. This trains
  the model to make iterative, stable refinements. **Mode A** ("simple") instead samples
  a random recursion depth per step and backprops a single forward pass — used for the
  non-recursive baseline.

## 16. Metrics

- **exact_match**: prediction equals target at every cell with identical shape (primary).
- **pixel_accuracy**: fraction of matching cells; `0.0` on any shape mismatch.
- **shape_accuracy**: predicted shape equals target shape.
- **latency**: wall-clock per-example mean/p50/p95, reported per recursion depth.

## 17. Baselines

`packages/model/baseline.py` (`BaselineARCModel`) uses the same packing and heads but a
fixed stack of **distinct** layers in a single forward pass (no weight reuse, no carried
state). Train it via `configs/baseline_v1.yaml` and compare on the Evals dashboard to
isolate the contribution of recursive refinement.

## 18. Reproducibility

- Deterministic seeds (`seed=42`) for splits and training.
- Dataset manifest with per-file sha256 and an aggregate `dataset_hash`.
- Checkpoints embed their config; the registry records checkpoint sha256, config hash,
  dataset hash, and git commit. Eval reports echo all hashes.

## 19. Known limitations

- This repo was developed/verified on a machine **without CUDA**, so the heavy full
  training run is provided as reproducible commands/configs but is not pre-trained here.
  The shipped `checkpoints/smoke.pt` is a tiny real model for pipeline validation — it is
  honest but weak (near-zero exact match expected).
- Tasks whose packed token sequence exceeds `MAX_SEQ_LEN=4096` are explicitly excluded
  from datasets (logged), never silently truncated.
- Latency grows roughly linearly with recursion depth (each step is a full pass).

## 20. Production deployment notes

- **Schema**: local/dev uses `create_all`; production should use Alembic
  (`migrations/`, `uv run alembic upgrade head`).
- **Inference**: the API caches a loaded `ModelRuntime` (no per-request reload). For
  GPU, build the CUDA worker image and set `ARC_DEVICE=cuda`.
- **Async eval**: large evaluations enqueue onto Redis/RQ (`apps/worker/worker.py`);
  if the queue is unavailable the API returns an explicit `503`.
- **Secrets**: configured via env (`.env.example`); no credentials in code. k8s manifests
  in `infra/k8s/` read DB/Redis URLs from a `arc-secrets` secret.

---

### Commands that must work (summary)

```bash
bash scripts/sync_arc_agi_2.sh
uv run pytest
bash scripts/smoke_train.sh
uv run python -m packages.eval.evaluate --checkpoint checkpoints/smoke.pt \
  --config configs/smoke.yaml --split train_holdout --depths 1 2 4 --out reports/smoke_eval.json
bash scripts/full_train.sh
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8080
cd apps/web && npm run dev
docker compose up --build
```
