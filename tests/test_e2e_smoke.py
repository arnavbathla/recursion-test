"""End-to-end smoke: real data -> tiny model -> one train step -> solve -> DB log."""

from __future__ import annotations

import torch

from packages.common.db import Run, init_db, session_scope
from packages.data.dataset import ARCDataset, collate
from packages.eval.verifier import validate_grid
from packages.inference.solver import solve_task
from packages.model.losses import arc_loss
from packages.model.trm_arc import TRMARCModel
from tests.conftest import requires_arc


@requires_arc
def test_e2e_train_solve_log(training_tasks):
    tasks = training_tasks[:4]
    dataset = ARCDataset(tasks, "train")
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)

    # One real training step on real ARC data.
    batch = collate([dataset[i] for i in range(min(2, len(dataset)))])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(
        tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
        segments=batch["segments"], answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"], recursion_steps=2,
    )
    loss, _ = arc_loss(out, batch)
    loss.backward()
    opt.step()

    # Solve a real task and check the prediction is a valid grid with a trace.
    result = solve_task(model, tasks[0], recursion_steps=4, return_trace=True, device="cpu")
    assert validate_grid(result.prediction)
    assert [t["step"] for t in result.trace] == [1, 2, 4]
    assert result.exact_match in (True, False)  # real target exists -> computed

    # Run logging into the (SQLite test) DB.
    init_db()
    run_id = "run_test_e2e"
    with session_scope() as session:
        session.merge(
            Run(
                id=run_id,
                task_id=tasks[0].task_id,
                model_id="test-tiny",
                recursion_steps=4,
                prediction_json=result.prediction,
                trace_json=result.trace,
                exact_match=result.exact_match,
                pixel_accuracy=result.pixel_accuracy,
                shape_accuracy=result.shape_accuracy,
                latency_ms=result.latency_ms,
            )
        )
    with session_scope() as session:
        stored = session.get(Run, run_id)
        assert stored is not None
        assert stored.task_id == tasks[0].task_id
        assert stored.prediction_json is not None
