"""Loss tests: finite scalar and backward works on a real batch."""

from __future__ import annotations

import torch

from packages.data.dataset import collate
from packages.model.losses import arc_loss
from packages.model.trm_arc import TRMARCModel
from tests.conftest import requires_arc


@requires_arc
def test_loss_finite_and_backward(small_dataset):
    batch = collate([small_dataset[i] for i in range(2)])
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)
    out = model(
        tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
        segments=batch["segments"], answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"], recursion_steps=2,
    )
    loss, metrics = arc_loss(out, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["num_valid_cells"] > 0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
