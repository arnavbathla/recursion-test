"""Model forward-pass tests on a real packed batch."""

from __future__ import annotations

import torch

from packages.data.dataset import collate
from packages.model.trm_arc import TRMARCModel
from tests.conftest import requires_arc


def _batch(small_dataset, n=2):
    return collate([small_dataset[i] for i in range(n)])


@requires_arc
def test_forward_shapes_and_depths(small_dataset):
    batch = _batch(small_dataset)
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)
    b = batch["tokens"].shape[0]
    for depth in (1, 4):
        out = model(
            tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
            segments=batch["segments"], answer_positions=batch["answer_positions"],
            attention_mask=batch["attention_mask"], recursion_steps=depth,
        )
        assert out.height_logits.shape == (b, 30)
        assert out.width_logits.shape == (b, 30)
        assert out.cell_logits.shape == (b, 900, 10)
        assert torch.isfinite(out.cell_logits).all()


@requires_arc
def test_return_trace(small_dataset):
    batch = _batch(small_dataset)
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)
    out, trace = model(
        tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
        segments=batch["segments"], answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"], recursion_steps=4, return_trace=True,
    )
    steps = [1, 2, 4]
    assert len(trace) == len(steps)
    for t in trace:
        assert t.cell_logits.shape == out.cell_logits.shape


@requires_arc
def test_state_carry(small_dataset):
    batch = _batch(small_dataset)
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)
    o1 = model(
        tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
        segments=batch["segments"], answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"], recursion_steps=1,
    )
    o2 = model(
        tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
        segments=batch["segments"], answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"], recursion_steps=1,
        latent_state=o1.latent_state.detach(), answer_state=o1.answer_state.detach(),
    )
    assert torch.isfinite(o2.cell_logits).all()
