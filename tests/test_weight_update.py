"""Verify that training actually updates model weights (not a no-op)."""

from __future__ import annotations

import torch

from packages.data.dataset import ARCDataset, collate
from packages.model.losses import arc_loss
from packages.model.trm_arc import TRMARCModel
from packages.training.weight_audit import audit_weight_updates, state_dict_sha256
from tests.conftest import requires_arc


@requires_arc
def test_param_changes_after_step(training_tasks):
    dataset = ARCDataset(training_tasks[:4], "train")
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    hash_before = state_dict_sha256(model)

    batch = collate([dataset[i] for i in range(2)])
    out = model(
        tokens=batch["tokens"], rows=batch["rows"], cols=batch["cols"],
        segments=batch["segments"], answer_positions=batch["answer_positions"],
        attention_mask=batch["attention_mask"], recursion_steps=2,
    )
    loss, _ = arc_loss(out, batch)
    loss.backward()
    opt.step()

    after = model.state_dict()
    changed = sum(1 for k in before if not torch.equal(before[k], after[k].detach()))
    assert changed > 0, "no parameters changed after an optimizer step"
    assert state_dict_sha256(model) != hash_before


@requires_arc
def test_weight_audit_reports_change():
    from packages.common.config import load_yaml_config

    report = audit_weight_updates(load_yaml_config("configs/smoke.yaml"), steps=3)
    assert report["weights_changed"] is True
    assert report["num_tensors_changed"] > 0
    assert report["total_weight_update_l2"] > 0.0
    assert report["state_dict_hash_changed"] is True
