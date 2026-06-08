"""Tiny evaluation over a couple real tasks returns metrics."""

from __future__ import annotations

from packages.data.dataset import ARCDataset
from packages.eval.evaluate import evaluate_depths
from packages.model.trm_arc import TRMARCModel
from tests.conftest import requires_arc


@requires_arc
def test_evaluate_depths_returns_metrics(training_tasks):
    dataset = ARCDataset(training_tasks[:2], "train")
    model = TRMARCModel(d_model=64, n_heads=4, mlp_dim=128)
    results = evaluate_depths(model, dataset, depths=[1, 2], device="cpu")
    assert set(results.keys()) == {"1", "2"}
    for depth_metrics in results.values():
        assert "exact_match" in depth_metrics
        assert "pixel_accuracy" in depth_metrics
        assert "mean_latency_ms" in depth_metrics
        assert depth_metrics["num_tasks"] >= 1
        assert 0.0 <= depth_metrics["pixel_accuracy"] <= 1.0
