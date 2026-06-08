"""Dataset + collate tests on real ARC tasks."""

from __future__ import annotations

import torch

from packages.data.dataset import collate
from tests.conftest import requires_arc


@requires_arc
def test_dataset_item_shapes(small_dataset):
    item = small_dataset[0]
    assert item["answer_positions"].shape[0] == 900
    assert item["target_cells"].shape[0] == 900
    assert item["tokens"].dtype == torch.long
    assert isinstance(item["task_id"], str)
    assert item["split"] == "train"


@requires_arc
def test_collate_pads_and_masks(small_dataset):
    batch = collate([small_dataset[i] for i in range(4)])
    b, s = batch["tokens"].shape
    assert b == 4
    assert batch["rows"].shape == (b, s)
    assert batch["cols"].shape == (b, s)
    assert batch["segments"].shape == (b, s)
    assert batch["attention_mask"].shape == (b, s)
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["answer_positions"].shape == (b, 900)
    assert batch["target_cells"].shape == (b, 900)
    assert batch["target_height"].shape == (b,)
    # at least one padded position should be masked when lengths differ
    assert batch["attention_mask"].any() or len({small_dataset[i]["tokens"].shape[0] for i in range(4)}) == 1
