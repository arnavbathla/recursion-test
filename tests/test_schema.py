"""Schema tests: real tasks validate, malformed grids are rejected (not coerced)."""

from __future__ import annotations

import pytest

from packages.data.schema import ARCPair, ARCTask, validate_grid_values
from tests.conftest import requires_arc


@requires_arc
def test_real_task_validates(training_tasks):
    task = training_tasks[0]
    assert isinstance(task, ARCTask)
    assert task.num_train_pairs >= 1
    assert task.num_test_pairs >= 1
    for pair in task.train:
        assert pair.output is not None


def test_valid_grid_passes():
    grid = [[0, 1, 2], [3, 4, 5]]
    assert validate_grid_values(grid) == grid


@pytest.mark.parametrize(
    "bad",
    [
        [],                       # empty
        [[]],                     # empty row
        [[0, 1], [0]],            # not rectangular
        [[0, 10]],                # value out of range
        [[0, -1]],                # negative
        [[0, 1.0]],               # float, not int
        [[0, True]],              # bool is not allowed
        [[0] * 31],               # width > 30
    ],
)
def test_invalid_grid_rejected(bad):
    with pytest.raises(ValueError):
        validate_grid_values(bad)


def test_invalid_grid_in_pair_rejected():
    with pytest.raises(ValueError):
        ARCPair(input=[[0, 99]])


def test_train_pair_requires_output():
    with pytest.raises(ValueError):
        ARCTask(
            task_id="x",
            split="training",
            train=[ARCPair(input=[[0]])],  # no output
            test=[ARCPair(input=[[0]])],
        )
