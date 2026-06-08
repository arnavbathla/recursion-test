"""Tokenizer tests on real ARC tasks."""

from __future__ import annotations

from packages.data.tokenizer import (
    ANSWER,
    MAX_CELLS,
    PAD,
    pack_task,
)
from tests.conftest import requires_arc


@requires_arc
def test_pack_real_task(training_tasks):
    packed = pack_task(training_tasks[0])
    assert packed.answer_positions.shape[0] == MAX_CELLS
    assert int(packed.answer_positions.min()) >= 0
    assert int(packed.answer_positions.max()) < packed.seq_len
    # answer slots are PAD tokens in the packed sequence
    for pos in packed.answer_positions.tolist():
        assert int(packed.tokens[pos]) == PAD


@requires_arc
def test_answer_marker_precedes_slots(training_tasks):
    packed = pack_task(training_tasks[0])
    first_slot = int(packed.answer_positions[0])
    assert int(packed.tokens[first_slot - 1]) == ANSWER


@requires_arc
def test_no_task_id_in_tokens(training_tasks):
    # task_id is a string; tokens are integers in [0, 18]. Confirm vocab bounds.
    packed = pack_task(training_tasks[0])
    assert int(packed.tokens.min()) >= 0
    assert int(packed.tokens.max()) <= 18
