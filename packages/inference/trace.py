"""Helpers for working with recursion traces.

The model emits intermediate readouts at canonical steps {1,2,4,8,16,32,64} that are
<= the requested recursion depth. ``decode_trace`` (in decode.py) turns those into
grids; the helper here documents which steps will be present for a given depth.
"""

from __future__ import annotations

TRACE_STEPS = [1, 2, 4, 8, 16, 32, 64]


def expected_trace_steps(recursion_steps: int) -> list[int]:
    """Steps at which a trace readout is emitted for the given recursion depth."""
    return [s for s in TRACE_STEPS if s <= recursion_steps]
