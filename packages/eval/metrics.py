"""Metric aggregation helpers for evaluation."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass


@dataclass
class DepthMetrics:
    exact_match: float
    pixel_accuracy: float
    shape_accuracy: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    num_tasks: int

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def aggregate(
    exact: list[bool],
    pixel: list[float],
    shape: list[bool],
    latencies_ms: list[float],
) -> DepthMetrics:
    n = len(exact)
    return DepthMetrics(
        exact_match=(sum(1 for e in exact if e) / n) if n else 0.0,
        pixel_accuracy=(sum(pixel) / n) if n else 0.0,
        shape_accuracy=(sum(1 for s in shape if s) / n) if n else 0.0,
        mean_latency_ms=(statistics.fmean(latencies_ms)) if latencies_ms else 0.0,
        p50_latency_ms=percentile(latencies_ms, 0.50),
        p95_latency_ms=percentile(latencies_ms, 0.95),
        num_tasks=n,
    )
