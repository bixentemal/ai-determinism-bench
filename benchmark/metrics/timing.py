"""Latency summary statistics (SPEC §Universal metrics, §Statistical Protocol)."""

from __future__ import annotations

import numpy as np


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_p99_ms": float(np.percentile(arr, 99)),
    }
