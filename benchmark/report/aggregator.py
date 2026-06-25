"""Combine per-preset rows into summary stats: cross-preset deltas and verdicts
(SPEC §Metrics, §Verdict generation rule).

In preset mode the baseline is `production` (the fast config real serving uses);
`det_overhead_pct`, `memory_overhead_mb`, and `slowdown_factor` are measured
strict-vs-production — exactly the headline contrast the SPEC defines.
"""

from __future__ import annotations

import math


def _band(overhead_pct: float) -> str:
    a = abs(overhead_pct)
    if a < 2.0:
        return "low"
    if a <= 10.0:
        return "moderate"
    return "high"


def _verdict(row) -> str:
    r = row.determinism_result
    if r == "BIT_EXACT":
        band = _band(row.det_overhead_pct or 0.0)
        return f"Deterministic, {band} cost"
    if r == "OUTPUT_STABLE":
        return "Stable tokens, not bit-exact"
    if r == "DRIFTED":
        if row.requested_det_level == "NONE":
            return "Baseline is not reproducible"
        return "Determinism not achieved despite controls"
    if r == "UNSUPPORTED":
        return "Full controls unavailable"
    if r == "ERROR":
        return "Nondeterministic op rejected under FULL_DET"
    return ""


def finalize(results, cfg) -> None:
    """Fill derived fields on each CellRow in place."""
    for mr in results:
        if mr.skipped or not mr.rows:
            continue
        baseline = next((r for r in mr.rows if r.preset == "production"), mr.rows[0])
        base_p50 = baseline.latency_p50_ms
        base_mem = baseline.memory_peak_mb

        for row in mr.rows:
            # Vision throughput (SPEC §Vision-Specific).
            if row.model_class == "vision" and row.latency_p50_ms > 0:
                row.throughput_img_s = cfg.batch_size / (row.latency_p50_ms / 1000.0)

            if base_p50 and not math.isnan(base_p50) and base_p50 > 0:
                row.slowdown_factor = row.latency_p50_ms / base_p50
                row.det_overhead_pct = (row.latency_p50_ms / base_p50 - 1.0) * 100.0
            if base_mem is not None and row.memory_peak_mb is not None:
                row.memory_overhead_mb = row.memory_peak_mb - base_mem

            row.verdict = _verdict(row)
